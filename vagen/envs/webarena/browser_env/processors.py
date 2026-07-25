import json
import pkgutil
import re
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Any, Optional, TypedDict, Union
from urllib.parse import urljoin, urlparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import playwright
import requests
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import CDPSession, Page, ViewportSize
from .html_tools.fetch import get_parsed_html

from browser_env.constants import (
    ASCII_CHARSET,
    FREQ_UNICODE_CHARSET,
    IGNORED_ACTREE_PROPERTIES,
    INJECTED_ATTR_NAME,
    UTTERANCE_MAX_LENGTH,
    BID_ATTR,
    DATA_REGEXP,
    IN_VIEWPORT_RATIO_THRESHOLD,
)

from .utils import (
    AccessibilityTree,
    AccessibilityTreeNode,
    BrowserConfig,
    BrowserInfo,
    DOMNode,
    DOMTree,
    Observation,
    png_bytes_to_numpy,
)


def remove_unicode(input_string):
    # Define a regex pattern to match Unicode characters
    unicode_pattern = re.compile(r"[^\x00-\x7F]+")

    # Use the pattern to replace Unicode characters with an empty string
    cleaned_string = unicode_pattern.sub("", input_string)

    return cleaned_string


class ObservationProcessor:
    def process(self, page: Page) -> Observation:
        raise NotImplementedError


class ObservationMetadata(TypedDict):
    obs_nodes_info: dict[str, Any]


def create_empty_metadata() -> ObservationMetadata:
    return {
        "obs_nodes_info": {},
    }


def extract_data_items_from_aria(string: str) -> tuple[list[str], str]:
    """
    Utility function to extract temporary data stored in the "aria-roledescription" attribute of a node
    """

    match = DATA_REGEXP.fullmatch(string)
    if not match:
        return [], string

    groups = match.groups()
    data_items = groups[:-1]
    original_aria = groups[-1]
    return data_items, original_aria


class TextObervationProcessor(ObservationProcessor):
    def __init__(
        self,
        observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        captioning_fn=None,
    ):
        self.observation_type = observation_type
        self.current_viewport_only = current_viewport_only
        self.viewport_size = viewport_size
        self.observation_tag = "text"
        self.meta_data = (
            create_empty_metadata()
        )  # use the store meta data of this observation type

        if self.observation_type in [
            "accessibility_tree_with_captioner",
            "image_som",
        ]:
            self.captioning_fn = captioning_fn
            # Cache captions.
            self.url2caption = {}

    def fetch_browser_info(
        self,
        page: Page,
    ) -> BrowserInfo:
        # extract domtree
        client = page.context.new_cdp_session(page)
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )
        client.detach()

        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds
        # add union bound placeholder
        tree["documents"][0]["layout"]["unionBounds"] = [None for _ in bounds]

        # extract browser info
        win_upper_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_upper_bound + win_height
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        config: BrowserConfig = {
            "win_upper_bound": win_upper_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        return info
    
    @staticmethod
    def get_bounding_client_rect(
        client: CDPSession, backend_node_id: str
    ) -> dict[str, Any]:
        try:
            remote_object = client.send(
                "DOM.resolveNode", {"backendNodeId": int(backend_node_id)}
            )
            remote_object_id = remote_object["object"]["objectId"]
            response = client.send(
                "Runtime.callFunctionOn",
                {
                    "objectId": remote_object_id,
                    "functionDeclaration": """
                        function() {
                            if (this.nodeType == 3) {
                                var range = document.createRange();
                                range.selectNode(this);
                                var rect = range.getBoundingClientRect().toJSON();
                                range.detach();
                                return rect;
                            } else {
                                return this.getBoundingClientRect().toJSON();
                            }
                        }
                    """,
                    "returnByValue": True,
                },
            )
            return response
        except Exception as e:
            return {"result": {"subtype": "error"}}

    @staticmethod
    def get_element_in_viewport_ratio(
        elem_left_bound: float,
        elem_top_bound: float,
        width: float,
        height: float,
        config: BrowserConfig,
    ) -> float:
        elem_right_bound = elem_left_bound + width
        elem_lower_bound = elem_top_bound + height

        win_left_bound = 0
        win_right_bound = config["win_width"]
        win_top_bound = 0
        win_lower_bound = config["win_height"]

        # Compute the overlap in x and y axes
        overlap_width = max(
            0,
            min(elem_right_bound, win_right_bound)
            - max(elem_left_bound, win_left_bound),
        )
        overlap_height = max(
            0,
            min(elem_lower_bound, win_lower_bound) - max(elem_top_bound, win_top_bound),
        )

        # Compute the overlap area
        ratio = overlap_width * overlap_height / width * height
        return ratio

    def fetch_page_html(
        self,
        info: BrowserInfo,
        page: Page,
        current_viewport_only: bool,
    ) -> DOMTree:
        # adopted from [natbot](https://github.com/nat/natbot)
        tree = info["DOMTree"]
        strings = tree["strings"]
        document = tree["documents"][0]
        nodes = document["nodes"]

        # make a dom tree that is easier to navigate
        dom_tree: DOMTree = []
        graph = defaultdict(list)
        client = page.context.new_cdp_session(page)
        for node_idx in range(len(nodes["nodeName"])):
            cur_node: DOMNode = {
                "nodeId": "",
                "nodeType": "",
                "nodeName": "",
                "nodeValue": "",
                "attributes": "",
                "backendNodeId": "",
                "parentId": "",
                "childIds": [],
                "cursor": 0,
                "union_bound": None,
            }

            node_type_idx = nodes["nodeType"][node_idx]
            node_type = "generic"
            if node_type_idx >= 0 and node_type_idx < len(strings):
                node_type = strings[node_type_idx]

            node_name = strings[nodes["nodeName"][node_idx]]

            node_value_idx = nodes["nodeValue"][node_idx]
            node_value = ""
            if node_value_idx >= 0 and node_value_idx < len(strings):
                node_value = " ".join(strings[node_value_idx].split())

            node_attributes = [strings[i] for i in nodes["attributes"][node_idx]]
            node_attributes_str = ""
            for i in range(0, len(node_attributes), 2):
                a = node_attributes[i]
                b = node_attributes[i + 1]
                b = " ".join(b.split())
                node_attributes_str += f'{a}="{b}" '
            node_attributes_str = node_attributes_str.strip()

            cur_node["nodeId"] = str(node_idx)
            cur_node["nodeType"] = node_type
            cur_node["nodeName"] = node_name
            cur_node["nodeValue"] = node_value
            cur_node["attributes"] = node_attributes_str
            cur_node["backendNodeId"] = str(nodes["backendNodeId"][node_idx])
            cur_node["parentId"] = str(nodes["parentIndex"][node_idx])

            if cur_node["parentId"] != "-1":
                graph[cur_node["parentId"]].append(str(cur_node["nodeId"]))

            # get the bound
            if cur_node["parentId"] == "-1":
                cur_node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
            else:
                response = self.get_bounding_client_rect(
                    client, cur_node["backendNodeId"]
                )
                if response.get("result", {}).get("subtype", "") == "error":
                    cur_node["union_bound"] = None
                else:
                    x = response["result"]["value"]["x"]
                    y = response["result"]["value"]["y"]
                    width = response["result"]["value"]["width"]
                    height = response["result"]["value"]["height"]
                    cur_node["union_bound"] = [x, y, width, height]

            dom_tree.append(cur_node)

        client.detach()
        # add parent children index to the node
        for parent_id, child_ids in graph.items():
            dom_tree[int(parent_id)]["childIds"] = child_ids

        # remove the nodes that are not in the current viewport
        if current_viewport_only:

            def remove_node_in_graph(node: DOMNode) -> None:
                # update the node information in the accessibility tree
                node_id = node["nodeId"]
                parent_id = node["parentId"]
                child_ids = node["childIds"]

                # update the children of the parent node
                assert dom_tree[int(parent_id)]["parentId"] != "[REMOVED]"
                # remove the nodeid from parent
                index = dom_tree[int(parent_id)]["childIds"].index(node_id)
                dom_tree[int(parent_id)]["childIds"].pop(index)

                # Insert children_nodeids in the same location
                for child_id in child_ids:
                    dom_tree[int(parent_id)]["childIds"].insert(index, child_id)
                    index += 1

                # update children node's parent
                for child_id in child_ids:
                    dom_tree[int(child_id)]["parentId"] = parent_id
                # mark as removed
                dom_tree[int(node_id)]["parentId"] = "[REMOVED]"

            config = info["config"]
            for cursor, node in enumerate(dom_tree):
                if not node["union_bound"]:
                    remove_node_in_graph(node)
                    continue

                [x, y, width, height] = node["union_bound"]

                # invisible node
                if width == 0.0 or height == 0.0:
                    remove_node_in_graph(node)
                    continue

                in_viewport_ratio = self.get_element_in_viewport_ratio(
                    elem_left_bound=float(x),
                    elem_top_bound=float(y),
                    width=float(width),
                    height=float(height),
                    config=config,
                )

                if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                    remove_node_in_graph(node)

            dom_tree = [
                node for node in dom_tree if node.get("parentId", "-1") != "[REMOVED]"
            ]

        return dom_tree

    @staticmethod
    def parse_html(dom_tree: DOMTree) -> tuple[str, dict[str, Any]]:
        """Parse the html tree into a string text"""

        obs_nodes_info = {}
        nodeid_to_cursor = {node["nodeId"]: idx for idx, node in enumerate(dom_tree)}

        def dfs(node_cursor: int, depth: int) -> str:
            tree_str = ""
            node = dom_tree[node_cursor]
            indent = "\t" * depth
            valid_node = True
            try:
                node_str = f"[{node_cursor}] <{node['nodeName']}"
                if node["attributes"]:
                    node_str += f" {node['attributes']}"
                node_str += f"> {node['nodeValue']}"
                valid_node = bool(node["attributes"] or node["nodeValue"])

                if valid_node:
                    obs_nodes_info[str(node_cursor)] = {
                        "backend_id": node["backendNodeId"],
                        "union_bound": node["union_bound"],
                        "text": node_str,
                    }
                    tree_str += f"{indent}{node_str}\n"

            except Exception as e:
                valid_node = False

            for child_ids in node["childIds"]:
                child_cursor = nodeid_to_cursor[child_ids]
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(child_cursor, child_depth)
                tree_str += child_str

            return tree_str

        html = dfs(0, 0)
        return html, obs_nodes_info

    def fetch_page_accessibility_tree(
        self,
        page: Page,
        info: BrowserInfo,
        current_viewport_only: bool,
    ) -> AccessibilityTree:
        client = page.context.new_cdp_session(page)
        accessibility_tree: AccessibilityTree = client.send(
            "Accessibility.getFullAXTree", {}
        )["nodes"]

        # a few nodes are repeated in the accessibility tree
        seen_ids = set()
        _accessibility_tree = []
        for node in accessibility_tree:
            if node["nodeId"] not in seen_ids:
                _accessibility_tree.append(node)
                seen_ids.add(node["nodeId"])
        accessibility_tree = _accessibility_tree

        nodeid_to_cursor = {}
        for cursor, node in enumerate(accessibility_tree):
            nodeid_to_cursor[node["nodeId"]] = cursor
            # usually because the node is not visible etc
            if "backendDOMNodeId" not in node:
                node["union_bound"] = None
                continue
            backend_node_id = str(node["backendDOMNodeId"])
            if node["role"]["value"] == "RootWebArea":
                # always inside the viewport
                node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
            else:
                response = self.get_bounding_client_rect(
                    client,
                    backend_node_id
                )
                if response.get("result", {}).get("subtype", "") == "error":
                    node["union_bound"] = None
                else:
                    x = response["result"]["value"]["x"]
                    y = response["result"]["value"]["y"]
                    width = response["result"]["value"]["width"]
                    height = response["result"]["value"]["height"]
                    node["union_bound"] = [x, y, width, height]

        client.detach()
        # filter nodes that are not in the current viewport
        if current_viewport_only:

            def remove_node_in_graph(node: AccessibilityTreeNode) -> None:
                # update the node information in the accessibility tree
                nodeid = node["nodeId"]
                node_cursor = nodeid_to_cursor[nodeid]
                parent_nodeid = node["parentId"]
                children_nodeids = node["childIds"]
                parent_cursor = nodeid_to_cursor[parent_nodeid]
                # update the children of the parent node
                assert (
                    accessibility_tree[parent_cursor].get("parentId", "Root")
                    is not None
                )
                # remove the nodeid from parent's childIds
                index = accessibility_tree[parent_cursor]["childIds"].index(nodeid)
                accessibility_tree[parent_cursor]["childIds"].pop(index)
                # Insert children_nodeids in the same location
                for child_nodeid in children_nodeids:
                    accessibility_tree[parent_cursor]["childIds"].insert(
                        index, child_nodeid
                    )
                    index += 1
                # update children node's parent
                for child_nodeid in children_nodeids:
                    child_cursor = nodeid_to_cursor[child_nodeid]
                    accessibility_tree[child_cursor]["parentId"] = parent_nodeid
                # mark as removed
                accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"

            config = info["config"]
            for node in accessibility_tree:
                if not node["union_bound"]:
                    remove_node_in_graph(node)
                    continue

                [x, y, width, height] = node["union_bound"]

                # invisible node
                if width == 0 or height == 0:
                    remove_node_in_graph(node)
                    continue

                in_viewport_ratio = self.get_element_in_viewport_ratio(
                    elem_left_bound=float(x),
                    elem_top_bound=float(y),
                    width=float(width),
                    height=float(height),
                    config=config,
                )

                if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                    remove_node_in_graph(node)

            accessibility_tree = [
                node
                for node in accessibility_tree
                if node.get("parentId", "Root") != "[REMOVED]"
            ]

        return accessibility_tree

    @staticmethod
    def parse_accessibility_tree(
        accessibility_tree: AccessibilityTree,
    ) -> tuple[str, dict[str, Any]]:
        """Parse the accessibility tree into a string text"""
        node_id_to_idx = {}
        for idx, node in enumerate(accessibility_tree):
            node_id_to_idx[node["nodeId"]] = idx

        obs_nodes_info = {}

        def dfs(idx: int, obs_node_id: str, depth: int) -> str:
            tree_str = ""
            node = accessibility_tree[idx]
            indent = "\t" * depth
            valid_node = True
            try:
                role = node["role"]["value"]
                name = node["name"]["value"]
                node_str = f"[{obs_node_id}] {role} {repr(name)}"
                properties = []
                for property in node.get("properties", []):
                    try:
                        if property["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        properties.append(
                            f'{property["name"]}: {property["value"]["value"]}'
                        )
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

                # check valid
                if not node_str.strip():
                    valid_node = False

                # empty generic node
                if not name.strip():
                    if not properties:
                        if role in [
                            "generic",
                            "img",
                            "list",
                            "strong",
                            "paragraph",
                            "banner",
                            "navigation",
                            "Section",
                            "LabelText",
                            "Legend",
                            "listitem",
                        ]:
                            valid_node = False
                    elif role in ["listitem"]:
                        valid_node = False

                if valid_node:
                    tree_str += f"{indent}{node_str}"
                    obs_nodes_info[obs_node_id] = {
                        "backend_id": node["backendDOMNodeId"],
                        "union_bound": node["union_bound"],
                        "text": node_str,
                    }

            except Exception as e:
                valid_node = False

            for _, child_node_id in enumerate(node["childIds"]):
                if child_node_id not in node_id_to_idx:
                    continue
                # mark this to save some tokens
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(
                    node_id_to_idx[child_node_id], child_node_id, child_depth
                )
                if child_str.strip():
                    if tree_str.strip():
                        tree_str += "\n"
                    tree_str += child_str

            return tree_str

        tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
        return tree_str, obs_nodes_info

    @staticmethod
    def clean_accesibility_tree(tree_str: str) -> str:
        """further clean accesibility tree"""
        clean_lines: list[str] = []
        for line in tree_str.split("\n"):
            # remove statictext if the content already appears in the previous line
            if "statictext" in line.lower():
                prev_lines = clean_lines[-3:]
                pattern = r"\[\d+\] StaticText (.+)"

                match = re.search(pattern, line, re.DOTALL)
                if match:
                    static_text = match.group(1)[1:-1]  # remove the quotes
                    if static_text and all(
                        static_text not in prev_line for prev_line in prev_lines
                    ):
                        clean_lines.append(line)
            else:
                clean_lines.append(line)

        return "\n".join(clean_lines)

    def fetch_image_related(self, page: Page, browser_info: BrowserInfo) -> str:
        # Check if the current page is an image url
        if page.url.endswith((".jpg", ".jpeg", ".png")):
            print("NOTE: We are on an image page!!!")
            # Load image from current url and run captioning on it.
            if page.url not in self.url2caption and self.captioning_fn is not None:
                try:
                    image = Image.open(requests.get(page.url, stream=True).raw)
                    caption = self.captioning_fn([image])[0].strip()
                    self.url2caption[page.url] = remove_unicode(caption)
                except Exception as e:
                    print("L579 WARNING: ", e)
            content = self.url2caption.get(page.url, "Image")

        else:
            if self.captioning_fn is not None:
                images = page.query_selector_all("img")
                image_urls = []
                for image in images:
                    try:
                        image_url = image.get_attribute("src")
                        if not image_url.startswith(("http://", "https://", "www.")):
                            image_url = urljoin(page.url, image_url)
                        if image_url not in self.url2caption:
                            image_urls.append(image_url)
                    except Exception as e:
                        print("L604 WARNING: ", e)

                # Run image captioning on image_url pixels. This is for models which use captioning as a baseline.
                if len(image_urls) > 0:
                    image_pixels = []
                    valid_urls = []
                    for url in image_urls:
                        if "data:image/svg" in url:
                            continue
                        else:
                            try:
                                image = Image.open(requests.get(url, stream=True).raw)
                                image_pixels.append(image)
                                valid_urls.append(url)
                            except Exception as e:
                                print("L616 WARNING: ", e)

                    # Caption images.
                    if image_pixels:
                        # Run in batches of 4.
                        bs = 4
                        captions = []
                        for i in range(0, len(image_pixels), bs):
                            try:
                                captions.extend(
                                    self.captioning_fn(image_pixels[i : i + bs])
                                )
                            except Exception as e:
                                print("L628 WARNING: ", e)
                                captions.extend([""] * len(image_pixels[i : i + bs]))
                        assert len(valid_urls) == len(
                            captions
                        ), f"len(images)={len(valid_urls)}, len(captions)={len(captions)}"
                        for image_url, caption in zip(valid_urls, captions):
                            self.url2caption[image_url] = remove_unicode(
                                caption.strip()
                            )

                image_idx = 0
                for image in images:
                    try:
                        original_alt = image.get_attribute("alt") or ""
                        image_url = image.get_attribute("src")
                        if not image_url.startswith(("http://", "https://", "www.")):
                            image_url = urljoin(page.url, image_url)

                        updated_alt = original_alt

                        if image_url in self.url2caption:
                            if self.url2caption[image_url] not in updated_alt:
                                updated_alt = f"{updated_alt}, description: {self.url2caption[image_url]}"
                        elif "data:image/svg" not in image_url:
                            print(f"WARNING: {image_url} not in self.url2caption")

                        if "url:" not in updated_alt:
                            updated_alt = f"{updated_alt}, url: {image_url}"

                        safe_updated_alt = json.dumps(updated_alt)
                        image.evaluate(f"node => node.alt = {safe_updated_alt}")
                    except Exception as e:
                        print("L653 WARNING:", e)

            if self.observation_type == "accessibility_tree_with_captioner":
                frame_ax_trees = self.fetch_page_accessibility_tree(
                    page,
                    browser_info,
                    current_viewport_only=self.current_viewport_only
                )
                content, obs_nodes_info = self.parse_accessibility_tree(frame_ax_trees)
                content = self.clean_accesibility_tree(content)
                self.obs_nodes_info = obs_nodes_info
                self.meta_data["obs_nodes_info"] = obs_nodes_info
            else:
                content = ""  # Not used for SoM

        return content

    def process(self, page: Page) -> str:
        # get the tab info
        open_tabs = page.context.pages
        try:
            tab_titles = [tab.title() for tab in open_tabs]
            current_tab_idx = open_tabs.index(page)
            for idx in range(len(open_tabs)):
                if idx == current_tab_idx:
                    tab_titles[idx] = f"Tab {idx} (current): {open_tabs[idx].title()}"
                else:
                    tab_titles[idx] = f"Tab {idx}: {open_tabs[idx].title()}"
            tab_title_str = " | ".join(tab_titles)
        except Exception:
            tab_title_str = " | ".join([f"Tab {idx}" for idx in range(len(open_tabs))])

        try:
            browser_info = self.fetch_browser_info(page)
        except Exception:
            page.wait_for_load_state("load", timeout=500)
            browser_info = self.fetch_browser_info(page)

        if self.observation_type == "html":
            dom_tree = self.fetch_page_html(
                browser_info,
                page,
                self.current_viewport_only,
            )
            content, obs_nodes_info = self.parse_html(dom_tree)
            self.obs_nodes_info = obs_nodes_info
            self.meta_data["obs_nodes_info"] = obs_nodes_info

        elif self.observation_type == "accessibility_tree":
            accessibility_tree = self.fetch_page_accessibility_tree(
                page,
                browser_info,
                self.current_viewport_only,
            )
            content, obs_nodes_info = self.parse_accessibility_tree(accessibility_tree)
            content = self.clean_accesibility_tree(content)
            self.obs_nodes_info = obs_nodes_info
            self.meta_data["obs_nodes_info"] = obs_nodes_info

        elif self.observation_type in [
            "accessibility_tree_with_captioner",
            "image_som",
        ]:
            content = self.fetch_image_related(
                page,
                browser_info,
            )

        elif self.observation_type == "":
            content = ""

        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        self.browser_config = browser_info["config"]
        content = f"{tab_title_str}\n\n{content}"

        return content

    def get_element_center(self, element_id: str) -> tuple[float, float]:
        node_info = self.obs_nodes_info[element_id]
        node_bound = node_info["union_bound"]
        x, y, width, height = node_bound
        center_x = x + width / 2
        center_y = y + height / 2
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )


class ImageObservationProcessor(ObservationProcessor):
    def __init__(
        self,
        observation_type: str,
        viewport_size: Optional[ViewportSize] = None,
    ):
        self.observation_type = observation_type
        self.observation_tag = "image"
        self.viewport_size = viewport_size
        self.meta_data = create_empty_metadata()

    def get_page_bboxes(self, page: Page) -> list[list[float]]:
        """JavaScript code to return bounding boxes and other metadata from HTML elements."""
        js_script = """
        (() => {
            const interactableSelectors = [
                'a[href]:not(:has(img))', 'a[href] img', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
                '[tabindex]:not([tabindex="-1"])', '[contenteditable="true"]', '[role="button"]', '[role="link"]',
                '[role="checkbox"]', '[role="menuitem"]', '[role="tab"]', '[draggable="true"]',
                '.btn', 'a[href="/notifications"]', 'a[href="/submit"]', '.fa.fa-star.is-rating-item', 'input[type="checkbox"]'

            ];

            const textSelectors = ['p', 'span', 'div:not(:has(*))', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article'];
            const modifiedTextSelectors = textSelectors.map(selector =>
                `:not(${interactableSelectors.join(', ')}):not(style) > ${selector}`
            );

            const combinedSelectors = [...interactableSelectors, ...modifiedTextSelectors];
            const elements = document.querySelectorAll(combinedSelectors.join(', '));

            const pixelRatio = window.devicePixelRatio;
            let csvContent = "ID,Element,Top,Right,Bottom,Left,Width,Height,Alt,Class,Id,TextContent,Interactable\\n";
            let counter = 1;

            elements.forEach(element => {
                const rect = element.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                let altText = element.getAttribute('alt') || '';
                altText = altText.replace(/"/g, ''); // Escape double quotes in alt text
                const classList = element.className || '';
                const id = element.id || '';
                let textContent = element.textContent || '';
                textContent = textContent.replace(/"/g, ''); // Escape double quotes in textContent

                // Determine if the element is interactable
                const isInteractable = interactableSelectors.some(selector => element.matches(selector));

                const dataString = [
                    counter, element.tagName, (rect.top + window.scrollY) * pixelRatio,
                    (rect.right + window.scrollX) * pixelRatio, (rect.bottom + window.scrollY) * pixelRatio,
                    (rect.left + window.scrollX) * pixelRatio, rect.width * pixelRatio, rect.height * pixelRatio,
                    altText, classList, id, textContent, isInteractable
                ].map(value => `"${value}"`).join(",");

                csvContent += dataString + "\\n";
                counter++;
            });

            return csvContent;
        })();
        """
        # Save the bbox as a CSV
        csv_content = page.evaluate(js_script)
        return csv_content

    def draw_bounding_boxes(
        self,
        data_string,
        screenshot_img,
        viewport_size=None,
        add_ids=True,
        bbox_color=None,
        min_width=8,
        min_height=8,
        bbox_padding=0,
        bbox_border=2,
        plot_ids=None,
    ):
        """
        min_width and min_height: Minimum dimensions of the bounding box to be plotted.
        """
        # Read CSV data
        df = pd.read_csv(StringIO(data_string), delimiter=",", quotechar='"')
        df["Area"] = df["Width"] * df["Height"]
        # Remove bounding boxes that are clipped.
        b_x, b_y = (
            self.browser_config["win_left_bound"],
            self.browser_config["win_upper_bound"],
        )
        if viewport_size is not None:
            df = df[
                (df["Bottom"] - b_y >= 0)
                & (df["Top"] - b_y <= viewport_size["height"])
                & (df["Right"] - b_x >= 0)
                & (df["Left"] - b_x <= viewport_size["width"])
            ]
            viewport_area = viewport_size["width"] * viewport_size["height"]
            # Filter out bounding boxes that too large (more than 80% of the viewport)
            df = df[df["Area"] <= 0.8 * viewport_area]

        # Open the screenshot image
        img = screenshot_img.copy()
        draw = ImageDraw.Draw(img)

        # Load a TTF font with a larger size
        font_path = "media/SourceCodePro-SemiBold.ttf"
        font_size, padding = 16, 2
        font = ImageFont.truetype(font_path, font_size)

        # Create a color cycle using one of the categorical color palettes in matplotlib
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        bbox_id2visid = {}
        bbox_id2desc = {}
        index = 0
        id2center = {}
        existing_text_rectangles = []
        text_to_draw = []
        # Provide [id] textContent inputs to the model as text.
        text_content_elements = []
        text_content_text = set()  # Store text of interactable elements

        # Iterate through each row in the CSV and draw bounding boxes
        for _, row in df.iterrows():
            if not row["Interactable"]:
                content = ""
                # Add image alt-text to the text representation.
                if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                    content += row["Alt"]
                # Add HTML textContent (if any) to the text representation.
                if pd.notna(row["TextContent"]):
                    content += (
                        row["TextContent"].strip().replace("\n", "").replace("\t", "")
                    )[
                        :200
                    ]  # Limit to 200 characters to avoid having too much text

                # Check if the text is a CSS selector
                if content and not (content.startswith(".") and "{" in content):
                    # Add elements which are not interactable as StaticText
                    if content not in text_content_text:
                        text_content_elements.append(f"[] [StaticText] [{content}]")
                        text_content_text.add(content)
                continue

            if (plot_ids is not None) and (row["ID"] not in plot_ids):
                continue

            unique_id = str(index + 1)
            bbox_id2visid[row["ID"]] = (
                unique_id  # map the bounding box ID to the unique character ID
            )
            top, right, bottom, left, width, height = (
                row["Top"],
                row["Right"],
                row["Bottom"],
                row["Left"],
                row["Width"],
                row["Height"],
            )
            left, right, top, bottom = left - b_x, right - b_x, top - b_y, bottom - b_y
            id2center[unique_id] = (
                (left + right) / 2,
                (bottom + top) / 2,
                width,
                height,
            )

            if width >= min_width and height >= min_height:
                # Get the next color in the cycle
                color = bbox_color or color_cycle[index % len(color_cycle)]
                draw.rectangle(
                    [
                        left - bbox_padding,
                        top - bbox_padding,
                        right + bbox_padding,
                        bottom + bbox_padding,
                    ],
                    outline=color,
                    width=bbox_border,
                )
                bbox_id2desc[row["ID"]] = color

                # Draw the text on top of the rectangle
                if add_ids:
                    # Calculate list of possible text positions
                    text_positions = [
                        (left - font_size, top - font_size),  # Top-left corner
                        (
                            left,
                            top - font_size,
                        ),  # A little to the right of the top-left corner
                        (right, top - font_size),  # Top-right corner
                        (
                            right - font_size - 2 * padding,
                            top - font_size,
                        ),  # A little to the left of the top-right corner
                        (left - font_size, bottom),  # Bottom-left corner
                        (
                            left,
                            bottom,
                        ),  # A little to the right of the bottom-left corner
                        (
                            right - font_size - 2 * padding,
                            bottom,
                        ),  # A little to the left of the bottom-right corner
                        (
                            left,
                            bottom,
                        ),  # A little to the right of the bottom-left corner
                        (
                            right - font_size - 2 * padding,
                            bottom,
                        ),  # A little to the left of the bottom-right corner
                    ]
                    text_width = draw.textlength(unique_id, font=font)
                    text_height = font_size  # Assume the text is one line

                    if viewport_size is not None:
                        for text_position in text_positions:
                            new_text_rectangle = [
                                text_position[0] - padding,
                                text_position[1] - padding,
                                text_position[0] + text_width + padding,
                                text_position[1] + text_height + padding,
                            ]

                            # Check if the new text rectangle is within the viewport
                            if (
                                new_text_rectangle[0] >= 0
                                and new_text_rectangle[1] >= 0
                                and new_text_rectangle[2] <= viewport_size["width"]
                                and new_text_rectangle[3] <= viewport_size["height"]
                            ):
                                # If the rectangle is within the viewport, check for overlaps
                                overlaps = False
                                for existing_rectangle in existing_text_rectangles:
                                    if self.rectangles_overlap(
                                        new_text_rectangle,
                                        existing_rectangle,
                                        padding * 2,
                                    ):
                                        overlaps = True
                                        break

                                if not overlaps:
                                    break
                            else:
                                # If the rectangle is outside the viewport, try the next position
                                continue
                    else:
                        # If none of the corners work, move the text rectangle by a fixed amount
                        text_position = (
                            text_positions[0][0] + padding,
                            text_positions[0][1],
                        )
                        new_text_rectangle = [
                            text_position[0] - padding,
                            text_position[1] - padding,
                            text_position[0] + text_width + padding,
                            text_position[1] + text_height + padding,
                        ]

                    existing_text_rectangles.append(new_text_rectangle)
                    text_to_draw.append(
                        (new_text_rectangle, text_position, unique_id, color)
                    )

                    content = ""
                    if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                        content += row["Alt"]
                    if pd.notna(row["TextContent"]):
                        content += (
                            row["TextContent"]
                            .strip()
                            .replace("\n", "")
                            .replace("\t", "")
                        )[
                            :200
                        ]  # Limit to 200 characters
                    text_content_elements.append(
                        f"[{unique_id}] [{row['Element']}] [{content}]"
                    )
                    if content in text_content_text:
                        # Remove text_content_elements with content
                        text_content_elements = [
                            element
                            for element in text_content_elements
                            if element.strip() != content
                        ]
                    text_content_text.add(content)

            index += 1

        for text_rectangle, text_position, unique_id, color in text_to_draw:
            # Draw a background rectangle for the text
            draw.rectangle(text_rectangle, fill=color)
            draw.text(text_position, unique_id, font=font, fill="white")

        content_str = "\n".join(text_content_elements)
        return img, id2center, content_str

    def rectangles_overlap(self, rect1, rect2, padding):
        """
        Check if two rectangles overlap.
        Each rectangle is represented as a list [x1, y1, x2, y2].
        """
        return not (
            rect1[2] < rect2[0] + padding
            or rect1[0] > rect2[2] - padding
            or rect1[1] > rect2[3] - padding
            or rect1[3] < rect2[1] + padding
        )

    def process(self, page: Page) -> npt.NDArray[np.uint8]:
        try:
            browser_info = self.fetch_browser_info(page)
        except Exception:
            page.wait_for_load_state("load", timeout=500)
            browser_info = self.fetch_browser_info(page)

        self.browser_config = browser_info["config"]

        if self.observation_type == "image_som":
            # Produce the SoM image, with bounding boxes
            try:
                screenshot_bytes = page.screenshot()
                som_bboxes = self.get_page_bboxes(page)
                screenshot_img = Image.open(BytesIO(screenshot_bytes))
                bbox_img, id2center, content_str = self.draw_bounding_boxes(
                    som_bboxes,
                    screenshot_img,
                    viewport_size=self.viewport_size,
                )
                self.som_id_info = id2center
                self.meta_data["obs_nodes_info"] = id2center
                screenshot_som = np.array(bbox_img)
                return screenshot_som, content_str
            except:
                page.wait_for_event("load")
                screenshot_bytes = page.screenshot()
                som_bboxes = self.get_page_bboxes(page)
                screenshot_img = Image.open(BytesIO(screenshot_bytes))
                bbox_img, id2center, content_str = self.draw_bounding_boxes(
                    som_bboxes,
                    screenshot_img,
                    viewport_size=self.viewport_size,
                )
                self.som_id_info = id2center
                self.meta_data["obs_nodes_info"] = id2center
                screenshot_som = np.array(bbox_img)
                return screenshot_som, content_str
        else:
            try:
                screenshot = png_bytes_to_numpy(page.screenshot())
            except:
                page.wait_for_event("load")
                screenshot = png_bytes_to_numpy(page.screenshot())
            return screenshot, ""

    def fetch_browser_info(self, page: Page) -> BrowserInfo:
        client = page.context.new_cdp_session(page)
        # extract domtree
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )
        client.detach()
        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds
        # add union bound placeholder
        tree["documents"][0]["layout"]["unionBounds"] = [None for _ in bounds]

        # extract browser info
        win_upper_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_upper_bound + win_height
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        config: BrowserConfig = {
            "win_upper_bound": win_upper_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        return info

    def get_element_center(self, element_id: str) -> tuple[float, float]:
        if not self.observation_type == "image_som":
            raise ValueError(
                "get_element_center() is only supported for 'image_som' observation type."
            )

        browser_config = self.browser_config
        center_x, center_y, width, height = self.som_id_info[element_id]
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )


class TextObervationProcessorWebRL(TextObervationProcessor):
    def __init__(
        self,
        observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        captioning_fn=None,
    ):
        super().__init__(
            observation_type,
            current_viewport_only,
            viewport_size,
            captioning_fn,
        )
        
    def process(self, page: Page) -> str:
        # get the tab info
        page_info = get_parsed_html(page)
        html = page_info["html"]
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        obs_nodes_info = {}
        for tag in soup.find_all(True):
            if tag.has_attr('id') and tag.has_attr('data-bbox'):
                backend_id = tag['id']
                union_bound = tag['data-bbox']
                union_bound = [float(num) for num in union_bound.split(',')]
                obs_nodes_info[str(backend_id)] = {
                    "backend_id": backend_id,
                    "union_bound": union_bound,
                    "text": str(tag)
                }
        self.obs_nodes_info = obs_nodes_info
        self.meta_data["obs_nodes_info"] = obs_nodes_info
        return html
    
    def get_element_center(self, element_id: str, page: Page=None) -> tuple[float, float]:
        
        if page is not None:
            element = page.query_selector(f"[data-label-id='{element_id}']")
            bbox = element.bounding_box()
            relative_bbox = (bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height'])
            center_x = (relative_bbox[0] + relative_bbox[2]) / 2
            center_y = (relative_bbox[1] + relative_bbox[3]) / 2
        else:
            node_info = self.obs_nodes_info[element_id]
            node_bound = node_info["union_bound"]
            x, y, width, height = node_bound
            center_x = x + width / 2
            center_y = y + height / 2
                
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )

class ObservationHandler:
    """Main entry point to access all observation processor"""

    def __init__(
        self,
        main_observation_type: str,
        text_observation_type: str,
        image_observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        captioning_fn=None,
    ) -> None:
        self.main_observation_type = main_observation_type
        if text_observation_type == "webrl":
            self.text_processor = TextObervationProcessorWebRL(
                text_observation_type,
                current_viewport_only,
                viewport_size,
                captioning_fn,
            )
        else:
            self.text_processor = TextObervationProcessor(
                text_observation_type,
                current_viewport_only,
                viewport_size,
                captioning_fn,
            )
        self.image_processor = ImageObservationProcessor(
            image_observation_type, viewport_size
        )
        self.viewport_size = viewport_size

    def get_observation_space(self) -> spaces.Dict:
        text_space = spaces.Text(
            min_length=0,
            max_length=UTTERANCE_MAX_LENGTH,
            charset=ASCII_CHARSET + FREQ_UNICODE_CHARSET,
        )

        image_space = spaces.Box(
            # Each position stores the RGB values. Note the swapped axes (height first).
            np.zeros(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            ),
            np.ones(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            )
            * 255.0,
            dtype=np.uint8,
        )

        return spaces.Dict({"text": text_space, "image": image_space})

    def get_observation(self, page: Page) -> dict[str, Observation]:
        text_obs = self.text_processor.process(page)
        image_obs, content_str = self.image_processor.process(page)
        if content_str != "":
            text_obs = content_str
        return {"text": text_obs, "image": image_obs}

    def get_observation_metadata(self) -> dict[str, ObservationMetadata]:
        return {
            "text": self.text_processor.meta_data,
            "image": self.image_processor.meta_data,
        }

    @property
    def action_processor(self) -> ObservationProcessor:
        """Return the main processor that is associated with the action space"""
        if self.main_observation_type == "text":
            return self.text_processor
        elif self.main_observation_type == "image":
            return self.image_processor
        else:
            raise ValueError("Invalid main observation type")
