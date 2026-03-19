#!/usr/bin/env python3
"""
Test equivalence between original per-node bounding rect approach
and the DOMSnapshot-based batch approach for accessibility tree observation.

Runs both methods on the same pages and compares:
  1. Output equivalence (text similarity)
  2. Performance (timing)

Usage:
    python test_obs_equivalence.py
    python test_obs_equivalence.py --seeds 0,17,22,35
    python test_obs_equivalence.py --rounds 3
"""

import argparse
import difflib
import json
import os
import time
from typing import Any

from playwright.sync_api import sync_playwright, CDPSession, Page

# Reuse existing processors
from vagen.envs.webarena.browser_env.processors import (
    ObservationHandler,
    TextObervationProcessor,
    IN_VIEWPORT_RATIO_THRESHOLD,
)
from vagen.envs.webarena.browser_env.utils import (
    AccessibilityTree,
    AccessibilityTreeNode,
    BrowserConfig,
    BrowserInfo,
)
from vagen.envs.webarena.browser_env.constants import IGNORED_ACTREE_PROPERTIES


# ─── New: DOMSnapshot-based bounding rect ────────────────────────

def fetch_accessibility_tree_fast(
    processor: TextObervationProcessor,
    info: BrowserInfo,
    client: CDPSession,
    current_viewport_only: bool,
) -> AccessibilityTree:
    """Same as original fetch_page_accessibility_tree but uses DOMSnapshot
    for bounding rects instead of per-node CDP calls.

    DOMSnapshot.captureSnapshot is already called in fetch_browser_info(),
    so we reuse info["DOMTree"] which contains layout bounds for all nodes.
    This replaces N×2 CDP calls with 0 additional calls.
    """
    accessibility_tree: AccessibilityTree = client.send(
        "Accessibility.getFullAXTree", {}
    )["nodes"]

    # De-duplicate
    seen_ids = set()
    _accessibility_tree = []
    for node in accessibility_tree:
        if node["nodeId"] not in seen_ids:
            _accessibility_tree.append(node)
            seen_ids.add(node["nodeId"])
    accessibility_tree = _accessibility_tree

    # Build backendNodeId -> bounds map from DOMSnapshot
    # The DOMSnapshot is already in info["DOMTree"] from fetch_browser_info()
    tree = info["DOMTree"]
    document = tree["documents"][0]
    nodes = document["nodes"]
    layout = document["layout"]

    # DOMSnapshot bounds are in PAGE coordinates (relative to document origin).
    # getBoundingClientRect() returns VIEWPORT coordinates (relative to viewport).
    # Difference = scroll offset. Subtract scroll offset to match original behavior.
    scroll_x = info["config"]["win_left_bound"]
    scroll_y = info["config"]["win_top_bound"]

    # layout["nodeIndex"] tells which DOM node each layout entry belongs to
    # layout["bounds"] has the bounding rect [x, y, width, height] in page coords
    # nodes["backendNodeId"] maps DOM node index to backendNodeId
    backend_to_bounds: dict[int, list[float]] = {}
    layout_node_indices = layout["nodeIndex"]
    layout_bounds = layout["bounds"]

    for i, dom_node_idx in enumerate(layout_node_indices):
        if dom_node_idx < len(nodes["backendNodeId"]):
            backend_id = nodes["backendNodeId"][dom_node_idx]
            b = layout_bounds[i]
            # Convert page coords to viewport coords by subtracting scroll offset
            backend_to_bounds[backend_id] = [
                b[0] - scroll_x,  # x
                b[1] - scroll_y,  # y
                b[2],             # width (unchanged)
                b[3],             # height (unchanged)
            ]

    # Assign bounds to accessibility tree nodes using the map
    nodeid_to_cursor = {}
    for cursor, node in enumerate(accessibility_tree):
        nodeid_to_cursor[node["nodeId"]] = cursor
        if "backendDOMNodeId" not in node:
            node["union_bound"] = None
            continue
        backend_node_id = int(node["backendDOMNodeId"])
        if node["role"]["value"] == "RootWebArea":
            node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
        else:
            bound = backend_to_bounds.get(backend_node_id)
            if bound is not None:
                node["union_bound"] = list(bound)
            else:
                node["union_bound"] = None

    # Viewport filtering — identical to original
    if current_viewport_only:

        def remove_node_in_graph(node: AccessibilityTreeNode) -> None:
            nodeid = node["nodeId"]
            node_cursor = nodeid_to_cursor[nodeid]
            parent_nodeid = node["parentId"]
            children_nodeids = node["childIds"]
            parent_cursor = nodeid_to_cursor[parent_nodeid]
            assert (
                accessibility_tree[parent_cursor].get("parentId", "Root")
                is not None
            )
            index = accessibility_tree[parent_cursor]["childIds"].index(nodeid)
            accessibility_tree[parent_cursor]["childIds"].pop(index)
            for child_nodeid in children_nodeids:
                accessibility_tree[parent_cursor]["childIds"].insert(
                    index, child_nodeid
                )
                index += 1
            for child_nodeid in children_nodeids:
                child_cursor = nodeid_to_cursor[child_nodeid]
                accessibility_tree[child_cursor]["parentId"] = parent_nodeid
            accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"

        config = info["config"]
        for node in accessibility_tree:
            if not node["union_bound"]:
                remove_node_in_graph(node)
                continue

            [x, y, width, height] = node["union_bound"]

            if width == 0 or height == 0:
                remove_node_in_graph(node)
                continue

            in_viewport_ratio = processor.get_element_in_viewport_ratio(
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


def get_obs_fast(
    processor: TextObervationProcessor,
    page: Page,
    client: CDPSession,
) -> str:
    """Get observation using DOMSnapshot-based fast method."""
    browser_info = processor.fetch_browser_info(page, client)

    accessibility_tree = fetch_accessibility_tree_fast(
        processor,
        browser_info,
        client,
        current_viewport_only=processor.current_viewport_only,
    )
    content, obs_nodes_info = processor.parse_accessibility_tree(
        accessibility_tree
    )
    content = processor.clean_accesibility_tree(content)

    # Tab info (same as original process())
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
        tab_title_str = " | ".join(
            ["Tab {idx}" for idx in range(len(open_tabs))]
        )

    content = f"{tab_title_str}\n\n{content}"
    return content


# ─── Comparison logic ────────────────────────────────────────────

def compare_texts(original: str, fast: str) -> dict:
    """Compare two observation texts and return similarity metrics."""
    orig_lines = original.strip().splitlines()
    fast_lines = fast.strip().splitlines()

    # Sequence matcher for overall similarity
    sm = difflib.SequenceMatcher(None, orig_lines, fast_lines)
    ratio = sm.ratio()

    # Count matching/different lines
    diff = list(difflib.unified_diff(orig_lines, fast_lines, n=0))
    added = sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))
    removed = sum(1 for l in diff if l.startswith('-') and not l.startswith('---'))

    return {
        "similarity": ratio,
        "orig_lines": len(orig_lines),
        "fast_lines": len(fast_lines),
        "orig_chars": len(original),
        "fast_chars": len(fast),
        "lines_added": added,
        "lines_removed": removed,
        "diff_preview": "\n".join(diff[:30]) if diff else "(identical)",
    }


# ─── Services ───────────────────────────────────────────────────

SERVICES = {
    "SHOPPING": os.environ.get("SHOPPING", "http://localhost:7770"),
    "SHOPPING_ADMIN": os.environ.get("SHOPPING_ADMIN", "http://localhost:7780/admin"),
    "GITLAB": os.environ.get("GITLAB", "http://localhost:8023"),
    "REDDIT": os.environ.get("REDDIT", "http://localhost:9999"),
}


def fmt_time(t: float) -> str:
    if t < 0.001:
        return f"{t*1e6:.0f}us"
    if t < 1.0:
        return f"{t*1e3:.1f}ms"
    return f"{t:.2f}s"


def run_comparison(rounds: int = 2, do_scroll: bool = True):
    """Run side-by-side comparison on each service."""

    print("=" * 70)
    print("  Accessibility Tree: Original vs DOMSnapshot-based")
    print(f"  Rounds: {rounds}, Scroll test: {do_scroll}")
    print("=" * 70)

    viewport = {"width": 1280, "height": 720}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--blink-settings=imagesEnabled=false"],
        )

        for name, url in SERVICES.items():
            print(f"\n--- {name}: {url} ---")

            context = browser.new_context(
                viewport=viewport,
                device_scale_factor=1,
            )
            page = context.new_page()
            client = page.context.new_cdp_session(page)
            client.send("Accessibility.enable")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                # Wait for page to fully settle so both methods see the same DOM
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception as e:
                print(f"  SKIP: goto failed: {e}")
                context.close()
                continue

            processor = TextObervationProcessor(
                "accessibility_tree",
                current_viewport_only=True,
                viewport_size=viewport,
            )

            scenarios = [("after_goto", False)]
            if do_scroll:
                scenarios.append(("after_scroll", True))

            for scenario_name, should_scroll in scenarios:
                if should_scroll:
                    page.evaluate("window.scrollBy(0, 500)")
                    page.wait_for_timeout(200)

                print(f"\n  [{scenario_name}]")

                orig_times = []
                fast_times = []

                for rnd in range(rounds):
                    # Original method
                    t0 = time.perf_counter()
                    orig_obs = processor.process(page, client)
                    dt_orig = time.perf_counter() - t0
                    orig_times.append(dt_orig)

                    # Fast method
                    t0 = time.perf_counter()
                    fast_obs = get_obs_fast(processor, page, client)
                    dt_fast = time.perf_counter() - t0
                    fast_times.append(dt_fast)

                    if rnd == 0:
                        comp = compare_texts(orig_obs, fast_obs)

                # Timing
                avg_orig = sum(orig_times) / len(orig_times)
                avg_fast = sum(fast_times) / len(fast_times)
                speedup = avg_orig / avg_fast if avg_fast > 0 else 0

                print(f"    Original:  avg={fmt_time(avg_orig)}  "
                      f"(times: {', '.join(fmt_time(t) for t in orig_times)})")
                print(f"    Fast:      avg={fmt_time(avg_fast)}  "
                      f"(times: {', '.join(fmt_time(t) for t in fast_times)})")
                print(f"    Speedup:   {speedup:.1f}x")

                # Equivalence
                print(f"    Similarity: {comp['similarity']:.4f} "
                      f"({comp['similarity']*100:.1f}%)")
                print(f"    Original:   {comp['orig_lines']} lines, "
                      f"{comp['orig_chars']} chars")
                print(f"    Fast:       {comp['fast_lines']} lines, "
                      f"{comp['fast_chars']} chars")
                print(f"    Diff:       +{comp['lines_added']} "
                      f"-{comp['lines_removed']} lines")

                if comp['similarity'] < 1.0:
                    print(f"    Diff preview:")
                    for line in comp['diff_preview'].split('\n')[:15]:
                        print(f"      {line}")

            context.close()

        browser.close()

    print(f"\n{'=' * 70}")
    print("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--no-scroll", action="store_true")
    args = parser.parse_args()

    run_comparison(rounds=args.rounds, do_scroll=not args.no_scroll)
