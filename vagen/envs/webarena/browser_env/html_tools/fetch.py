import os
import json
import base64
from .html_parser import HtmlParser
from .configs import basic_attrs
from .scripts import *

def get_window(page):
    x = page.evaluate("window.scrollX")
    y = page.evaluate("window.scrollY")
    w = page.evaluate("window.innerWidth")
    h = page.evaluate("window.innerHeight")
    return (x, y, w, h)

def modify_page(page):
    page.wait_for_timeout(500)
    
    try:
        page.evaluate(remove_id_script)
    except:
        pass
    
    packet = {
        "raw_html": page.evaluate("document.documentElement.outerHTML"),
        "window": get_window(page)
    }
    
    page.evaluate(prepare_script)
    page.wait_for_timeout(100)
    
    img_bytes = page.screenshot(path="debug_info/screenshot_raw.png")
    raw_image = base64.b64encode(img_bytes).decode()
    
    page.evaluate(clickable_checker_script)
    page.wait_for_timeout(50)
    
    # get all clickable elements
    start_id = 0
    items, start_id = page.evaluate(label_script, {
        "selector": ".possible-clickable-element",
        "startIndex": start_id
    })
    page.wait_for_timeout(50)
    
    # mark our own labels and get the images
    items = page.evaluate(label_marker_script, items)
    page.wait_for_timeout(100)
    img_bytes = page.screenshot(path="debug_info/marked.png")
    marked_image = base64.b64encode(img_bytes).decode()
    
    # remove markers on the page
    page.evaluate(remove_label_mark_script)
    
    packet.update({
        "raw_image": raw_image,
        "marked_image": marked_image,
        "modified_html": page.evaluate("document.documentElement.outerHTML")
    })
    
    # element_info, include "all_elements" and "clickable_elements"
    element_info = page.evaluate(element_info_script)
    page.wait_for_timeout(100)
    packet.update(element_info)
    return packet

def save_debug_info(packet):
    with open("debug_info/raw.html", "w") as f:
        f.write(packet["modified_html"])
    with open("debug_info/parsed.html", "w") as f:
        f.write(packet["html"])
    with open("debug_info/all_element.json", "w") as f:
        f.write(json.dumps(packet["all_elements"]))
        
def get_parsed_html(page):
    if not os.path.exists("debug_info"):
        os.makedirs("debug_info")
        
    print("parsing html...")
    
    packet = modify_page(page)
    raw_html = packet["modified_html"]
    
    args = {
        "use_position": True,
        "rect_dict": {},
        "window_size": packet["window"],
        "id-attr": "data-backend-node-id",
        "label_attr": "data-label-id",
        "label_generator": "order",
        "regenerate_label": False,
        "attr_list": basic_attrs,
        "prompt": "xml",
        "dataset": "pipeline"
    }
    
    hp = HtmlParser(raw_html, args)
    res = hp.parse_tree()
    page_html = res.get("html", "")
    
    packet["html"] = page_html
    
    # for debug
    save_debug_info(packet)
    
    print("parsing finished.")
    
    return packet

