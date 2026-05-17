() => {
    function getElementInfo(element) {
        return {
            "bid": element.getAttribute("data-backend-node-id") || "",
            "label": element.getAttribute("data-label-id") || "",
            "tag": element.tagName.toLowerCase?.() || "",
            "area": JSON.parse("[" + (element.getAttribute("data-bbox") || "") + "]"),
            "text": element.innerText?.trim().replace(/\s{2,}/g, " ") || "",
            "id": element.getAttribute("id") || "",
            "role": element.getAttribute("role") || "",
            "aria-label": element.getAttribute("aria-label") || "",
            "href": element.getAttribute("href") || "",
        };
    }
    
    var all_items = Array.prototype.slice.call(
        document.querySelectorAll("*")
    ).map((element) => {
        return getElementInfo(element);
    });
    
    var clickable_items = Array.prototype.slice.call(
        document.querySelectorAll("*")
    ).filter(
        element => element.getAttribute("data-label-id")
    ).map((element) => {
        return getElementInfo(element);
    });
    
    return {
        all_elements: all_items, 
        clickable_elements: clickable_items
    };
}