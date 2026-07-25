() => {
    // mark backend node id
    var vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
    var vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);

    var backendId = 0;
    Array.prototype.slice.call(
        document.querySelectorAll("*")
    ).forEach((element) => {
        element.setAttribute("data-backend-node-id", backendId);
        backendId++;
        
        var tag = element.tagName.toLowerCase?.() || "";
        var bb = element.getClientRects();
        var rect = {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
            width: 0,
            height: 0
        };

        if (bb.length > 0) {
            bb = bb[0];
            // rect = {
            //     left: Math.round(Math.max(0, bb.left) * 100) / 100,
            //     top: Math.round(Math.max(0, bb.top) * 100) / 100,
            //     right: Math.round(Math.min(vw, bb.right) * 100) / 100,
            //     bottom: Math.round(Math.min(vh, bb.bottom) * 100) / 100
            // };
            rect = {
                left: (Math.round(bb.left) * 100) / 100,
                top: (Math.round(bb.top) * 100) / 100,
                right: (Math.round(bb.right) * 100) / 100,
                bottom: (Math.round(bb.bottom) * 100) / 100
            };
            rect = {
                ...rect,
                width: Math.round((rect.right - rect.left) * 100) / 100,
                height: Math.round((rect.bottom - rect.top) * 100) / 100
            };
            
            element.setAttribute("data-bbox", `${rect.left},${rect.top},${rect.width},${rect.height}`);
        }

        if (element.hasChildNodes()) {
            let children = Array.prototype.slice.call(element.childNodes);
            var texts = children.filter(
                (node) => node.nodeType == Node.TEXT_NODE
            ).map(
                (node) => node.textContent.trim().replace(/\s{2,}/g, " ") || ""
            ).filter(
                (text) => text.length > 0
            )
            element.setAttribute("data-text", texts.join(","));
        } 

        // fix select issue
        if (tag == "select") {
            var value = element.value;
            var text = element.options[element.selectedIndex]?.text || "";
            element.setAttribute("data-value", value);
            element.setAttribute("data-text", text);
            element.options[element.selectedIndex]?.setAttribute("data-status", "selected");
        }

        if (tag == "input") {
            var input_type = element.getAttribute("type") || "";
            if (input_type == "checkbox") {
                var status = element.checked? "checked" : "not-checked";
                element.setAttribute("data-status", status);
            }
        }
    });

    // fix input and textarea issue
    Array.prototype.slice.call(
        document.querySelectorAll("input, textarea")
    ).forEach(element => {
        element.setAttribute("data-value", element.value);
    });
}