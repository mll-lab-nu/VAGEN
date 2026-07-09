import os
from pathlib import Path
rootdir = Path(__file__).parent
    
with open(os.path.join(rootdir,'prepare.js'), 'r') as f:
    prepare_script = f.read()
    
with open(os.path.join(rootdir, 'clickable_checker.js'), 'r') as f:
    clickable_checker_script = f.read()

with open(os.path.join(rootdir, 'label.js'), 'r') as f:
    label_script = f.read()

with open(os.path.join(rootdir, 'element_info.js'), 'r') as f:
    element_info_script = f.read()
    
# draw label on page
with open(os.path.join(rootdir, 'label_marker.js'), 'r') as f:
    label_marker_script = f.read()

# remove label draw on page
remove_label_mark_script = """
    () => {
        document.querySelectorAll(".our-dom-marker").forEach(item => {
            document.body.removeChild(item);
        });
    }
"""

remove_id_script = """
    () => {
        Array.from(document.getElementsByClassName('possible-clickable-element')).forEach((element) => {
            element.classList.remove('possible-clickable-element');
            element.removeAttribute('data-value');
            element.removeAttribute('data-text');
            element.removeAttribute('data-label');
            element.removeAttribute('data-bbox');
            element.removeAttribute('data-status');
            element.removeAttribute('data-backend-node-id');
            element.removeAttribute('data-label-id');
        });
    }
"""
