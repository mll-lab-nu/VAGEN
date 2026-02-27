"""Build observations from the current browser page state."""

import io
from typing import Any, Dict, List

from PIL import Image
from playwright.async_api import Page


# ---------------------------------------------------------------------------
# Accessibility tree extraction
# ---------------------------------------------------------------------------

# JavaScript to inject into the page that:
# 1. Assigns sequential data-webarena-id attributes to interactive elements
# 2. Returns a text-based accessibility tree
_AX_TREE_JS = """
() => {
    const INTERACTIVE_TAGS = new Set([
        'A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA',
        'DETAILS', 'SUMMARY', 'LABEL',
    ]);
    const INTERACTIVE_ROLES = new Set([
        'button', 'link', 'textbox', 'combobox', 'listbox',
        'menuitem', 'tab', 'checkbox', 'radio', 'switch',
        'searchbox', 'option', 'slider',
    ]);

    let nextId = 0;
    const lines = [];

    function isVisible(el) {
        if (!el.offsetParent && el.tagName !== 'BODY' && el.tagName !== 'HTML') return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        if (parseFloat(style.opacity) === 0) return false;
        return true;
    }

    function isInteractive(el) {
        if (INTERACTIVE_TAGS.has(el.tagName)) return true;
        const role = el.getAttribute('role');
        if (role && INTERACTIVE_ROLES.has(role.toLowerCase())) return true;
        if (el.hasAttribute('onclick') || el.hasAttribute('tabindex')) return true;
        return false;
    }

    function getLabel(el) {
        // Try aria-label, then title, then alt, then innerText (truncated)
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel) return ariaLabel.trim();
        const title = el.getAttribute('title');
        if (title) return title.trim();
        if (el.tagName === 'IMG') {
            const alt = el.getAttribute('alt');
            if (alt) return alt.trim();
            return '[image]';
        }
        if (el.tagName === 'INPUT') {
            const placeholder = el.getAttribute('placeholder');
            const value = el.value;
            if (value) return 'value=' + value.substring(0, 50);
            if (placeholder) return 'placeholder=' + placeholder;
            return el.type || 'input';
        }
        if (el.tagName === 'SELECT') {
            const selected = el.options[el.selectedIndex];
            return selected ? selected.text.substring(0, 50) : 'select';
        }
        const text = el.innerText || el.textContent || '';
        return text.substring(0, 80).replace(/\\s+/g, ' ').trim();
    }

    function walk(el, depth) {
        if (!el || el.nodeType !== 1) return;
        if (!isVisible(el)) return;

        const tag = el.tagName.toLowerCase();
        const role = el.getAttribute('role') || tag;

        if (isInteractive(el)) {
            const eid = nextId++;
            el.setAttribute('data-webarena-id', String(eid));
            const label = getLabel(el);
            const indent = '  '.repeat(depth);
            lines.push(indent + '[' + eid + '] ' + role + ' ' + JSON.stringify(label));
        }

        for (const child of el.children) {
            walk(child, depth + (isInteractive(el) ? 1 : 0));
        }
    }

    // Clean old IDs
    document.querySelectorAll('[data-webarena-id]').forEach(el => {
        el.removeAttribute('data-webarena-id');
    });

    walk(document.body, 0);
    return lines.join('\\n');
}
"""


async def get_accessibility_tree(page: Page) -> str:
    """Inject JS and return a text-based accessibility tree with element IDs."""
    try:
        tree = await page.evaluate(_AX_TREE_JS)
        return tree if tree else "(empty page)"
    except Exception as e:
        return f"(error extracting accessibility tree: {e})"


async def get_screenshot(page: Page) -> Image.Image:
    """Capture a screenshot and return as a PIL Image."""
    screenshot_bytes = await page.screenshot(full_page=False)
    return Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")


# ---------------------------------------------------------------------------
# Observation builders
# ---------------------------------------------------------------------------

async def build_observation_vision(
    page: Page,
    task_intent: str,
    remaining_steps: int,
    image_placeholder: str = "<image>",
) -> Dict[str, Any]:
    """Build a vision-mode observation with screenshot."""
    screenshot = await get_screenshot(page)
    url = page.url

    obs_str = (
        f"Task: {task_intent}\n"
        f"Current URL: {url}\n"
        f"{image_placeholder}\n"
        f"Budget: {remaining_steps} steps remaining."
    )
    return {
        "obs_str": obs_str,
        "multi_modal_input": {
            image_placeholder: [screenshot],
        },
    }


async def build_observation_text(
    page: Page,
    task_intent: str,
    remaining_steps: int,
) -> Dict[str, Any]:
    """Build a text-mode observation with accessibility tree."""
    ax_tree = await get_accessibility_tree(page)
    url = page.url

    obs_str = (
        f"Task: {task_intent}\n"
        f"Current URL: {url}\n\n"
        f"Accessibility Tree:\n{ax_tree}\n\n"
        f"Budget: {remaining_steps} steps remaining."
    )
    return {"obs_str": obs_str}


async def build_observation(
    page: Page,
    task_intent: str,
    remaining_steps: int,
    render_mode: str = "vision",
    image_placeholder: str = "<image>",
) -> Dict[str, Any]:
    """Build observation based on render_mode."""
    if render_mode == "vision":
        return await build_observation_vision(
            page, task_intent, remaining_steps, image_placeholder
        )
    elif render_mode == "text":
        return await build_observation_text(page, task_intent, remaining_steps)
    else:
        raise ValueError(f"Unknown render_mode: {render_mode}")
