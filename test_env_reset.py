#!/usr/bin/env python3
"""Test WebArena environment reset for all train and test tasks."""
import json
import os
import sys
import time
from playwright.sync_api import sync_playwright

# Ensure env vars are set
required_vars = ["SHOPPING", "SHOPPING_ADMIN", "GITLAB", "REDDIT", "WIKIPEDIA", "MAP", "HOMEPAGE"]
for var in required_vars:
    val = os.environ.get(var, "")
    print(f"  {var}={val}")
    if not val:
        print(f"ERROR: {var} not set!")
        sys.exit(1)

def test_config_file(config_path, label):
    """Load a config JSON and test connectivity to all start_urls."""
    with open(config_path) as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        print(f"  {label}: not a JSON array, skipping")
        return

    # Collect unique start_urls
    urls = set()
    for task in tasks:
        url = task.get("start_url", "")
        if url:
            # Extract base url (scheme + host + port)
            from urllib.parse import urlparse
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            urls.add(base)

    print(f"\n{'='*60}")
    print(f"{label}: {len(tasks)} tasks, {len(urls)} unique base URLs")
    print(f"{'='*60}")

    # Test each unique URL with playwright
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)

    results = {"ok": 0, "fail": 0}
    for url in sorted(urls):
        page = browser.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=10000)
            status = "OK"
            results["ok"] += 1
        except Exception as e:
            status = f"FAIL: {e}"
            results["fail"] += 1
        finally:
            page.close()
        print(f"  [{status[:4]}] {url}")

    browser.close()
    pw.stop()

    print(f"\nResult: {results['ok']} OK, {results['fail']} FAIL out of {len(urls)} URLs")
    return results["fail"]

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base, "vagen/envs/webarena/config_files/wa")

    total_fails = 0

    train_path = os.path.join(config_dir, "train_webarena_lite.json")
    if os.path.exists(train_path):
        total_fails += test_config_file(train_path, "TRAIN (train_webarena_lite)")
    else:
        print(f"Train config not found: {train_path}")

    test_path = os.path.join(config_dir, "test_webarena_lite.json")
    if os.path.exists(test_path):
        total_fails += test_config_file(test_path, "TEST (test_webarena_lite)")
    else:
        print(f"Test config not found: {test_path}")

    print(f"\n{'='*60}")
    if total_fails == 0:
        print("ALL URLs OK!")
    else:
        print(f"TOTAL FAILURES: {total_fails}")
    sys.exit(total_fails)
