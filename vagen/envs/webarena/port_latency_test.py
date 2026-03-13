"""
Test response latency of each WebArena service (different ports).

Measures:
  1. HTTP response time (simple GET request)
  2. Browser page load time (Playwright)

Usage:
    python -m vagen.envs.webarena.port_latency_test
    python -m vagen.envs.webarena.port_latency_test --rounds 5 --skip_browser
"""

import os
import time
import statistics
from typing import Dict, List

import fire
import requests


# Service URLs from environment variables
SERVICES: Dict[str, str] = {
    "REDDIT": os.environ.get("REDDIT", "http://localhost:9999"),
    "SHOPPING": os.environ.get("SHOPPING", "http://localhost:7770"),
    "SHOPPING_ADMIN": os.environ.get("SHOPPING_ADMIN", "http://localhost:7780/admin"),
    "GITLAB": os.environ.get("GITLAB", "http://localhost:8023"),
    "WIKIPEDIA": os.environ.get("WIKIPEDIA", "http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"),
    "MAP": os.environ.get("MAP", "http://localhost:3000"),
    "HOMEPAGE": os.environ.get("HOMEPAGE", "http://localhost:4399"),
}


def test_http_latency(url: str, timeout: float = 30.0) -> Dict:
    """Measure HTTP GET latency."""
    try:
        t0 = time.perf_counter()
        resp = requests.get(url, timeout=timeout, allow_redirects=True)
        dt = time.perf_counter() - t0
        return {
            "time": dt,
            "status": resp.status_code,
            "size_kb": len(resp.content) / 1024,
            "error": None,
        }
    except Exception as e:
        dt = time.perf_counter() - t0
        return {"time": dt, "status": None, "size_kb": 0, "error": str(e)}


def test_browser_latency(url: str, headless: bool = True, timeout_ms: int = 60000) -> Dict:
    """Measure Playwright page load latency."""
    from playwright.sync_api import sync_playwright

    try:
        t_total = time.perf_counter()
        with sync_playwright() as p:
            t0 = time.perf_counter()
            browser = p.chromium.launch(headless=headless)
            dt_launch = time.perf_counter() - t0

            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            page.set_default_navigation_timeout(timeout_ms)

            t0 = time.perf_counter()
            page.goto(url, wait_until="domcontentloaded")
            dt_goto = time.perf_counter() - t0

            t0 = time.perf_counter()
            page.wait_for_load_state("networkidle")
            dt_idle = time.perf_counter() - t0

            browser.close()

        return {
            "launch": dt_launch,
            "goto": dt_goto,
            "network_idle": dt_idle,
            "total": time.perf_counter() - t_total,
            "error": None,
        }
    except Exception as e:
        return {
            "launch": 0, "goto": 0, "network_idle": 0,
            "total": time.perf_counter() - t_total,
            "error": str(e),
        }


def benchmark(
    rounds: int = 3,
    skip_browser: bool = False,
    headless: bool = True,
    http_timeout: float = 30.0,
    browser_timeout: int = 60000,
):
    """
    Benchmark response latency of all WebArena services.

    Args:
        rounds: Number of HTTP requests per service.
        skip_browser: Skip Playwright browser tests (faster).
        headless: Run browser in headless mode.
        http_timeout: HTTP request timeout in seconds.
        browser_timeout: Playwright timeout in ms.
    """
    print("=" * 70)
    print("WebArena Port Latency Benchmark")
    print("=" * 70)

    # --- HTTP latency ---
    print(f"\n--- HTTP GET Latency ({rounds} rounds each) ---\n")
    print(f"{'Service':<18} {'URL':<60} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10} {'Status':<8} {'Size(KB)':<10}")
    print("-" * 126)

    http_results: Dict[str, List[Dict]] = {}
    for name, url in SERVICES.items():
        results = []
        for r in range(rounds):
            res = test_http_latency(url, timeout=http_timeout)
            results.append(res)
            status_str = str(res["status"]) if res["status"] else "ERR"
            print(f"  [{name}] round {r+1}: {res['time']:.3f}s (status={status_str})", flush=True)
        http_results[name] = results

    print(f"\n{'Service':<18} {'URL':<60} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10} {'Status':<8} {'Size(KB)':<10}")
    print("-" * 126)
    for name, url in SERVICES.items():
        results = http_results[name]
        times = [r["time"] for r in results]
        statuses = [r["status"] for r in results]
        sizes = [r["size_kb"] for r in results]
        errors = [r for r in results if r["error"]]

        avg_t = statistics.mean(times)
        min_t = min(times)
        max_t = max(times)
        status_str = str(statuses[0]) if statuses[0] else "ERR"
        size_str = f"{sizes[0]:.1f}" if sizes[0] else "N/A"

        marker = " !!!" if errors else ""
        print(f"{name:<18} {url:<60} {avg_t:<10.3f} {min_t:<10.3f} {max_t:<10.3f} {status_str:<8} {size_str:<10}{marker}")

    # Rank by average latency
    print("\n--- Ranking (fastest to slowest, by avg HTTP latency) ---\n")
    ranked = sorted(
        http_results.items(),
        key=lambda kv: statistics.mean(r["time"] for r in kv[1]),
    )
    for i, (name, results) in enumerate(ranked, 1):
        avg_t = statistics.mean(r["time"] for r in results)
        err_count = sum(1 for r in results if r["error"])
        err_str = f" ({err_count} errors)" if err_count else ""
        print(f"  {i}. {name:<18} avg={avg_t:.3f}s{err_str}")

    # --- Browser latency ---
    if not skip_browser:
        print(f"\n--- Browser Page Load Latency (Playwright, 1 round each) ---\n")
        print(f"{'Service':<18} {'goto(s)':<12} {'idle(s)':<12} {'total(s)':<12} {'Error'}")
        print("-" * 80)

        for name, url in SERVICES.items():
            res = test_browser_latency(url, headless=headless, timeout_ms=browser_timeout)
            err_str = res["error"][:50] if res["error"] else ""
            print(
                f"{name:<18} {res['goto']:<12.3f} {res['network_idle']:<12.3f} "
                f"{res['total']:<12.3f} {err_str}"
            )

    print("\nDone.")


if __name__ == "__main__":
    fire.Fire(benchmark)
