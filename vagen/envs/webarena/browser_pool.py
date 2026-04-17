"""Chromium browser pool backing the WebArena handler.

Model: K long-lived Chromium browsers; each owns a dedicated Python OS thread
(Playwright sync API's greenlets are bound to the thread that called
`sync_playwright()`, so every operation on a browser / context / page derived
from it MUST be dispatched back onto that same thread).

A session (one VAGEN env instance) borrows a `BrowserSlot` and creates its
own Playwright context on it. The slot's `run(fn)` dispatches to the
dedicated executor so the caller's asyncio coroutine can `await` it.

Capacity is bounded by a semaphore over total contexts = K * M. Sessions wait
if all slots are full.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Callable, List, Optional

LOGGER = logging.getLogger(__name__)


class BrowserSlot:
    """One Chromium browser + its dedicated single-thread executor.

    All Playwright calls touching this browser's objects must be issued
    through `run()`, which submits to the slot's executor. Never call the
    Playwright objects directly from another thread.
    """

    def __init__(self, browser_id: int, launch_args: Optional[List[str]] = None):
        self.browser_id = browser_id
        self._launch_args = launch_args or [
            "--blink-settings=imagesEnabled=false",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-extensions",
        ]
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"pw-browser-{browser_id}"
        )
        self._pw_cm = None  # sync_playwright() context manager
        self._pw = None     # Playwright instance
        self._browser = None  # Browser instance
        self.n_contexts: int = 0
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        launch_args = self._launch_args

        def _launch():
            from playwright.sync_api import sync_playwright
            cm = sync_playwright()
            pw = cm.__enter__()
            browser = pw.chromium.launch(headless=True, args=launch_args)
            return cm, pw, browser

        loop = asyncio.get_running_loop()
        future = self._executor.submit(_launch)
        self._pw_cm, self._pw, self._browser = await asyncio.wrap_future(future, loop=loop)
        self._started = True
        LOGGER.info(f"[BrowserSlot {self.browser_id}] launched")

    async def close(self) -> None:
        if not self._started:
            self._executor.shutdown(wait=False)
            return

        def _shutdown():
            try:
                if self._browser is not None:
                    self._browser.close()
            finally:
                if self._pw_cm is not None:
                    try:
                        self._pw_cm.__exit__(None, None, None)
                    except Exception:
                        pass

        try:
            loop = asyncio.get_running_loop()
            future = self._executor.submit(_shutdown)
            await asyncio.wait_for(asyncio.wrap_future(future, loop=loop), timeout=30.0)
        except Exception as e:
            LOGGER.warning(f"[BrowserSlot {self.browser_id}] shutdown error: {e}")
        finally:
            self._executor.shutdown(wait=False)
            self._started = False
            LOGGER.info(f"[BrowserSlot {self.browser_id}] closed")

    # ------------------------------------------------------------------
    # Thread-affinitive execution
    # ------------------------------------------------------------------

    async def run(self, fn: Callable, *args, timeout: Optional[float] = None, **kwargs) -> Any:
        """Run `fn(*args, **kwargs)` on this slot's dedicated thread."""
        loop = asyncio.get_running_loop()
        future = self._executor.submit(fn, *args, **kwargs)
        awaitable = asyncio.wrap_future(future, loop=loop)
        if timeout is not None:
            return await asyncio.wait_for(awaitable, timeout=timeout)
        return await awaitable

    @property
    def browser(self):
        """The Playwright Browser. Only touch via `run()`."""
        return self._browser


class BrowserPool:
    """Pool of K BrowserSlots with a total-context semaphore.

    Acquire a slot → caller creates a context on it → on release, slot
    returns to the pool. Slots pick round-robin weighted by `n_contexts`.
    """

    def __init__(self, n_browsers: int, max_contexts_per_browser: int):
        assert n_browsers > 0
        assert max_contexts_per_browser > 0
        self.n_browsers = n_browsers
        self.max_contexts_per_browser = max_contexts_per_browser
        self._slots: List[BrowserSlot] = [BrowserSlot(i) for i in range(n_browsers)]
        self._total_capacity = n_browsers * max_contexts_per_browser
        self._sem = asyncio.Semaphore(self._total_capacity)
        self._assign_lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await asyncio.gather(*(s.start() for s in self._slots))
        self._started = True
        LOGGER.info(
            f"[BrowserPool] started: {self.n_browsers} browsers × "
            f"{self.max_contexts_per_browser} contexts = {self._total_capacity} total"
        )

    async def close(self) -> None:
        if not self._started:
            return
        await asyncio.gather(*(s.close() for s in self._slots), return_exceptions=True)
        self._started = False

    async def acquire_slot(self, timeout: float = 300.0) -> BrowserSlot:
        """Reserve one context slot on the least-loaded browser."""
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"BrowserPool acquire timed out after {timeout}s "
                f"({self.stats_str()})"
            )
        async with self._assign_lock:
            slot = min(self._slots, key=lambda s: s.n_contexts)
            slot.n_contexts += 1
        return slot

    def release_slot(self, slot: BrowserSlot) -> None:
        slot.n_contexts = max(0, slot.n_contexts - 1)
        self._sem.release()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "n_browsers": self.n_browsers,
            "max_contexts_per_browser": self.max_contexts_per_browser,
            "per_browser": [s.n_contexts for s in self._slots],
            "total_active": sum(s.n_contexts for s in self._slots),
            "total_capacity": self._total_capacity,
        }

    def stats_str(self) -> str:
        s = self.stats()
        return f"per_browser={s['per_browser']} total={s['total_active']}/{s['total_capacity']}"
