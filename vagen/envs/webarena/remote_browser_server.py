"""
Remote Playwright server for WebArena.

Run this on the same machine as WebArena Docker containers.
Training machines send lightweight HTTP requests instead of
running Playwright locally (eliminates network latency for
the many sub-requests Playwright makes per step).

Key optimisation: Chromium browser processes are pooled and reused
across sessions. Only the browser *context* (cookies, pages) is
created/destroyed per reset, saving ~1-2s of browser launch overhead.

Usage:
    python -m vagen.envs.webarena.remote_browser_server --port 5100 --max-sessions 32
"""

import argparse
import concurrent.futures
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, abort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ── Browser pool ──────────────────────────────────────────────────

class BrowserPool:
    """Pool of reusable Chromium browser instances.

    Each browser process can host multiple contexts (sessions).
    This avoids the ~1-2s cost of launching a new Chromium per reset.

    All Playwright calls are dispatched to a dedicated worker thread
    that has no asyncio event loop, avoiding the "Sync API inside
    asyncio loop" error.
    """

    def __init__(self, pool_size: int = 4):
        self.pool_size = pool_size
        self._lock = threading.Lock()
        self._browsers: List[Dict[str, Any]] = []
        self._initialized = False
        # Dedicated thread pool for Playwright sync API calls.
        # Multiple workers are needed for concurrent reset/step requests.
        # Each worker thread gets its own "clean" context without asyncio loop.
        self._pw_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=pool_size * 2, thread_name_prefix="playwright-pool"
        )

    def _run_in_pw_thread(self, fn, *args, **kwargs):
        """Run a function in the dedicated Playwright thread."""
        future = self._pw_executor.submit(fn, *args, **kwargs)
        return future.result(timeout=120)

    def initialize(self):
        """Pre-launch browser processes in the dedicated thread."""
        with self._lock:
            if self._initialized:
                return

        def _do_init():
            from playwright.sync_api import sync_playwright
            logger.info("Initializing browser pool with %d browsers...", self.pool_size)
            for i in range(self.pool_size):
                entry = self._do_launch_browser()
                with self._lock:
                    self._browsers.append(entry)
                logger.info("  Browser %d/%d launched", i + 1, self.pool_size)
            with self._lock:
                self._initialized = True
            logger.info("Browser pool ready.")

        self._run_in_pw_thread(_do_init)

    def _do_launch_browser(self) -> Dict[str, Any]:
        """Must be called from the Playwright thread."""
        from playwright.sync_api import sync_playwright
        cm = sync_playwright()
        pw = cm.__enter__()
        browser = pw.chromium.launch(
            headless=True,
            args=["--blink-settings=imagesEnabled=false"],
        )
        return {
            "context_manager": cm,
            "playwright": pw,
            "browser": browser,
            "session_count": 0,
        }

    def acquire(self) -> Tuple[Any, int]:
        """Get the least-loaded browser. Returns (browser, pool_index)."""
        with self._lock:
            if not self._initialized:
                raise RuntimeError("Browser pool not initialized. Call initialize() first.")

            min_idx = min(range(len(self._browsers)),
                          key=lambda i: self._browsers[i]["session_count"])
            entry = self._browsers[min_idx]
            entry["session_count"] += 1
            return entry["browser"], min_idx

    def release(self, pool_index: int):
        """Decrement session count for a browser."""
        with self._lock:
            if 0 <= pool_index < len(self._browsers):
                self._browsers[pool_index]["session_count"] = max(
                    0, self._browsers[pool_index]["session_count"] - 1
                )

    def _close_browser(self, entry: Dict[str, Any]):
        try:
            entry["browser"].close()
        except Exception:
            pass
        try:
            entry["context_manager"].__exit__(None, None, None)
        except Exception:
            pass

    def close_all(self):
        with self._lock:
            for entry in self._browsers:
                self._close_browser(entry)
            self._browsers.clear()
            self._initialized = False
        self._pw_executor.shutdown(wait=False)


# ── Session management ─────────────────────────────────────────────

@dataclass
class BrowserSession:
    session_id: str
    pool_index: int           # which browser in the pool
    context: Any              # BrowserContext
    page: Any                 # Page
    client: Any               # CDPSession
    observation_handler: Any  # ObservationHandler
    config: dict
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    observation_type: str = "accessibility_tree"
    prev_obs_text: str = ""


class SessionManager:
    def __init__(self, max_sessions: int = 32, session_timeout: float = 600.0,
                 browser_pool_size: int = 4):
        self.sessions: Dict[str, BrowserSession] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self._lock = threading.Lock()
        self._reset_semaphore = threading.Semaphore(8)  # Max concurrent resets
        self.browser_pool = BrowserPool(pool_size=browser_pool_size)

    def create_session(
        self,
        config: dict,
        observation_type: str,
        current_viewport_only: bool,
        viewport_width: int,
        viewport_height: int,
        storage_state: Optional[dict],
    ) -> BrowserSession:
        with self._lock:
            if len(self.sessions) >= self.max_sessions:
                oldest_sid = min(self.sessions, key=lambda s: self.sessions[s].last_used)
                logger.warning("Evicting session %s (max sessions reached)", oldest_sid)
                self._close_session(oldest_sid)

        # Limit concurrent resets to avoid resource exhaustion
        with self._reset_semaphore:
            session_id = str(uuid.uuid4())[:8]

            # Handle storage_state: write to temp file if provided as dict
            storage_state_path = None
            tmp_dir = None
            if storage_state:
                tmp_dir = tempfile.mkdtemp(prefix="webarena_remote_")
                storage_state_path = os.path.join(tmp_dir, "storage_state.json")
                with open(storage_state_path, "w") as f:
                    json.dump(storage_state, f)
            elif "storage_state" in config:
                ss = config.get("storage_state")
                if ss and not os.path.isabs(ss):
                    webarena_root = os.environ.get("WEBARENA_ROOT", "")
                    if webarena_root:
                        storage_state_path = os.path.join(webarena_root, ss)
                elif ss:
                    storage_state_path = ss

            # Acquire a browser from the pool (no launch cost!)
            browser, pool_index = self.browser_pool.acquire()

            geolocation = config.get("geolocation", None)
            viewport = {"width": viewport_width, "height": viewport_height}
            start_url = config.get("start_url", None)

            # All Playwright calls run in the dedicated thread
            def _do_create():
                context = browser.new_context(
                    viewport=viewport,
                    storage_state=storage_state_path,
                    geolocation=geolocation,
                    device_scale_factor=1,
                )

                # Navigate to start URL(s)
                if start_url:
                    start_urls = start_url.split(" |AND| ")
                    for url in start_urls:
                        page = context.new_page()
                        client = page.context.new_cdp_session(page)
                        if observation_type == "accessibility_tree":
                            client.send("Accessibility.enable")
                        page.client = client  # type: ignore
                        page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    active_page = context.pages[0]
                    active_page.bring_to_front()
                else:
                    active_page = context.new_page()
                    client = active_page.context.new_cdp_session(active_page)
                    if observation_type == "accessibility_tree":
                        client.send("Accessibility.enable")
                    active_page.client = client  # type: ignore

                active_client = active_page.client  # type: ignore

                # Create observation handler
                from vagen.envs.webarena.browser_env.processors import ObservationHandler
                obs_handler = ObservationHandler(
                    "text", observation_type, "",
                    current_viewport_only=current_viewport_only,
                    viewport_size=viewport,
                )

                # Get initial observation
                obs = obs_handler.get_observation(active_page, active_client)
                obs_text = obs.get("text", str(obs)) if isinstance(obs, dict) else str(obs)

                return context, active_page, active_client, obs_handler, obs_text

            context, active_page, active_client, obs_handler, obs_text = \
                self.browser_pool._run_in_pw_thread(_do_create)

            # Clean up temp storage_state file
            if tmp_dir:
                try:
                    os.unlink(storage_state_path)
                    os.rmdir(tmp_dir)
                except OSError:
                    pass

            session = BrowserSession(
                session_id=session_id,
                pool_index=pool_index,
                context=context,
                page=active_page,
                client=active_client,
                observation_handler=obs_handler,
                config=config,
                observation_type=observation_type,
                prev_obs_text=obs_text,
            )

            with self._lock:
                self.sessions[session_id] = session

            return session

    def get_session(self, session_id: str) -> BrowserSession:
        session = self.sessions.get(session_id)
        if not session:
            abort(404, description=f"Session {session_id} not found")
        session.last_used = time.time()
        return session

    def close_session(self, session_id: str):
        with self._lock:
            self._close_session(session_id)

    def _close_session(self, session_id: str):
        session = self.sessions.pop(session_id, None)
        if session:
            def _do_close():
                try:
                    session.client.detach()
                except Exception:
                    pass
                try:
                    session.context.close()
                except Exception:
                    pass
            try:
                self.browser_pool._run_in_pw_thread(_do_close)
            except Exception:
                pass
            self.browser_pool.release(session.pool_index)

    def cleanup_expired(self):
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, s in self.sessions.items()
                if now - s.last_used > self.session_timeout
            ]
            for sid in expired:
                logger.info("Cleaning up expired session %s", sid)
                self._close_session(sid)

    def close_all(self):
        with self._lock:
            for sid in list(self.sessions):
                self._close_session(sid)
        self.browser_pool.close_all()


session_manager = SessionManager()


# ── Cleanup thread ─────────────────────────────────────────────────

def _cleanup_loop():
    while True:
        time.sleep(60)
        try:
            session_manager.cleanup_expired()
        except Exception as e:
            logger.error("Cleanup error: %s", e)


cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
cleanup_thread.start()


# ── Routes ─────────────────────────────────────────────────────────

@app.route("/reset", methods=["POST"])
def reset():
    req = request.get_json()
    t0 = time.time()

    try:
        session = session_manager.create_session(
            config=req.get("config", {}),
            observation_type=req.get("observation_type", "accessibility_tree"),
            current_viewport_only=req.get("current_viewport_only", True),
            viewport_width=req.get("viewport_width", 1280),
            viewport_height=req.get("viewport_height", 720),
            storage_state=req.get("storage_state"),
        )
    except Exception as e:
        logger.error("Reset failed: %s", e)
        return jsonify({"error": str(e)}), 500

    elapsed = time.time() - t0
    logger.info("reset session=%s in %.1fs (pool_idx=%d)",
                session.session_id, elapsed, session.pool_index)

    url = ""
    try:
        url = session.page.url
    except Exception:
        pass

    return jsonify({
        "session_id": session.session_id,
        "observation": session.prev_obs_text,
        "url": url,
        "info": {"reset_time": elapsed},
    })


@app.route("/step/<session_id>", methods=["POST"])
def step(session_id):
    session = session_manager.get_session(session_id)
    req = request.get_json()
    action_str = req.get("action", "")
    t0 = time.time()

    def _do_step():
        from vagen.envs.webarena.browser_env import create_id_based_action
        from vagen.envs.webarena.browser_env.actions import execute_action

        _fail_error = ""
        try:
            action = create_id_based_action(action_str)
            session.page = execute_action(
                action,
                session.page,
                session.context,
                session.observation_handler.action_processor,
            )
            # Update CDP client if page changed (e.g. new_tab, tab_focus)
            if not hasattr(session.page, 'client') or session.page.client is None:
                client = session.page.context.new_cdp_session(session.page)
                if session.observation_type == "accessibility_tree":
                    client.send("Accessibility.enable")
                session.page.client = client  # type: ignore
                session.client = client
        except Exception as e:
            _fail_error = str(e)

        # Get observation
        _obs_text = ""
        try:
            obs = session.observation_handler.get_observation(session.page, session.client)
            _obs_text = obs.get("text", str(obs)) if isinstance(obs, dict) else str(obs)
        except Exception as e:
            _obs_text = f"Observation failed: {e}"

        _url = ""
        try:
            _url = session.page.url
        except Exception:
            pass

        return _obs_text, _fail_error, _url

    obs_text, fail_error, url = session_manager.browser_pool._run_in_pw_thread(_do_step)

    obs_changed = obs_text != session.prev_obs_text
    session.prev_obs_text = obs_text

    elapsed = time.time() - t0

    clean_info = {
        "step_time": elapsed,
        "obs_changed": obs_changed,
        "fail_error": fail_error,
        "url": url,
    }

    return jsonify({
        "observation": obs_text,
        "reward": 0.0 if fail_error else 1.0,
        "done": False,
        "info": clean_info,
    })


@app.route("/eval/<session_id>", methods=["POST"])
def evaluate(session_id):
    session = session_manager.get_session(session_id)
    req = request.get_json()

    config_dir = tempfile.mkdtemp(prefix="webarena_eval_")
    config_path = os.path.join(config_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(req.get("config", {}), f)

    def _do_eval():
        from vagen.envs.webarena.evaluation_harness.evaluators import evaluator_router
        try:
            evaluator = evaluator_router(config_path)
            last_action = {"answer": req.get("answer", "")}
            trajectory = [last_action]

            page = session.page
            client = page.context.new_cdp_session(page)
            try:
                return evaluator(
                    trajectory=trajectory,
                    config_file=config_path,
                    page=page,
                    client=client,
                )
            finally:
                client.detach()
        except Exception as e:
            logger.error("Evaluation failed: %s", e)
            return 0.0

    score = session_manager.browser_pool._run_in_pw_thread(_do_eval)

    try:
        os.unlink(config_path)
        os.rmdir(config_dir)
    except OSError:
        pass

    return jsonify({"score": score})


@app.route("/session/<session_id>", methods=["DELETE"])
def close(session_id):
    session_manager.close_session(session_id)
    return jsonify({"status": "ok"})


@app.route("/health", methods=["GET"])
def health():
    pool_info = []
    for i, entry in enumerate(session_manager.browser_pool._browsers):
        pool_info.append({
            "index": i,
            "active_sessions": entry["session_count"],
        })
    return jsonify({
        "status": "ok",
        "active_sessions": len(session_manager.sessions),
        "browser_pool": pool_info,
    })


@app.route("/stats", methods=["GET"])
def stats():
    sessions_info = []
    for sid, s in session_manager.sessions.items():
        sessions_info.append({
            "session_id": sid,
            "pool_index": s.pool_index,
            "age_s": round(time.time() - s.created_at, 1),
            "idle_s": round(time.time() - s.last_used, 1),
        })
    return jsonify({
        "active_sessions": len(session_manager.sessions),
        "max_sessions": session_manager.max_sessions,
        "sessions": sessions_info,
    })


def main():
    parser = argparse.ArgumentParser(description="WebArena Remote Browser Server")
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--max-sessions", type=int, default=32)
    parser.add_argument("--session-timeout", type=float, default=600.0)
    parser.add_argument("--browser-pool-size", type=int, default=4,
                        help="Number of Chromium processes to keep alive (default: 4)")
    parser.add_argument("--threaded", action="store_true", default=True,
                        help="Handle requests in threads (default: True)")
    args = parser.parse_args()

    session_manager.max_sessions = args.max_sessions
    session_manager.session_timeout = args.session_timeout
    session_manager.browser_pool = BrowserPool(pool_size=args.browser_pool_size)

    # Pre-launch browsers before accepting requests
    session_manager.browser_pool.initialize()

    logger.info(
        "Starting WebArena Remote Browser Server on %s:%d "
        "(max_sessions=%d, browser_pool=%d, threaded=%s)",
        args.host, args.port, args.max_sessions,
        args.browser_pool_size, args.threaded,
    )
    app.run(host=args.host, port=args.port, threaded=args.threaded)


if __name__ == "__main__":
    main()