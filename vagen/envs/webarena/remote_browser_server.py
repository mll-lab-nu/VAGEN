"""
Remote Playwright server for WebArena.

Run this on the same machine as WebArena Docker containers.
Training machines send lightweight HTTP requests instead of
running Playwright locally (eliminates network latency for
the many sub-requests Playwright makes per step).

Usage:
    python -m vagen.envs.webarena.remote_browser_server --port 5100 --max-sessions 32
"""

import argparse
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, abort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ── Session management ─────────────────────────────────────────────

@dataclass
class BrowserSession:
    session_id: str
    browser_env: Any  # ScriptBrowserEnv
    config: dict
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    observation_type: str = "accessibility_tree"
    prev_obs_text: str = ""


class SessionManager:
    def __init__(self, max_sessions: int = 32, session_timeout: float = 600.0):
        self.sessions: Dict[str, BrowserSession] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self._lock = threading.Lock()

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

        session_id = str(uuid.uuid4())[:8]

        # Write config to temp file (ScriptBrowserEnv expects file path)
        config_dir = tempfile.mkdtemp(prefix="webarena_remote_")
        config_path = os.path.join(config_dir, "config.json")

        # Handle storage_state
        if storage_state:
            state_path = os.path.join(config_dir, "storage_state.json")
            with open(state_path, "w") as f:
                json.dump(storage_state, f)
            config["storage_state"] = state_path
        elif "storage_state" in config:
            ss = config["storage_state"]
            if ss and not os.path.isabs(ss):
                webarena_root = os.environ.get("WEBARENA_ROOT", "")
                if webarena_root:
                    config["storage_state"] = os.path.join(webarena_root, ss)

        with open(config_path, "w") as f:
            json.dump(config, f)

        # Create and reset ScriptBrowserEnv
        from vagen.envs.webarena.browser_env.envs import ScriptBrowserEnv

        browser_env = ScriptBrowserEnv(
            headless=True,
            observation_type=observation_type,
            current_viewport_only=current_viewport_only,
            viewport_size={"width": viewport_width, "height": viewport_height},
        )

        obs, info = browser_env.reset(options={"config_file": config_path})

        obs_text = obs.get("text", str(obs)) if isinstance(obs, dict) else str(obs)

        session = BrowserSession(
            session_id=session_id,
            browser_env=browser_env,
            config=config,
            observation_type=observation_type,
            prev_obs_text=obs_text,
        )

        with self._lock:
            self.sessions[session_id] = session

        # Clean up temp config
        try:
            os.unlink(config_path)
            os.rmdir(config_dir)
        except OSError:
            pass

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
            try:
                session.browser_env.close()
            except Exception:
                pass

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
    logger.info("reset session=%s in %.1fs", session.session_id, elapsed)

    url = ""
    try:
        url = session.browser_env.page.url
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

    from vagen.envs.webarena.browser_env import create_id_based_action

    try:
        action = create_id_based_action(action_str)
        obs, reward, terminated, truncated, info = session.browser_env.step(action)
    except Exception as e:
        obs = {"text": f"Action failed: {e}"}
        reward = 0.0
        terminated = False
        info = {"fail_error": str(e)}

    obs_text = obs.get("text", str(obs)) if isinstance(obs, dict) else str(obs)
    obs_changed = obs_text != session.prev_obs_text
    session.prev_obs_text = obs_text

    elapsed = time.time() - t0

    # Serialize info (remove non-serializable objects)
    clean_info = {"step_time": elapsed, "obs_changed": obs_changed}
    for k, v in info.items():
        if k == "page":
            clean_info["url"] = v.url if hasattr(v, "url") else str(v)
        elif k == "observation_metadata":
            continue
        else:
            try:
                json.dumps(v)
                clean_info[k] = v
            except (TypeError, ValueError):
                clean_info[k] = str(v)

    return jsonify({
        "observation": obs_text,
        "reward": reward,
        "done": terminated,
        "info": clean_info,
    })


@app.route("/eval/<session_id>", methods=["POST"])
def evaluate(session_id):
    session = session_manager.get_session(session_id)
    req = request.get_json()

    from vagen.envs.webarena.evaluation_harness.evaluators import evaluator_router

    config_dir = tempfile.mkdtemp(prefix="webarena_eval_")
    config_path = os.path.join(config_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(req.get("config", {}), f)

    try:
        evaluator = evaluator_router(config_path)
        last_action = {"answer": req.get("answer", "")}
        trajectory = [last_action]

        page = session.browser_env.page
        client = page.context.new_cdp_session(page)
        try:
            score = evaluator(
                trajectory=trajectory,
                config_file=config_path,
                page=page,
                client=client,
            )
        finally:
            client.detach()
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        score = 0.0
    finally:
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
    return jsonify({
        "status": "ok",
        "active_sessions": len(session_manager.sessions),
    })


@app.route("/stats", methods=["GET"])
def stats():
    sessions_info = []
    for sid, s in session_manager.sessions.items():
        sessions_info.append({
            "session_id": sid,
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
    parser.add_argument("--threaded", action="store_true", default=True,
                        help="Handle requests in threads (default: True)")
    args = parser.parse_args()

    session_manager.max_sessions = args.max_sessions
    session_manager.session_timeout = args.session_timeout

    logger.info(
        "Starting WebArena Remote Browser Server on %s:%d (max_sessions=%d, threaded=%s)",
        args.host, args.port, args.max_sessions, args.threaded,
    )
    app.run(host=args.host, port=args.port, threaded=args.threaded)


if __name__ == "__main__":
    main()