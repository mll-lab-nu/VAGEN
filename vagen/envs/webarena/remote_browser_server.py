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
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Request / Response models ──────────────────────────────────────

class ResetRequest(BaseModel):
    config: dict  # Full task config JSON (not a file path)
    observation_type: str = "accessibility_tree"
    current_viewport_only: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    storage_state: Optional[dict] = None  # Cookie/auth state


class StepRequest(BaseModel):
    action: str  # Raw action string, e.g. "click [42]"


class ResetResponse(BaseModel):
    session_id: str
    observation: str  # accessibility tree text
    url: str
    info: dict


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: dict


class EvalRequest(BaseModel):
    config: dict
    answer: str = ""


class EvalResponse(BaseModel):
    score: float


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
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
        # Close all sessions
        for sid in list(self.sessions):
            session = self.sessions.pop(sid, None)
            if session:
                try:
                    session.browser_env.close()
                except Exception:
                    pass

    def create_session_sync(
        self,
        config: dict,
        observation_type: str,
        current_viewport_only: bool,
        viewport_width: int,
        viewport_height: int,
        storage_state: Optional[dict],
    ) -> BrowserSession:
        """Synchronous session creation — must be called from a thread, not the event loop."""
        # Evict if needed
        if len(self.sessions) >= self.max_sessions:
            oldest_sid = min(self.sessions, key=lambda s: self.sessions[s].last_used)
            logger.warning("Evicting session %s (max sessions reached)", oldest_sid)
            session = self.sessions.pop(oldest_sid, None)
            if session:
                try:
                    session.browser_env.close()
                except Exception:
                    pass

        session_id = str(uuid.uuid4())[:8]

        # Write config to temp file (ScriptBrowserEnv expects file path)
        import tempfile
        config_dir = tempfile.mkdtemp(prefix="webarena_remote_")
        config_path = os.path.join(config_dir, "config.json")

        # Handle storage_state: write to file if provided as dict
        if storage_state:
            state_path = os.path.join(config_dir, "storage_state.json")
            with open(state_path, "w") as f:
                json.dump(storage_state, f)
            config["storage_state"] = state_path
        elif "storage_state" in config:
            # storage_state is a file path - resolve it
            ss = config["storage_state"]
            if ss and not os.path.isabs(ss):
                # Try to resolve relative to server's webarena dir
                webarena_root = os.environ.get("WEBARENA_ROOT", "")
                if webarena_root:
                    config["storage_state"] = os.path.join(webarena_root, ss)

        with open(config_path, "w") as f:
            json.dump(config, f)

        # Create ScriptBrowserEnv
        from vagen.envs.webarena.browser_env.envs import ScriptBrowserEnv

        browser_env = ScriptBrowserEnv(
            headless=True,
            observation_type=observation_type,
            current_viewport_only=current_viewport_only,
            viewport_size={"width": viewport_width, "height": viewport_height},
        )

        # Reset (this launches browser, navigates, etc.)
        obs, info = browser_env.reset(options={"config_file": config_path})

        obs_text = obs.get("text", str(obs)) if isinstance(obs, dict) else str(obs)

        session = BrowserSession(
            session_id=session_id,
            browser_env=browser_env,
            config=config,
            observation_type=observation_type,
            prev_obs_text=obs_text,
        )

        self.sessions[session_id] = session

        # Clean up temp config (browser already loaded it)
        try:
            os.unlink(config_path)
            os.rmdir(config_dir)
        except OSError:
            pass

        return session

    def get_session(self, session_id: str) -> BrowserSession:
        session = self.sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        session.last_used = time.time()
        return session

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            now = time.time()
            expired = [
                sid for sid, s in self.sessions.items()
                if now - s.last_used > self.session_timeout
            ]
            for sid in expired:
                logger.info("Cleaning up expired session %s", sid)
                session = self.sessions.pop(sid, None)
                if session:
                    try:
                        session.browser_env.close()
                    except Exception:
                        pass


# ── FastAPI app ────────────────────────────────────────────────────

session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await session_manager.start()
    yield
    await session_manager.stop()


app = FastAPI(title="WebArena Remote Browser", lifespan=lifespan)


_executor = None

def _get_executor():
    global _executor
    if _executor is None:
        import concurrent.futures
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=32)
    return _executor


@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest):
    """Create a new browser session, navigate to start_url, return accessibility tree."""
    t0 = time.time()
    loop = asyncio.get_event_loop()

    def _do_reset():
        return session_manager.create_session_sync(
            config=req.config,
            observation_type=req.observation_type,
            current_viewport_only=req.current_viewport_only,
            viewport_width=req.viewport_width,
            viewport_height=req.viewport_height,
            storage_state=req.storage_state,
        )

    session = await loop.run_in_executor(_get_executor(), _do_reset)

    elapsed = time.time() - t0
    logger.info("reset session=%s in %.1fs", session.session_id, elapsed)

    return ResetResponse(
        session_id=session.session_id,
        observation=session.prev_obs_text,
        url=session.browser_env.page.url if session.browser_env.page else "",
        info={"reset_time": elapsed},
    )


@app.post("/step/{session_id}", response_model=StepResponse)
async def step(session_id: str, req: StepRequest):
    """Execute an action in the browser and return the new accessibility tree."""
    session = session_manager.get_session(session_id)
    t0 = time.time()
    loop = asyncio.get_event_loop()

    def _do_step():
        from vagen.envs.webarena.browser_env import create_id_based_action
        try:
            action = create_id_based_action(req.action)
            obs, reward, terminated, truncated, info = session.browser_env.step(action)
        except Exception as e:
            obs = {"text": f"Action failed: {e}"}
            reward = 0.0
            terminated = False
            info = {"fail_error": str(e)}
        return obs, reward, terminated, info

    obs, reward, terminated, info = await loop.run_in_executor(_get_executor(), _do_step)

    obs_text = obs.get("text", str(obs)) if isinstance(obs, dict) else str(obs)
    obs_changed = obs_text != session.prev_obs_text
    session.prev_obs_text = obs_text

    elapsed = time.time() - t0
    info["step_time"] = elapsed
    info["obs_changed"] = obs_changed

    # Serialize info (remove non-serializable objects)
    clean_info = {}
    for k, v in info.items():
        if k == "page":
            clean_info["url"] = v.url if hasattr(v, "url") else str(v)
        elif k == "observation_metadata":
            continue  # Skip, not needed remotely
        else:
            clean_info[k] = v

    return StepResponse(
        observation=obs_text,
        reward=reward,
        done=terminated,
        info=clean_info,
    )


@app.post("/eval/{session_id}", response_model=EvalResponse)
async def evaluate(session_id: str, req: EvalRequest):
    """Run WebArena evaluator on the current page state."""
    session = session_manager.get_session(session_id)
    loop = asyncio.get_event_loop()

    def _do_eval():
        from vagen.envs.webarena.evaluation_harness.evaluators import evaluator_router
        import tempfile

        config_dir = tempfile.mkdtemp(prefix="webarena_eval_")
        config_path = os.path.join(config_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(req.config, f)

        try:
            evaluator = evaluator_router(config_path)
            last_action = {"answer": req.answer}
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
        return score

    score = await loop.run_in_executor(_get_executor(), _do_eval)
    return EvalResponse(score=score)


@app.delete("/session/{session_id}")
async def close(session_id: str):
    """Close a browser session."""
    loop = asyncio.get_event_loop()

    def _do_close():
        session = session_manager.sessions.pop(session_id, None)
        if session:
            try:
                session.browser_env.close()
            except Exception:
                pass

    await loop.run_in_executor(_get_executor(), _do_close)
    return {"status": "ok"}


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "active_sessions": len(session_manager.sessions),
    }


@app.get("/stats")
async def stats():
    """Return server statistics."""
    sessions_info = []
    for sid, s in session_manager.sessions.items():
        sessions_info.append({
            "session_id": sid,
            "age_s": time.time() - s.created_at,
            "idle_s": time.time() - s.last_used,
        })
    return {
        "active_sessions": len(session_manager.sessions),
        "max_sessions": session_manager.max_sessions,
        "sessions": sessions_info,
    }


def main():
    parser = argparse.ArgumentParser(description="WebArena Remote Browser Server")
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--max-sessions", type=int, default=32)
    parser.add_argument("--session-timeout", type=float, default=600.0)
    args = parser.parse_args()

    session_manager.max_sessions = args.max_sessions
    session_manager.session_timeout = args.session_timeout

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()