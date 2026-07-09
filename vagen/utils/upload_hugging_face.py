import os

from typing import List, Optional

import ray
from omegaconf import OmegaConf


@ray.remote(num_cpus=1)
class HFUploadActor:
    """Ray actor for non-blocking model uploads to HuggingFace Hub."""

    def __init__(self, repo_id: str, private: bool = True, **kwargs):
        from huggingface_hub import HfApi

        # Enable hf_transfer by default for uploads, without affecting processes
        # that import this module but never create an upload actor.
        if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        api_kwargs = {}
        if "token" in kwargs:
            api_kwargs["token"] = kwargs["token"]
        if "endpoint" in kwargs:
            api_kwargs["endpoint"] = kwargs["endpoint"]

        self.api = HfApi(**api_kwargs)

        # Auto-prefix with username if repo_id has no '/'
        if "/" not in repo_id:
            try:
                username = self.api.whoami()["name"]
                repo_id = f"{username}/{repo_id}"
            except Exception:
                pass
        self.repo_id = repo_id

        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type="model",
            exist_ok=True,
            private=private,
        )
        print(f"[HFUpload] Initialized upload actor for repo: {self.repo_id}")

    def upload(self, local_path: str, label: str, path_in_repo: str = "",
               allow_patterns: Optional[List[str]] = None):
        """Upload a local folder to HuggingFace Hub.

        Args:
            local_path: Local directory to upload (the "root" against which
                ``allow_patterns`` is interpreted).
            label: Short identifier used in the commit message (e.g. step
                number or ``"best_val"``).
            path_in_repo: Target path inside the HF repo.
            allow_patterns: Glob patterns relative to ``local_path``. Only
                matching files are uploaded. Folder patterns are expressed as
                ``"<dir>/**"`` (e.g. ``"actor/huggingface/**"``); file
                patterns use the file glob directly (e.g. ``"config.json"``).

        Returns:
            label on success, None on failure.
        """
        if not os.path.exists(local_path):
            print(f"[HFUpload] Warning: {local_path} does not exist, skipping upload")
            return None

        commit_message = (
            f"Upload checkpoint {label} ({path_in_repo})"
            if path_in_repo
            else f"Upload checkpoint {label}"
        )
        try:
            self.api.upload_folder(
                repo_id=self.repo_id,
                folder_path=local_path,
                path_in_repo=path_in_repo,
                commit_message=commit_message,
                repo_type="model",
                allow_patterns=allow_patterns,
            )
            print(f"[HFUpload] Successfully uploaded {local_path} to {self.repo_id} [{label}]")
            return label
        except Exception as e:
            print(f"[HFUpload] Error uploading to {self.repo_id} [{label}]: {e}")
            return None


class HFUploadManager:
    """Manages non-blocking HuggingFace Hub uploads during training.

    Two upload triggers are supported:

    1. Periodic step uploads via :meth:`maybe_upload(global_steps)` —
       fires every ``hf_save_freq`` steps and uploads from the
       per-step checkpoint folder.

    2. On-demand best-validation uploads via
       :meth:`maybe_upload_best_val(global_steps, local_path)` —
       fires whenever the trainer detects a new best validation
       score and writes the model to a ``best_val/`` directory.

    Both reuse one ``_pending_future`` slot, so ``flush()`` waits on
    whichever upload was started most recently.

    Usage in trainer::

        self.hf_upload_manager = HFUploadManager(config)

        # In the training loop, at save-checkpoint time:
        self.hf_upload_manager.flush()          # wait for previous upload
        self._save_checkpoint()
        self.hf_upload_manager.maybe_upload(global_steps)

        # When a new best validation score is detected:
        self.hf_upload_manager.maybe_upload_best_val(global_steps, best_val_path)

        # At the end of training:
        self.hf_upload_manager.flush()
    """

    def __init__(self, config):
        hf_hub_cfg = OmegaConf.to_container(config.get("huggingface_hub", {}), resolve=True) or {}
        self._hf_save_freq: Optional[int] = hf_hub_cfg.get("hf_save_freq", None)
        self._actor: Optional[ray.actor.ActorHandle] = None
        self._pending_future = None
        self._default_local_dir: str = config.trainer.default_local_dir
        self._project_name: str = config.trainer.get("project_name", "default_project")
        self._experiment_name: str = config.trainer.get("experiment_name", "default_experiment")

        # ``upload_contents``: raw HF glob patterns interpreted relative to
        # ``<default_local_dir>/global_step_<N>/`` (or ``best_val/``).
        # Examples:
        #   ['actor/huggingface/**']   → just the HF model under actor/
        #   ['actor/**']               → everything under actor/
        #   ['actor/huggingface/**', 'data.pt']  → HF model + dataloader state
        # ``None`` uploads everything (no allow_patterns filter).
        self._upload_contents: Optional[List[str]] = hf_hub_cfg.get("upload_contents", None)

        if not self._hf_save_freq:
            return

        if self._upload_contents is not None:
            if not isinstance(self._upload_contents, list) or not all(
                isinstance(x, str) for x in self._upload_contents
            ):
                raise ValueError("upload_contents must be a list of glob-pattern strings.")

        save_freq = config.trainer.save_freq
        if save_freq <= 0:
            raise ValueError(
                f"hf_save_freq={self._hf_save_freq} requires save_freq > 0, but got save_freq={save_freq}."
            )
        if self._hf_save_freq % save_freq != 0:
            raise ValueError(
                f"hf_save_freq ({self._hf_save_freq}) must be a multiple of save_freq ({save_freq})."
            )

        # Strip manager-only keys before forwarding to the actor.
        actor_kwargs = {k: v for k, v in hf_hub_cfg.items()
                        if k not in ("hf_save_freq", "upload_contents")}
        if actor_kwargs.get("repo_id"):
            self._actor = HFUploadActor.remote(**actor_kwargs)
        else:
            print("Warning: hf_save_freq is set but huggingface_hub.repo_id is missing, skipping HF upload.")

    @property
    def enabled(self) -> bool:
        return self._hf_save_freq is not None and self._actor is not None

    def should_upload(self, global_steps: int) -> bool:
        return self.enabled and global_steps % self._hf_save_freq == 0

    def maybe_upload(self, global_steps: int):
        """Start a non-blocking upload of the current step's checkpoint."""
        if not self.should_upload(global_steps):
            return

        self.flush()

        # ``upload_contents`` patterns are relative to the verl checkpoint
        # root for this step:
        #   <default_local_dir>/global_step_<N>/
        # i.e. the same ``local_path`` that ray_trainer._save_checkpoint
        # uses as ``local_global_step_folder`` (see ray_trainer.py).
        local_path = os.path.join(self._default_local_dir, f"global_step_{global_steps}")
        if not os.path.exists(local_path):
            print(
                f"[HFUpload] Warning: {local_path} does not exist at step {global_steps}. "
                f"Make sure hf_save_freq aligns with save_freq."
            )
            return

        path_in_repo = f"{self._project_name}/{self._experiment_name}/global_step_{global_steps}"
        self._pending_future = self._actor.upload.remote(
            local_path=local_path,
            label=f"global_step_{global_steps}",
            path_in_repo=path_in_repo,
            allow_patterns=self._upload_contents,
        )
        print(f"[HFUpload] Started async upload for step {global_steps} to {path_in_repo} "
              f"(patterns: {self._upload_contents or 'all'})")

    def maybe_upload_best_val(self, global_steps: int, local_path: str):
        """Start a non-blocking upload of ``local_path`` as the new best-val model.

        Same ``upload_contents`` glob patterns apply, but they are now
        interpreted relative to ``local_path`` (e.g.
        ``<default_local_dir>/best_val/``). No-op if HF upload is disabled.
        """
        if not self.enabled:
            return
        if not os.path.exists(local_path):
            print(f"[HFUpload] Warning: {local_path} does not exist; skipping best-val upload.")
            return

        self.flush()
        path_in_repo = f"{self._project_name}/{self._experiment_name}/best_val"
        self._pending_future = self._actor.upload.remote(
            local_path=local_path,
            label=f"best_val (step {global_steps})",
            path_in_repo=path_in_repo,
            allow_patterns=self._upload_contents,
        )
        print(f"[HFUpload] Started async best-val upload (step {global_steps}) to {path_in_repo}")

    def flush(self):
        """Block until any pending upload finishes."""
        if self._pending_future is not None:
            ray.get(self._pending_future)
            self._pending_future = None
