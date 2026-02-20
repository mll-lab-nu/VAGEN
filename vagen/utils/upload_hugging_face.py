import os
from typing import Optional

import ray
from omegaconf import OmegaConf


@ray.remote(num_cpus=1)
class HFUploadActor:
    """Ray actor for non-blocking model uploads to HuggingFace Hub."""

    def __init__(self, repo_id: str, private: bool = True, **kwargs):
        from huggingface_hub import HfApi

        api_kwargs = {}
        if "token" in kwargs:
            api_kwargs["token"] = kwargs["token"]
        if "endpoint" in kwargs:
            api_kwargs["endpoint"] = kwargs["endpoint"]

        self.api = HfApi(**api_kwargs)
        self.repo_id = repo_id

        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type="model",
            exist_ok=True,
            private=private,
        )
        print(f"[HFUpload] Initialized upload actor for repo: {self.repo_id}")

    def upload(self, local_path: str, global_step: int, path_in_repo: str = ""):
        """Upload a local folder to HuggingFace Hub.

        Returns:
            global_step on success, None on failure.
        """
        if not os.path.exists(local_path):
            print(f"[HFUpload] Warning: {local_path} does not exist, skipping upload")
            return None

        commit_message = (
            f"Upload model at global_step_{global_step} ({path_in_repo})"
            if path_in_repo
            else f"Upload model at global_step_{global_step}"
        )
        try:
            self.api.upload_folder(
                repo_id=self.repo_id,
                folder_path=local_path,
                path_in_repo=path_in_repo,
                commit_message=commit_message,
                repo_type="model",
            )
            print(f"[HFUpload] Successfully uploaded {local_path} to {self.repo_id} at step {global_step}")
            return global_step
        except Exception as e:
            print(f"[HFUpload] Error uploading to {self.repo_id} at step {global_step}: {e}")
            return None


class HFUploadManager:
    """Manages non-blocking HuggingFace Hub uploads during training.

    Usage in trainer:
        self.hf_upload_manager = HFUploadManager(config)

        # In the training loop, at save-checkpoint time:
        self.hf_upload_manager.flush()          # wait for previous upload before ckpt deletion
        self._save_checkpoint()
        self.hf_upload_manager.maybe_upload(global_steps)

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

        if not self._hf_save_freq:
            return

        save_freq = config.trainer.save_freq
        if save_freq <= 0:
            raise ValueError(
                f"hf_save_freq={self._hf_save_freq} requires save_freq > 0, but got save_freq={save_freq}."
            )
        if self._hf_save_freq % save_freq != 0:
            raise ValueError(
                f"hf_save_freq ({self._hf_save_freq}) must be a multiple of save_freq ({save_freq})."
            )

        # Remove hf_save_freq from the dict before passing to HFUploadActor
        actor_kwargs = {k: v for k, v in hf_hub_cfg.items() if k != "hf_save_freq"}
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
        """Start a non-blocking upload if conditions are met."""
        if not self.should_upload(global_steps):
            return

        self.flush()

        local_path = os.path.join(
            self._default_local_dir,
            f"global_step_{global_steps}",
            "actor",
            "huggingface",
        )

        if not os.path.exists(local_path):
            print(
                f"[HFUpload] Warning: {local_path} does not exist at step {global_steps}. "
                f"Make sure hf_save_freq aligns with save_freq and the checkpoint saves hf_model."
            )
            return

        path_in_repo = f"{self._project_name}/{self._experiment_name}/global_step_{global_steps}"
        self._pending_future = self._actor.upload.remote(
            local_path=local_path,
            global_step=global_steps,
            path_in_repo=path_in_repo,
        )
        print(f"[HFUpload] Started async upload for step {global_steps} to {path_in_repo}")

    def flush(self):
        """Block until any pending upload finishes."""
        if self._pending_future is not None:
            ray.get(self._pending_future)
            self._pending_future = None
