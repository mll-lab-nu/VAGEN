import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from typing import List, Optional

import ray
from omegaconf import OmegaConf


# Mapping from upload_contents names to file glob patterns within the actor/ directory
_CONTENT_TO_PATTERNS = {
    "model": ["model_world_size_*.pt"],
    "optimizer": ["optim_world_size_*.pt"],
    "extra": ["extra_state_world_size_*.pt"],
    "hf_model": ["huggingface/**"],
}

# Files always included when uploading (metadata needed for checkpoint loading)
_ALWAYS_INCLUDE_PATTERNS = ["fsdp_config.json", "huggingface/*.json", "huggingface/tokenizer*",
                            "huggingface/special_tokens*", "huggingface/added_tokens*",
                            "huggingface/preprocessor_config*", "huggingface/chat_template*"]

VALID_UPLOAD_CONTENTS = set(_CONTENT_TO_PATTERNS.keys())


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

    def upload(self, local_path: str, global_step: int, path_in_repo: str = "",
               allow_patterns: Optional[List[str]] = None):
        """Upload a local folder to HuggingFace Hub.

        Args:
            local_path: Local directory to upload.
            global_step: Current training step.
            path_in_repo: Target path in the HF repo.
            allow_patterns: If set, only files matching these glob patterns are uploaded.

        Returns:
            global_step on success, None on failure.
        """
        if not os.path.exists(local_path):
            print(f"[HFUpload] Warning: {local_path} does not exist, skipping upload")
            return None

        commit_message = (
            f"Upload checkpoint at global_step_{global_step} ({path_in_repo})"
            if path_in_repo
            else f"Upload checkpoint at global_step_{global_step}"
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

        # upload_contents: which checkpoint components to upload
        self._upload_contents: Optional[List[str]] = hf_hub_cfg.get("upload_contents", None)
        self._allow_patterns: Optional[List[str]] = None

        if not self._hf_save_freq:
            return

        # Validate upload_contents
        if self._upload_contents is not None:
            if not isinstance(self._upload_contents, list) or not all(
                isinstance(x, str) for x in self._upload_contents
            ):
                raise ValueError("upload_contents must be a list of strings.")

            invalid = set(self._upload_contents) - VALID_UPLOAD_CONTENTS
            if invalid:
                raise ValueError(
                    f"Invalid upload_contents: {invalid}. "
                    f"Valid values are: {VALID_UPLOAD_CONTENTS}"
                )

            # Validate upload_contents is a subset of save_contents
            save_contents = list(
                config.actor_rollout_ref.actor.checkpoint.get("save_contents", None) or []
            )
            if save_contents:
                not_saved = set(self._upload_contents) - set(save_contents)
                if not_saved:
                    raise ValueError(
                        f"upload_contents {not_saved} are not in actor save_contents {save_contents}. "
                        f"upload_contents must be a subset of save_contents."
                    )

            # Build allow_patterns from upload_contents
            patterns = list(_ALWAYS_INCLUDE_PATTERNS)  # always include metadata
            for content in self._upload_contents:
                patterns.extend(_CONTENT_TO_PATTERNS[content])
            self._allow_patterns = patterns

        save_freq = config.trainer.save_freq
        if save_freq <= 0:
            raise ValueError(
                f"hf_save_freq={self._hf_save_freq} requires save_freq > 0, but got save_freq={save_freq}."
            )
        if self._hf_save_freq % save_freq != 0:
            raise ValueError(
                f"hf_save_freq ({self._hf_save_freq}) must be a multiple of save_freq ({save_freq})."
            )

        # Remove non-actor kwargs before passing to HFUploadActor
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
        """Start a non-blocking upload if conditions are met."""
        if not self.should_upload(global_steps):
            return

        self.flush()

        # Upload the entire actor/ directory (filtered by allow_patterns)
        local_path = os.path.join(
            self._default_local_dir,
            f"global_step_{global_steps}",
            "actor",
        )

        if not os.path.exists(local_path):
            print(
                f"[HFUpload] Warning: {local_path} does not exist at step {global_steps}. "
                f"Make sure hf_save_freq aligns with save_freq."
            )
            return

        path_in_repo = f"{self._project_name}/{self._experiment_name}/global_step_{global_steps}"
        self._pending_future = self._actor.upload.remote(
            local_path=local_path,
            global_step=global_steps,
            path_in_repo=path_in_repo,
            allow_patterns=self._allow_patterns,
        )
        contents_desc = self._upload_contents or "all"
        print(f"[HFUpload] Started async upload for step {global_steps} to {path_in_repo} "
              f"(contents: {contents_desc})")

    def flush(self):
        """Block until any pending upload finishes."""
        if self._pending_future is not None:
            ray.get(self._pending_future)
            self._pending_future = None
