"""Uploading checkpoints to the HuggingFace Hub is optional; failing to must not end training.

The checkpoint being uploaded is already on disk. Before this, `create_repo` ran in the
upload actor's `__init__`: with `HF_HUB_OFFLINE=1` (or no network, or a bad token) it
raised, the actor died in its creation task, and the trainer died with it on the next
`flush()` -- which runs before the save of every upload step and at the end of training.
"""

from __future__ import annotations

import huggingface_hub
import ray
from huggingface_hub.errors import OfflineModeIsEnabled
from omegaconf import OmegaConf

from vagen.utils import upload_hugging_face as hf


class _OfflineApi:
    """What HfApi does with HF_HUB_OFFLINE=1: every call that needs the network raises."""

    created = 0

    def __init__(self, **kwargs):
        self.uploads = []

    def whoami(self):
        raise OfflineModeIsEnabled("offline mode is enabled")

    def create_repo(self, **kwargs):
        raise OfflineModeIsEnabled("offline mode is enabled")

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)


class _OnlineApi(_OfflineApi):
    def create_repo(self, **kwargs):
        type(self).created += 1


def _actor(monkeypatch, api):
    monkeypatch.setattr(huggingface_hub, "HfApi", api)
    # The class under the @ray.remote, constructed in-process: no Ray cluster needed.
    return hf.HFUploadActor.__ray_actor_class__("someone/run")


def test_an_unreachable_hub_does_not_kill_the_upload_actor(monkeypatch, tmp_path):
    """★ The constructor must not touch the network: an exception there is a dead actor."""
    actor = _actor(monkeypatch, _OfflineApi)
    assert actor.upload(str(tmp_path), global_step=10) is None, "a failed upload reports None"
    assert actor.api.uploads == []


def test_the_repo_is_created_once_by_the_first_upload(monkeypatch, tmp_path):
    _OnlineApi.created = 0
    actor = _actor(monkeypatch, _OnlineApi)
    assert _OnlineApi.created == 0
    assert actor.upload(str(tmp_path), global_step=10) == 10
    assert actor.upload(str(tmp_path), global_step=20) == 20
    assert _OnlineApi.created == 1
    assert len(actor.api.uploads) == 2


class _Handle:
    class upload:  # noqa: N801 -- mirrors `actor.upload.remote(...)`
        @staticmethod
        def remote(**kwargs):
            return "pending-upload"


class _ActorClass:
    @staticmethod
    def remote(**kwargs):
        return _Handle()


def _manager_with_a_pending_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(hf, "HFUploadActor", _ActorClass)
    (tmp_path / "global_step_10" / "actor").mkdir(parents=True)
    config = OmegaConf.create({
        "trainer": {"default_local_dir": str(tmp_path), "save_freq": 10},
        "huggingface_hub": {"hf_save_freq": 10, "repo_id": "someone/run"},
        "actor_rollout_ref": {"actor": {"checkpoint": {}}},
    })
    manager = hf.HFUploadManager(config)
    manager.maybe_upload(10)
    return manager


def _raises(exc):
    def get(_future):
        raise exc
    return get


def test_flush_survives_a_dead_upload_actor(monkeypatch, tmp_path):
    """★ What the trainer used to die on. A dead actor stays dead, so uploads stop."""
    manager = _manager_with_a_pending_upload(monkeypatch, tmp_path)
    monkeypatch.setattr(hf.ray, "get", _raises(ray.exceptions.ActorDiedError()))
    manager.flush()  # must not raise
    assert not manager.enabled
    assert not manager.should_upload(20)
    manager.flush()  # nothing pending: still fine


def test_flush_survives_a_failed_upload_and_keeps_uploading(monkeypatch, tmp_path):
    manager = _manager_with_a_pending_upload(monkeypatch, tmp_path)
    monkeypatch.setattr(hf.ray, "get", _raises(RuntimeError("503 from the Hub")))
    manager.flush()  # must not raise
    assert manager.enabled and manager.should_upload(20), "one failed upload switched them off"


def test_the_trainer_flushes_through_the_manager():
    """The guarantee lives in `flush()`, so the trainer must wait on uploads through it."""
    import inspect

    from vagen.training.trainer.mixin import VagenLogicMixin, VagenV0Mixin

    assert "self._hf_upload_manager.flush()" in inspect.getsource(VagenLogicMixin._vagen_flush_hf)
    assert "_vagen_flush_hf()" in inspect.getsource(VagenV0Mixin._fit_save_checkpoint)
