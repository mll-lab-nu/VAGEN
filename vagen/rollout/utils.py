
"""Utilities for rollout helpers.

This module provides helpers that are **side-effect free** and can be safely
imported from any worker / driver process.  The main feature required now is a
function that converts a `torch.Tensor` to a deterministic, hashable string so
that identical tensor contents yield identical IDs (e.g. for grouping or
caching).

All public APIs intentionally avoid any heavy dependencies beyond the Python
standard library and `torch`.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Optional

import torch
# Note: avoid circular import; tensor_to_uuid is defined in this module.

__all__ = [
    "tensor_to_hash",
    "tensor_to_uuid",
]


def _filter_pad(tensor: torch.Tensor, pad_token_id: Optional[int] = None) -> torch.Tensor:
    """Remove *all* occurrences of ``pad_token_id`` from *tensor*.

    Parameters
    ----------
    tensor:
        The input tensor (any shape, any dtype). It will be detached, moved to
        CPU and flattened.
    pad_token_id:
        If *None* the tensor will be returned as-is (after detach/flatten).  If
        an ``int`` is given, every element equal to that value will be dropped
        before hashing.
    """
    data = tensor.detach().cpu().flatten()
    if pad_token_id is not None:
        data = data[data != pad_token_id]
    return data


def tensor_to_hash(
    tensor: torch.Tensor,
    *,
    pad_token_id: Optional[int] = None,
    algo: str = "md5",
) -> str:
    """Return **hex digest** of *tensor* content.

    This is deterministic across machines/versions as long as the underlying
    byte representation of the tensor (dtype + little-endian) stays the same.

    Parameters
    ----------
    tensor:
        The tensor to hash.  It can be on any device / with *requires_grad*;
        the function will make a detached CPU copy under the hood.
    pad_token_id:
        If provided, all elements equal to this value will be removed prior to
        hashing so that padding does not affect the final digest.
    algo:
        Any algorithm name accepted by :pymod:`hashlib` ("md5", "sha1",
        "sha256" …).  *md5* is sufficient for UUID generation and faster than
        longer digests.

    Returns
    -------
    str
        Hexadecimal digest string (lowercase).
    """
    data = _filter_pad(tensor, pad_token_id)
    byte_data = data.contiguous().numpy().tobytes() if data.numel() > 0 else b""

    h = hashlib.new(algo)
    h.update(byte_data)
    return h.hexdigest()


def tensor_to_uuid(
    tensor: torch.Tensor,
    *,
    pad_token_id: Optional[int] = None,
    algo: str = "md5",
    namespace: uuid.UUID = uuid.NAMESPACE_OID,
) -> str:
    """Return a **stable UUIDv5 string** derived from tensor content.

    Internally the function first computes ``tensor_to_hash`` with *algo*
    (default *md5*, producing 32 hex chars) and then feeds that hex string into
    :pyfunc:`uuid.uuid5` using the provided *namespace*.

    The resulting UUID is *deterministic*: identical tensors yield identical
    UUIDs; different tensors (or different padding-filtered versions) yield
    different UUIDs.
    """
    digest = tensor_to_hash(tensor, pad_token_id=pad_token_id, algo=algo)
    return str(uuid.uuid5(namespace, digest))
