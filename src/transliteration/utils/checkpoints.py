"""Robust checkpoint loading for raw and Lightning checkpoints.

Baseline ``baseline_best.ckpt`` is a PyTorch Lightning checkpoint::

    {"epoch": ..., "global_step": ..., "pytorch-lightning_version": ...,
     "state_dict": {"model.encoder...": ..., ...}, "loops": ...}

while ``attention_best.pt`` / ``transformer_best.pt`` are raw state dicts
(``OrderedDict`` of tensors).  This module normalizes both forms so
``app.py`` and ``scripts/eval.py`` can load any of them.

Safety note: ``torch.load`` without ``weights_only=True`` can execute
arbitrary pickle payloads.  We first try ``weights_only=True`` (safe).
Lightning envelopes contain non-tensor objects (``loops``, version
strings), so that attempt fails and we fall back to
``weights_only=False``.  Only use the fallback for locally trained,
trusted checkpoints — never for untrusted downloads.
"""

from pathlib import Path
from typing import Any, Union

import torch


def safe_torch_load(
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> Any:
    """Load a checkpoint file as safely as the format allows.

    Tries ``weights_only=True`` first; falls back to ``weights_only=False``
    for Lightning envelopes that contain non-tensor metadata.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Very old torch without the weights_only kwarg.
        return torch.load(path, map_location=map_location)
    except Exception:
        # Lightning checkpoints (epoch/loops/version) are not
        # weights-only loadable; trusted-local fallback.
        return torch.load(path, map_location=map_location, weights_only=False)


def is_lightning_checkpoint(obj: Any) -> bool:
    """Return True if obj looks like a Lightning checkpoint envelope."""
    return (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    )


def normalize_state_dict(obj: Any) -> dict:
    """Extract a loadable state_dict from a raw or Lightning checkpoint.

    - Lightning envelope (``{"state_dict": {...}}``) -> inner dict.
    - ``model.`` prefix (Lightning ``LightningModule`` wrapping
      ``self.model = Seq2Seq(...)``) is stripped per-key.
    """
    if is_lightning_checkpoint(obj):
        state = obj["state_dict"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise ValueError(
            f"Unsupported checkpoint format: {type(obj).__name__}. "
            "Expected a state_dict or Lightning {'state_dict': ...} envelope."
        )
    normalized: dict = {}
    for key, value in state.items():
        new_key = key[len("model.") :] if key.startswith("model.") else key
        normalized[new_key] = value
    return normalized


def load_model_state_dict(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> dict:
    """Load a raw or Lightning checkpoint into ``model``.

    Returns the normalized state_dict that was loaded.
    """
    if isinstance(map_location, str):
        map_location = torch.device(map_location)
    raw = safe_torch_load(checkpoint_path, map_location=map_location)
    state = normalize_state_dict(raw)
    model.load_state_dict(state, strict=strict)
    return state
