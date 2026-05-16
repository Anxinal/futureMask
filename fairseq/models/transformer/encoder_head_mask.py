"""Per-head encoder self-attention masks for the future-mask experiment.

Supports per-head mixtures of:
  C - causal mask          (attend to j <= i)
  F - future-only mask     (attend to j >  i; diagonal optional via allow_self)
  B - bidirectional / none (attend to everything)

Spec strings look like ``"F,F,C,C,B,B,B,B"`` (one token per attention head).
"""

from functools import lru_cache
from typing import Tuple

import torch

from fairseq import utils


_VALID_TOKENS = ("C", "F", "B")


def parse_head_mask_spec(spec: str, num_heads: int) -> Tuple[str, ...]:
    if spec is None or spec == "":
        return tuple()
    tokens = tuple(tok.strip().upper() for tok in spec.split(","))
    if len(tokens) != num_heads:
        raise ValueError(
            f"encoder_head_mask_spec has {len(tokens)} tokens but num_heads={num_heads}"
        )
    for tok in tokens:
        if tok not in _VALID_TOKENS:
            raise ValueError(
                f"invalid head-mask token {tok!r}; expected one of {_VALID_TOKENS}"
            )
    return tokens


def _causal_mask(T: int) -> torch.Tensor:
    return torch.triu(utils.fill_with_neg_inf(torch.zeros([T, T])), 1)


def _future_mask(T: int, allow_self: bool) -> torch.Tensor:
    diagonal = -1 if allow_self else 0
    return torch.tril(utils.fill_with_neg_inf(torch.zeros([T, T])), diagonal)


def _bidir_mask(T: int) -> torch.Tensor:
    return torch.zeros([T, T])


@lru_cache(maxsize=32)
def _build_cached(
    spec: Tuple[str, ...],
    T: int,
    allow_self: bool,
) -> torch.Tensor:
    """Build a [H, T, T] mask on CPU; callers move/cast as needed."""
    heads = []
    for tok in spec:
        if tok == "C":
            heads.append(_causal_mask(T))
        elif tok == "F":
            heads.append(_future_mask(T, allow_self))
        elif tok == "B":
            heads.append(_bidir_mask(T))
        else:  # pragma: no cover - validated upstream
            raise ValueError(tok)
    return torch.stack(heads, dim=0)


def build_per_head_encoder_mask(
    spec: Tuple[str, ...],
    T: int,
    device: torch.device,
    dtype: torch.dtype,
    allow_self: bool = True,
) -> torch.Tensor:
    """Return a ``[num_heads, T, T]`` attention-bias mask (-inf at masked entries).

    The mask is cached on CPU keyed by (spec, T, allow_self) and moved/cast
    to (device, dtype) on each call.
    """
    mask = _build_cached(spec, T, allow_self)
    return mask.to(device=device, dtype=dtype)
