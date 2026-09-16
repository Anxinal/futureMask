# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Single-layer models for the self-attention recovery experiment.

The question: can attention that only ever sees one side of each position reproduce
what unmasked attention computes?

Stage A trains a *teacher*: one layer, one unmasked head, MLM objective. Stage B
freezes that teacher and trains a *student* -- same layer shape, but its heads carry
per-head masks -- to regress the teacher's **attention sublayer output** (before the
residual add, FFN and layernorm; that is the quantity the recovery claim is about).

Conditions, set with ``--head-mask-spec``:
  C,F  experimental arm -- one causal head, one future-only head
  C,C  control          -- no access to the future; should not close the gap
  B,B  ceiling          -- unmasked student; bounds the error not caused by masking

Heads here keep the **full** model width. fairseq's ``MultiheadAttention`` splits
``embed_dim`` across heads (d/H each); this module gives every head its own d->d
Q/K/V, so the causal and future-only heads each produce a full d-dimensional average
of value vectors, and ``out_proj`` (H*d -> d, no bias) is the fixed linear map that
has to combine them. That map is the object under test.

Both stages run under the stock ``masked_lm`` task: stage A with the stock
``masked_lm`` criterion, stage B with ``attn_recovery``.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn

from fairseq import checkpoint_utils, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.masks import BidirectionalMask, CausalMask, FutureOnlyMask
from fairseq.models.roberta.model import RobertaLMHead
from fairseq.models.transformer import Embedding
from fairseq.modules import LayerNorm, PositionalEmbedding
from fairseq.utils import safe_getattr

logger = logging.getLogger(__name__)

_VALID_TOKENS = ("C", "F", "B")

DEFAULT_MAX_SOURCE_POSITIONS = 1024


def parse_mask_spec(spec: str) -> Tuple[str, ...]:
    """Parse a per-head mask spec into a tuple of tokens, one per head.

    Accepts either comma-separated (``"C,F"``, as the encoder specs elsewhere in the
    repo are written) or bare (``"CF"``, as ``transformer_lm_position_probe``'s help
    text writes them).
    """
    if not spec:
        raise ValueError("head mask spec must name at least one head")
    tokens = (
        tuple(tok.strip().upper() for tok in spec.split(","))
        if "," in spec
        else tuple(ch.upper() for ch in spec.strip())
    )
    for tok in tokens:
        if tok not in _VALID_TOKENS:
            raise ValueError(
                f"invalid head-mask token {tok!r}; expected one of {_VALID_TOKENS}"
            )
    return tokens


@lru_cache(maxsize=32)
def _build_head_masks(spec: Tuple[str, ...], T: int, allow_self: bool) -> torch.Tensor:
    """Build a ``[H, T, T]`` additive mask on CPU (0 = attend, -inf = block).

    Cached by (spec, T, allow_self) exactly as
    ``fairseq/models/transformer/encoder_head_mask.py`` does; callers move and cast.
    """
    heads = []
    for tok in spec:
        if tok == "C":
            heads.append(CausalMask(T).tensor)
        elif tok == "F":
            heads.append(FutureOnlyMask(T, allow_self=allow_self).tensor)
        else:
            heads.append(BidirectionalMask(T).tensor)
    return torch.stack(heads, dim=0)


class FullWidthMaskedAttention(nn.Module):
    """Self-attention where every head keeps the full model width.

    Returns ``[B, T, embed_dim]`` -- the attention sublayer output, which is what the
    student is trained to match.
    """

    def __init__(
        self,
        embed_dim: int,
        mask_spec: str,
        dropout: float = 0.0,
        future_mask_allow_self: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_spec = parse_mask_spec(mask_spec)
        self.num_heads = len(self.mask_spec)
        self.future_mask_allow_self = future_mask_allow_self
        self.scaling = embed_dim**-0.5

        def linear(in_f, out_f, bias=True):
            m = nn.Linear(in_f, out_f, bias=bias)
            nn.init.xavier_uniform_(m.weight)
            if bias:
                nn.init.constant_(m.bias, 0.0)
            return m

        self.q_proj = nn.ModuleList(
            linear(embed_dim, embed_dim) for _ in range(self.num_heads)
        )
        self.k_proj = nn.ModuleList(
            linear(embed_dim, embed_dim) for _ in range(self.num_heads)
        )
        self.v_proj = nn.ModuleList(
            linear(embed_dim, embed_dim) for _ in range(self.num_heads)
        )
        # The recovery map. bias=False keeps it a pure d x d (here H*d x d) linear
        # map, as the experiment plan specifies.
        self.out_proj = linear(self.num_heads * embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None):
        """x: ``[B, T, C]``; padding_mask: ``[B, T]``, True at padded positions."""
        B, T, C = x.shape
        head_masks = _build_head_masks(
            self.mask_spec, T, self.future_mask_allow_self
        ).to(device=x.device, dtype=x.dtype)

        outs = []
        for h in range(self.num_heads):
            q = self.q_proj[h](x) * self.scaling
            k = self.k_proj[h](x)
            v = self.v_proj[h](x)

            logits = torch.bmm(q, k.transpose(1, 2))  # [B, T, T]
            logits = logits + head_masks[h].unsqueeze(0)
            if padding_mask is not None:
                logits = logits.masked_fill(padding_mask.unsqueeze(1), float("-inf"))

            # softmax in fp32 (utils.softmax), then back to the input dtype for fp16.
            attn = utils.softmax(logits, dim=-1).type_as(q)
            # A row with every key masked softmaxes to NaN. Only padded query rows can
            # hit this, and the criterion drops them, but NaNs would still poison the
            # fp16 grad scaler, so zero them here.
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.dropout(attn)
            outs.append(torch.bmm(attn, v))

        return self.out_proj(torch.cat(outs, dim=-1))


class SimpleAttnEncoder(FairseqEncoder):
    """Embeddings -> one full-width masked-attention sublayer -> (optional) FFN block.

    ``ffn_embed_dim=None`` builds attention only, which is all the student needs: it
    is trained on the attention output, so an FFN would sit in the graph collecting
    no gradient.
    """

    def __init__(
        self,
        dictionary,
        embed_tokens,
        mask_spec: str,
        dropout: float = 0.1,
        ffn_embed_dim: Optional[int] = None,
        activation_fn: str = "gelu",
        use_sinusoidal_pos: bool = False,
        max_source_positions: int = DEFAULT_MAX_SOURCE_POSITIONS,
        future_mask_allow_self: bool = True,
    ):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        self.embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = max_source_positions

        self.embed_positions = (
            PositionalEmbedding(
                max_source_positions,
                self.embed_dim,
                self.padding_idx,
                learned=False,
            )
            if use_sinusoidal_pos
            else None
        )

        self.dropout = nn.Dropout(dropout)
        self.self_attn = FullWidthMaskedAttention(
            self.embed_dim,
            mask_spec,
            dropout=dropout,
            future_mask_allow_self=future_mask_allow_self,
        )

        if ffn_embed_dim is not None:
            self.attn_layer_norm = LayerNorm(self.embed_dim)
            self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
            self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
            self.final_layer_norm = LayerNorm(self.embed_dim)
            self.activation_fn = utils.get_activation_fn(activation_fn)
        else:
            self.attn_layer_norm = None

    def forward(self, src_tokens, src_lengths=None, **unused):
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        x = self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)
        x = self.dropout(x)

        attn_out = self.self_attn(x, padding_mask=padding_mask)

        out = {
            "attn_out": attn_out,  # [B, T, C] -- the regression target
            "padding_mask": padding_mask,
        }

        if self.attn_layer_norm is not None:
            h = self.attn_layer_norm(x + self.dropout(attn_out))
            f = self.fc2(self.dropout(self.activation_fn(self.fc1(h))))
            out["features"] = self.final_layer_norm(h + self.dropout(f))

        return out

    def max_positions(self):
        return self.max_source_positions


@dataclass
class SimpleAttnTeacherConfig(FairseqDataclass):
    encoder_embed_dim: int = field(
        default=1024, metadata={"help": "model width d"}
    )
    encoder_ffn_embed_dim: int = field(
        default=4096, metadata={"help": "FFN inner dimension"}
    )
    head_mask_spec: str = field(
        default="B",
        metadata={
            "help": "per-head self-attention mask, one token per head: C=causal, "
            "F=future-only, B=bidirectional. e.g. 'C,F'. The teacher is 'B'."
        },
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    activation_fn: str = field(default="gelu", metadata={"help": "FFN activation"})
    use_sinusoidal_pos: bool = field(
        default=False,
        metadata={
            "help": "add sinusoidal positional embeddings. Off by default: this repo's "
            "experiments are NoPos, and the pseudocode has no positions."
        },
    )
    future_mask_allow_self: bool = field(
        default=True,
        metadata={"help": "F heads keep the diagonal, so no row is fully masked"},
    )
    max_source_positions: int = field(
        default=DEFAULT_MAX_SOURCE_POSITIONS,
        metadata={"help": "maximum input length"},
    )


@register_model("simple_attn_teacher", dataclass=SimpleAttnTeacherConfig)
class SimpleAttnTeacherModel(FairseqEncoderModel):
    """Stage A: one unmasked full-width head, trained with the MLM objective.

    ``forward`` matches what ``fairseq/criterions/masked_lm.py`` calls -- it passes
    ``masked_tokens`` and reads element 0 -- so the stock criterion works unmodified.
    """

    def __init__(self, encoder, lm_head):
        super().__init__(encoder)
        self.lm_head = lm_head

    @classmethod
    def build_model(cls, args, task):
        embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
        embed_tokens = Embedding(
            len(task.source_dictionary), embed_dim, task.source_dictionary.pad()
        )
        encoder = SimpleAttnEncoder(
            task.source_dictionary,
            embed_tokens,
            mask_spec=safe_getattr(args, "head_mask_spec", "B"),
            dropout=safe_getattr(args, "dropout", 0.1),
            ffn_embed_dim=safe_getattr(args, "encoder_ffn_embed_dim", 4096),
            activation_fn=safe_getattr(args, "activation_fn", "gelu"),
            use_sinusoidal_pos=safe_getattr(args, "use_sinusoidal_pos", False),
            max_source_positions=safe_getattr(
                args, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS
            ),
            future_mask_allow_self=safe_getattr(args, "future_mask_allow_self", True),
        )
        lm_head = RobertaLMHead(
            embed_dim=embed_dim,
            output_dim=len(task.source_dictionary),
            activation_fn=safe_getattr(args, "activation_fn", "gelu"),
            weight=embed_tokens.weight,  # tied
        )
        return cls(encoder, lm_head)

    def forward(self, src_tokens, src_lengths=None, masked_tokens=None, **unused):
        encoder_out = self.encoder(src_tokens, src_lengths)
        logits = self.lm_head(encoder_out["features"], masked_tokens=masked_tokens)
        return logits, encoder_out


@dataclass
class SimpleAttnStudentConfig(FairseqDataclass):
    teacher_checkpoint: str = field(
        default="",
        metadata={"help": "checkpoint of the trained simple_attn_teacher to match"},
    )
    head_mask_spec: str = field(
        default="C,F",
        metadata={
            "help": "per-head student mask: 'C,F' (experiment), 'C,C' (control), "
            "'B,B' (ceiling)"
        },
    )
    student_embed: str = field(
        default="random",
        metadata={
            "help": "student embedding table. 'random' (default): its own table, fresh "
            "init, retrained from scratch -- the student inherits nothing from the "
            "teacher. 'copy': its own trainable table, initialised from the teacher's. "
            "'shared': the teacher's tensor itself, which is therefore frozen (training "
            "it would mutate the teacher); that isolates attention as the only "
            "difference but leaves one student parameter untrained."
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "student dropout; 0 keeps the regression target clean"},
    )
    future_mask_allow_self: bool = field(default=True, metadata={"help": "see teacher"})


@register_model("simple_attn_student", dataclass=SimpleAttnStudentConfig)
class SimpleAttnStudentModel(FairseqEncoderModel):
    """Stage B: masked student trained to reproduce the frozen teacher's attention.

    The student is retrained from scratch and inherits nothing from the teacher: every
    parameter it owns is learned, its embedding table included (see ``--student-embed``).
    Only the teacher is frozen, since it defines the target.

    Holds the frozen teacher as a submodule, mirroring the frozen-decoder pattern in
    ``fairseq/models/fixed_attn_lm.py``. fairseq's trainer filters the optimizer by
    ``requires_grad``, so the teacher's parameters never enter it.

    Two consequences of carrying the teacher inside the model:
      * Under DDP (more than one GPU) its parameters are unused in the backward pass,
        so a multi-GPU run needs ``--find-unused-parameters``. The experiment script
        requests a single GPU, where this does not arise.
      * Under ``--fp16`` the frozen teacher stays in fp16 rather than getting an fp32
        master copy, so its outputs differ slightly from its own fp32 training. That
        shift is identical across conditions, which are only compared to each other.
    """

    def __init__(self, encoder, teacher):
        super().__init__(encoder)
        self.teacher = teacher
        self.teacher.eval()

    def train(self, mode=True):
        """Keep the teacher in eval mode.

        ``Trainer`` calls ``model.train()`` before every step, which would otherwise
        switch the teacher's dropout back on and make the regression target noisy.
        """
        super().train(mode)
        self.teacher.eval()
        return self

    @classmethod
    def build_model(cls, args, task):
        ckpt_path = safe_getattr(args, "teacher_checkpoint", "")
        assert ckpt_path, "--teacher-checkpoint is required for simple_attn_student"

        state = checkpoint_utils.load_checkpoint_to_cpu(
            ckpt_path, load_on_all_ranks=True
        )
        assert "cfg" in state, f"{ckpt_path} has no cfg; cannot rebuild the teacher"
        teacher = task.build_model(state["cfg"]["model"])
        teacher.load_state_dict(state["model"], strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        assert isinstance(teacher, SimpleAttnTeacherModel), (
            f"--teacher-checkpoint must hold a simple_attn_teacher, got "
            f"{type(teacher).__name__}"
        )

        # Take the width from the teacher: the two attention outputs are compared
        # elementwise, so a mismatch is not recoverable.
        embed_dim = teacher.encoder.embed_dim

        embed_mode = safe_getattr(args, "student_embed", "random")
        assert embed_mode in ("copy", "random", "shared"), (
            f"--student-embed must be one of copy/random/shared, got {embed_mode!r}"
        )
        if embed_mode == "shared":
            # The teacher's own tensor: frozen along with it.
            embed_tokens = teacher.encoder.embed_tokens
        else:
            embed_tokens = Embedding(
                len(task.source_dictionary), embed_dim, task.source_dictionary.pad()
            )
            if embed_mode == "copy":
                with torch.no_grad():
                    embed_tokens.weight.copy_(teacher.encoder.embed_tokens.weight)
            embed_tokens.weight.requires_grad_(True)

        encoder = SimpleAttnEncoder(
            task.source_dictionary,
            embed_tokens,
            mask_spec=safe_getattr(args, "head_mask_spec", "C,F"),
            dropout=safe_getattr(args, "dropout", 0.0),
            ffn_embed_dim=None,  # student is trained on the attention output only
            use_sinusoidal_pos=teacher.encoder.embed_positions is not None,
            max_source_positions=teacher.encoder.max_source_positions,
            future_mask_allow_self=safe_getattr(args, "future_mask_allow_self", True),
        )

        trainable = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )
        logger.info(
            "student spec=%s vs teacher spec=%s, d=%d, embed=%s, trainable=%.1fM",
            encoder.self_attn.mask_spec,
            teacher.encoder.self_attn.mask_spec,
            embed_dim,
            embed_mode,
            trainable / 1e6,
        )
        return cls(encoder, teacher)

    def forward(self, src_tokens, src_lengths=None, **unused):
        student_out = self.encoder(src_tokens, src_lengths)
        with torch.no_grad():
            teacher_out = self.teacher.encoder(src_tokens, src_lengths)
        return student_out["attn_out"], {
            "teacher_attn_out": teacher_out["attn_out"],
            "padding_mask": student_out["padding_mask"],
        }


@register_model_architecture("simple_attn_teacher", "simple_attn_teacher_base")
def simple_attn_teacher_base(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 4096)
    args.head_mask_spec = safe_getattr(args, "head_mask_spec", "B")
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.use_sinusoidal_pos = safe_getattr(args, "use_sinusoidal_pos", False)
    args.future_mask_allow_self = safe_getattr(args, "future_mask_allow_self", True)
    args.max_source_positions = safe_getattr(
        args, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS
    )


@register_model_architecture("simple_attn_student", "simple_attn_student_base")
def simple_attn_student_base(args):
    args.head_mask_spec = safe_getattr(args, "head_mask_spec", "C,F")
    args.student_embed = safe_getattr(args, "student_embed", "random")
    args.dropout = safe_getattr(args, "dropout", 0.0)
    args.future_mask_allow_self = safe_getattr(args, "future_mask_allow_self", True)
