# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import checkpoint_utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.models.transformer_lm import (
    TransformerLanguageModel,
    TransformerLanguageModelConfig,
    base_lm_architecture,
)
from fairseq.utils import safe_getattr

logger = logging.getLogger(__name__)

DEFAULT_MAX_TARGET_POSITIONS = 1024


def zero_and_freeze_qk(decoder):
    """Pin every self-attention Q/K projection to zero and freeze it.

    With ``W_q = W_k = 0`` (and zero bias) every pre-softmax attention logit is
    exactly 0, so the softmax over the unmasked keys is exactly uniform: each
    position reads the *mean* of the value vectors it can see. Under a causal mask
    that is the running prefix mean

        h_t = x_t + out_proj( 1/(t+1) * sum_{j<=t} v_j )

    whose ``1/(t+1)`` weighting is the only position-dependent quantity in the
    network -- the signal this experiment probes for. Under a bidirectional mask
    every position reads the same global mean instead, which is the control.

    Contrast with pinning Q/K to *ones*: that makes every query and key a scalar
    multiple of the ones-vector, so the logit for key j is
    ``d_head * s_q(t) * s_k(j)`` -- it varies with the key and, at d_head=512,
    reaches ~1e4, saturating the softmax into a near-argmax rather than an average.

    ``v_proj`` and ``out_proj`` are deliberately left alone so they can train.

    Returns:
        bool: True if any Q/K weight or bias was non-zero before this call. Used to
        warn when a checkpoint that was *not* trained with the pin is loaded.
    """
    had_nonzero = False
    for layer in decoder.layers:
        for proj_name in ("q_proj", "k_proj"):
            proj = getattr(layer.self_attn, proj_name)
            had_nonzero |= bool(proj.weight.data.abs().max() > 0)
            proj.weight.data.zero_()
            proj.weight.requires_grad_(False)
            if proj.bias is not None:
                had_nonzero |= bool(proj.bias.data.abs().max() > 0)
                proj.bias.data.zero_()
                proj.bias.requires_grad_(False)
    return had_nonzero


def _assert_arch_config_ran(args):
    """Guard the strict-load invariant shared by the two stages.

    ``base_lm_architecture`` forces ``decoder_normalize_before=True`` while the
    dataclass default is False, and pre-norm/post-norm use the *same* parameter
    names -- so a checkpoint built without the arch config function loads cleanly
    under ``strict=True`` and then silently evaluates a different function. Fail
    loudly instead. (``no_decoder_final_norm`` changes the key set, so that one
    already fails on its own.)
    """
    assert safe_getattr(args, "decoder_normalize_before", False), (
        "arch config function did not run base_lm_architecture -- pass "
        "--arch fixed_attn_base_lm or --arch fixed_attn_probe, not a bare model name"
    )


@dataclass
class FixedAttnLanguageModelConfig(TransformerLanguageModelConfig):
    pretrained_decoder_filename: str = field(
        default="",
        metadata={
            "help": "if not empty load the decoder weights from the given checkpoint filename"
        },
    )
    probe_layer_idx: int = field(
        default=-1,
        metadata={"help": "decoder layer index whose hidden states to probe (-1 = last)"},
    )
    non_linear_probe: bool = field(
        default=False,
        metadata={"help": "if True, use a 2-layer MLP probe with ReLU instead of linear"},
    )
    decoder_head_mask_spec: str = field(
        default="",
        metadata={
            "help": (
                "per-head decoder self-attention mask, one character per head: "
                "C=causal, B=bidirectional, F=future-only. Empty uses the default "
                "causal mask. TransformerConfig._copy_keys maps this onto "
                "cfg.decoder.head_mask_spec, which TransformerDecoder reads."
            )
        },
    )


@register_model("fixed_attn_lm", dataclass=FixedAttnLanguageModelConfig)
class FixedAttnLanguageModel(TransformerLanguageModel):
    """Stage-B probe: decoder-only transformer with Q/K pinned to zero.

    Loads a decoder trained by :class:`FixedAttnBaseLanguageModel` (which held the
    same pin throughout its own training), freezes it entirely, and trains only the
    probe layer(s) to predict absolute position from one layer's hidden states.
    See :func:`zero_and_freeze_qk` for what the pin does to attention.
    """

    def __init__(self, decoder, position_probe_layers, probe_layer_idx, non_linear_probe):
        super().__init__(decoder)
        self.decoder.eval()
        self.probe_layer_idx = probe_layer_idx
        self.non_linear_probe = non_linear_probe
        self.position_probe_layer_0 = position_probe_layers[0]
        if non_linear_probe:
            self.position_probe_layer_1 = position_probe_layers[1]
            self.relu = nn.ReLU()



    def forward(self, src_tokens, **kwargs):
        self.decoder.eval()
        with torch.no_grad():
            # features_only=True skips the full-vocab output projection, which we
            # never read -- only ``inner_states`` is probed.
            kwargs.setdefault("features_only", True)
            decoder_out = super().forward(src_tokens, **kwargs)

        x = decoder_out[1]["inner_states"][self.probe_layer_idx].transpose(0, 1)
        x = self.position_probe_layer_0(x)
        if self.non_linear_probe:
            x = self.position_probe_layer_1(self.relu(x))

        return x, decoder_out[1]

    @classmethod
    def build_model(cls, args, task):
        _assert_arch_config_ran(args)

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if safe_getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = safe_getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.decoder_input_dim
        )

        decoder = TransformerDecoder(
            args, task.target_dictionary, embed_tokens,
            no_encoder_attn=True, output_projection=None,
        )

        # ---- Load pretrained decoder weights ----
        # Must happen BEFORE the Q/K pin: load_state_dict would otherwise overwrite
        # the zeros with whatever the checkpoint holds.
        pretrained = safe_getattr(args, "pretrained_decoder_filename", "")
        if pretrained:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                pretrained, load_on_all_ranks=True
            )
            # Checkpoints saved by a FairseqLanguageModel prefix every key with
            # "decoder."; strip it so the keys line up with TransformerDecoder.
            layers_to_delete = []
            for layer_name in state["model"].copy():
                if layer_name.startswith("decoder."):
                    state["model"][layer_name[len("decoder."):]] = state["model"][
                        layer_name
                    ]
                    layers_to_delete.append(layer_name)
            for layer_name in layers_to_delete:
                del state["model"][layer_name]

            decoder.load_state_dict(state["model"], strict=True)

        # ---- Pin Q/K to zero (a no-op on a stage-A checkpoint) ----
        if zero_and_freeze_qk(decoder) and pretrained:
            logger.warning(
                "loaded checkpoint %s has non-zero Q/K weights -- it was not trained "
                "with the fixed-attention pin (--arch fixed_attn_base_lm). The probe "
                "will run with attention the base LM never saw during training.",
                pretrained,
            )

        # ---- Freeze the whole decoder; only the probe is trained ----
        for p in decoder.parameters():
            p.requires_grad_(False)

        # ---- Build probe layers ----
        num_positions = args.tokens_per_sample + decoder.dictionary.nspecial + 1

        def make_linear(in_f, out_f):
            m = nn.Linear(in_f, out_f, bias=False)
            nn.init.xavier_uniform_(m.weight)
            return m

        position_probe_layers = []
        if args.non_linear_probe:
            position_probe_layers.append(make_linear(args.decoder_output_dim, args.decoder_output_dim * 2))
            position_probe_layers.append(make_linear(args.decoder_output_dim * 2, num_positions))
        else:
            position_probe_layers.append(make_linear(args.decoder_output_dim, num_positions))

        return cls(decoder, position_probe_layers, int(args.probe_layer_idx), args.non_linear_probe)

    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


@dataclass
class FixedAttnBaseLanguageModelConfig(TransformerLanguageModelConfig):
    """Stage-A config. Same knobs as ``transformer_lm`` -- the Q/K pin is
    unconditional, so there is nothing extra to configure."""


@register_model("fixed_attn_base", dataclass=FixedAttnBaseLanguageModelConfig)
class FixedAttnBaseLanguageModel(TransformerLanguageModel):
    """Stage-A base LM: a plain NoPos LM whose Q/K are pinned to zero *before*
    training and held fixed throughout it.

    Everything else -- V, out_proj, the FFN, the embeddings -- trains normally, so
    the network co-adapts to fixed uniform attention rather than being trained with
    content-based attention that is then thrown away at probe time.
    """

    @classmethod
    def build_model(cls, args, task):
        _assert_arch_config_ran(args)
        model = super().build_model(args, task)
        zero_and_freeze_qk(model.decoder)
        return model


@register_model_architecture("fixed_attn_lm", "fixed_attn_probe")
def fixed_attn_probe(args):
    args.decoder_layers = safe_getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 1)
    args.no_token_positional_embeddings = True
    args.dropout = safe_getattr(args, "dropout", 0.0)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.0)
    base_lm_architecture(args)


@register_model_architecture("fixed_attn_base", "fixed_attn_base_lm")
def fixed_attn_base_lm(args):
    """Arch for the stage-A base LM -- reuses ``fixed_attn_probe``'s config verbatim.

    The probe loads this stage's checkpoint with ``strict=True``, so sharing the
    config function is what keeps the two decoders from drifting; see
    :func:`_assert_arch_config_ran` for the divergence that would otherwise be
    silent.

    Note the deliberate model/arch name split: ``register_model`` auto-registers a
    *no-op* arch config function under the bare model name, so the model is called
    ``fixed_attn_base`` and this -- the arch you actually want -- keeps the name
    ``fixed_attn_base_lm``.
    """
    fixed_attn_probe(args)
