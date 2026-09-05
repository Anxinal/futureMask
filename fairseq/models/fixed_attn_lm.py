# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

DEFAULT_MAX_TARGET_POSITIONS = 1024


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


@register_model("fixed_attn_lm", dataclass=FixedAttnLanguageModelConfig)
class FixedAttnLanguageModel(TransformerLanguageModel):
    """Decoder-only transformer with Q/K/V fixed to all-ones for positional probing.

    The entire decoder is frozen; only the probe linear layer(s) are trained.
    """

    def __init__(self, decoder, probe, probe_layer_idx):
        super().__init__(decoder)
        self.decoder.eval()
        self.probe_layer_idx = probe_layer_idx
        self.probe = probe

    def forward(self, src_tokens, **kwargs):
        self.decoder.eval()
        with torch.no_grad():
            # features_only=True skips the full-vocab output projection, which we
            # never read -- only ``inner_states`` is probed.
            kwargs.setdefault("features_only", True)
            decoder_out = super().forward(src_tokens, **kwargs)

        x = decoder_out[1]["inner_states"][self.probe_layer_idx].transpose(0, 1)
        x = self.probe(x)

        return x, decoder_out[1]

    @classmethod
    def build_model(cls, args, task):
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

        # ---- Load pretrained decoder weights (before pinning Q/K/V) ----
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

        # ---- Freeze the whole decoder; only the probe is trained ----
        for p in decoder.parameters():
            p.requires_grad_(False)

        # ---- Fix Q/K/V to all-ones ----
        for layer in decoder.layers:
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(layer.self_attn, proj_name)
                proj.weight.data.fill_(1.0)
                if proj.bias is not None:
                    proj.bias.data.fill_(0.0)

        # ---- Build MLP probe ----
        # Output one class per position (0..tokens_per_sample-1).
        # Use with criterion "position_probe_ce" which targets raw indices.
        num_positions = args.tokens_per_sample

        probe = nn.Sequential(
            nn.Linear(args.decoder_output_dim, num_positions),
            nn.ReLU(),
            nn.Linear(num_positions, num_positions),
        )

        return cls(decoder, probe, int(args.probe_layer_idx))

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


@register_model_architecture("fixed_attn_lm", "fixed_attn_probe")
def fixed_attn_probe(args):
    args.decoder_layers = safe_getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 1)
    args.no_token_positional_embeddings = True
    args.dropout = safe_getattr(args, "dropout", 0.0)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.0)
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "fixed_attn_base_lm")
def fixed_attn_base_lm(args):
    """Stage-A base LM whose checkpoint ``fixed_attn_probe`` loads.

    Deliberately reuses ``fixed_attn_probe``'s config function verbatim. The probe
    loads this checkpoint with ``strict=True``, but the settings that matter most
    here do not change the key set -- ``decoder_normalize_before`` in particular
    swaps pre-norm for post-norm using the very same parameter names. A drift
    would therefore load cleanly and silently evaluate a different function than
    the one that was trained. Sharing the config function makes that impossible.

    Note this is why ``--arch transformer_lm`` must NOT be used for stage A:
    ``register_model`` auto-registers a *no-op* arch function under the bare model
    name, so ``base_lm_architecture`` never runs and ``decoder_normalize_before``
    would stay at its dataclass default of False.
    """
    fixed_attn_probe(args)
