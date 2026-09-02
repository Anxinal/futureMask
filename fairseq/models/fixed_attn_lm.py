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
    probe_layer_idx: int = field(
        default=-1,
        metadata={"help": "decoder layer index whose hidden states to probe (-1 = last)"},
    )
    non_linear_probe: bool = field(
        default=False,
        metadata={"help": "if True, use a 2-layer MLP probe with ReLU instead of linear"},
    )


@register_model("fixed_attn_lm", dataclass=FixedAttnLanguageModelConfig)
class FixedAttnLanguageModel(TransformerLanguageModel):
    """Decoder-only transformer with Q/K/V fixed to all-ones for positional probing.

    The entire decoder is frozen; only the probe linear layer(s) are trained.
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
            decoder_out = super().forward(src_tokens, **kwargs)

        x = decoder_out[1]["inner_states"][self.probe_layer_idx].transpose(0, 1)
        x = self.position_probe_layer_0(x)
        if self.non_linear_probe:
            x = self.position_probe_layer_1(self.relu(x))

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

        # ---- Fix Q/K/V to all-ones and freeze ----
        for layer in decoder.layers:
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(layer.self_attn, proj_name)
                proj.weight.data.fill_(1.0)
                if proj.bias is not None:
                    proj.bias.data.fill_(0.0)
                proj.weight.requires_grad = False
                if proj.bias is not None:
                    proj.bias.requires_grad = False

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


@register_model_architecture("fixed_attn_lm", "fixed_attn_probe")
def fixed_attn_probe(args):
    args.decoder_layers = safe_getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 1)
    args.no_token_positional_embeddings = True
    args.dropout = safe_getattr(args, "dropout", 0.0)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.0)
    base_lm_architecture(args)
