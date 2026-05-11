"""Per-head attention-mask unit tests for the future-mask experiment.

Verifies that:
  1. MultiheadAttention accepts a [H, T, T] mask and produces different
     attention patterns per head.
  2. build_per_head_encoder_mask emits the expected mask shape & values
     for a mixed spec.
"""

import math
import unittest

import torch

from fairseq.models.transformer.encoder_head_mask import (
    build_per_head_encoder_mask,
    parse_head_mask_spec,
)
from fairseq.modules import MultiheadAttention


class TestPerHeadEncoderMask(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_head_mask_spec("", 4), tuple())
        self.assertEqual(
            parse_head_mask_spec("C,F,B,C", 4), ("C", "F", "B", "C")
        )
        with self.assertRaises(ValueError):
            parse_head_mask_spec("C,F", 4)
        with self.assertRaises(ValueError):
            parse_head_mask_spec("C,X,B,F", 4)

    def test_mask_values(self):
        spec = parse_head_mask_spec("C,F,B,C", 4)
        mask = build_per_head_encoder_mask(
            spec, T=5, device=torch.device("cpu"), dtype=torch.float32, allow_self=True
        )
        self.assertEqual(mask.shape, (4, 5, 5))
        # head 0 (causal): mask[i, j] is -inf when j > i
        c = mask[0]
        for i in range(5):
            for j in range(5):
                if j > i:
                    self.assertTrue(torch.isinf(c[i, j]) and c[i, j] < 0)
                else:
                    self.assertEqual(c[i, j].item(), 0.0)
        # head 1 (future, allow_self): mask[i, j] is -inf when j < i (diagonal OK)
        f = mask[1]
        for i in range(5):
            for j in range(5):
                if j < i:
                    self.assertTrue(torch.isinf(f[i, j]) and f[i, j] < 0)
                else:
                    self.assertEqual(f[i, j].item(), 0.0)
        # head 2 (bidirectional): all zeros
        self.assertTrue(torch.all(mask[2] == 0).item())


class TestMultiheadPerHeadMask(unittest.TestCase):
    def test_per_head_attention_patterns(self):
        torch.manual_seed(0)
        embed_dim = 16
        num_heads = 4
        T = 6
        B = 2
        attn = MultiheadAttention(embed_dim, num_heads, self_attention=True)
        attn.eval()

        # spec: head 0 causal, head 1 future-only, heads 2/3 bidirectional
        spec = parse_head_mask_spec("C,F,B,B", num_heads)
        mask = build_per_head_encoder_mask(
            spec, T=T, device=torch.device("cpu"), dtype=torch.float32, allow_self=True
        )

        x = torch.randn(T, B, embed_dim)
        # We need raw attention weights — call with need_weights=True and
        # need_head_weights=True to get per-head attention.
        _out, attn_weights = attn(
            query=x, key=x, value=x, attn_mask=mask,
            need_weights=True, need_head_weights=True,
        )
        # attn_weights: [num_heads, B, T, T]
        self.assertEqual(attn_weights.shape, (num_heads, B, T, T))
        # head 0 causal: probs on strict future positions must be 0
        for i in range(T):
            for j in range(i + 1, T):
                self.assertTrue(
                    torch.all(attn_weights[0, :, i, j] < 1e-6),
                    f"head 0 (causal) leaked future at i={i}, j={j}",
                )
        # head 1 future: probs on strict past positions must be 0 (diagonal allowed)
        for i in range(T):
            for j in range(0, i):
                self.assertTrue(
                    torch.all(attn_weights[1, :, i, j] < 1e-6),
                    f"head 1 (future) leaked past at i={i}, j={j}",
                )
        # heads 2/3 bidirectional: should generally be nonzero everywhere
        self.assertTrue(torch.all(attn_weights[2] > 0))
        self.assertTrue(torch.all(attn_weights[3] > 0))


if __name__ == "__main__":
    unittest.main()
