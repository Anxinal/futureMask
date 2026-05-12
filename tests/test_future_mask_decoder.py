import unittest

import torch

from fairseq.models.transformer.transformer_decoder import TransformerDecoderBase


class TestFutureMaskDecoder(unittest.TestCase):
    def test_future_only_mask_allow_self(self):
        decoder = TransformerDecoderBase.__new__(TransformerDecoderBase)
        decoder._future_only_mask = torch.empty(0)

        x = torch.zeros(4, 1, 8)
        mask = TransformerDecoderBase.buffered_future_only_mask(
            decoder, x, allow_self=True
        )

        self.assertEqual(mask.shape, torch.Size([4, 4]))
        self.assertTrue(torch.isneginf(mask[2, 1]))
        self.assertEqual(mask[2, 2].item(), 0.0)
        self.assertEqual(mask[2, 3].item(), 0.0)

    def test_future_only_mask_disallow_self(self):
        decoder = TransformerDecoderBase.__new__(TransformerDecoderBase)
        decoder._future_only_mask = torch.empty(0)

        x = torch.zeros(4, 1, 8)
        mask = TransformerDecoderBase.buffered_future_only_mask(
            decoder, x, allow_self=False
        )

        self.assertTrue(torch.isneginf(mask[2, 2]))
        self.assertEqual(mask[2, 3].item(), 0.0)

    def test_encdec_decoder_forces_causal_mask(self):
        """In an encoder-decoder model the decoder must use vanilla causal masks
        regardless of --decoder-future-mask-layers."""
        decoder = TransformerDecoderBase.__new__(TransformerDecoderBase)
        decoder._future_mask = torch.empty(0)
        decoder._future_only_mask = torch.empty(0)
        decoder.no_encoder_attn = False  # encoder-decoder
        decoder.future_mask_decoder_layers = {0, 1, 2, 3}  # would normally trigger future-only
        decoder.future_mask_allow_self = True

        x = torch.zeros(4, 1, 8)
        # Layer 0 is in future_mask_decoder_layers, but encoder-decoder gate forces causal.
        mask = TransformerDecoderBase._select_decoder_self_attn_mask(decoder, 0, x)
        # Causal: upper-triangle (j > i) is -inf, lower-triangle (including diag) is 0.
        self.assertTrue(torch.isneginf(mask[0, 1]))
        self.assertEqual(mask[1, 0].item(), 0.0)
        self.assertEqual(mask[2, 2].item(), 0.0)

    def test_decoder_only_lm_keeps_future_mask(self):
        """Decoder-only LM (no_encoder_attn=True) must still honour the per-layer
        future-mask spec."""
        decoder = TransformerDecoderBase.__new__(TransformerDecoderBase)
        decoder._future_mask = torch.empty(0)
        decoder._future_only_mask = torch.empty(0)
        decoder.no_encoder_attn = True  # decoder-only LM
        decoder.future_mask_decoder_layers = {0}
        decoder.future_mask_allow_self = True

        x = torch.zeros(4, 1, 8)
        # Layer 0 in spec → future-only mask. Layer 1 not in spec → causal.
        m0 = TransformerDecoderBase._select_decoder_self_attn_mask(decoder, 0, x)
        m1 = TransformerDecoderBase._select_decoder_self_attn_mask(decoder, 1, x)
        # Future-only allow_self: row 2 col 1 (past) is -inf, col 2 (self) is 0.
        self.assertTrue(torch.isneginf(m0[2, 1]))
        self.assertEqual(m0[2, 2].item(), 0.0)
        # Causal: row 2 col 3 (future) is -inf.
        self.assertTrue(torch.isneginf(m1[2, 3]))


if __name__ == "__main__":
    unittest.main()
