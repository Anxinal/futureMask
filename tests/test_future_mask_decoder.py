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


if __name__ == "__main__":
    unittest.main()
