"""Wraps a TokenBlockDataset into encoder-decoder (prefix -> suffix) samples.

Each block of length T is split at position k = round(prefix_fraction * T).
Encoder source = block[:k], decoder target = block[k:].
``prev_output_tokens`` is built by `collate` via ``move_eos_to_beginning``.
"""

import numpy as np
import torch

from fairseq.data import FairseqDataset
from fairseq.data.language_pair_dataset import collate


class PrefixSuffixDataset(FairseqDataset):
    def __init__(
        self,
        block_dataset,
        vocab,
        prefix_fraction: float = 0.5,
        left_pad_source: bool = True,
        left_pad_target: bool = False,
    ):
        super().__init__()
        assert 0.0 < prefix_fraction < 1.0
        self.dataset = block_dataset
        self.vocab = vocab
        self.prefix_fraction = prefix_fraction
        self.pad_idx = vocab.pad()
        self.eos_idx = vocab.eos()
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        # Sizes used for batching: total length of the (src, tgt) pair = full block.
        sizes = np.asarray(block_dataset.sizes, dtype=np.int64)
        self._sizes = sizes
        self._src_sizes = np.maximum(1, (sizes * prefix_fraction).astype(np.int64))
        self._tgt_sizes = np.maximum(1, sizes - self._src_sizes)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        # TokenBlockDataset with include_targets=True returns (source, future, past).
        if isinstance(item, tuple):
            tokens = item[0]
        else:
            tokens = item
        T = tokens.size(0)
        k = max(1, min(T - 1, int(round(self.prefix_fraction * T))))
        source = tokens[:k].long()
        target = tokens[k:].long()
        # Ensure target ends with eos so move_eos_to_beginning produces a proper prev_output.
        if target.numel() == 0 or target[-1].item() != self.eos_idx:
            target = torch.cat([target, target.new_tensor([self.eos_idx])])
        return {"id": index, "source": source, "target": target}

    def collater(self, samples, pad_to_length=None):
        return collate(
            samples,
            pad_idx=self.pad_idx,
            eos_idx=self.eos_idx,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=True,
            pad_to_length=pad_to_length,
        )

    def num_tokens(self, index):
        return int(self._sizes[index])

    def size(self, index):
        return (int(self._src_sizes[index]), int(self._tgt_sizes[index]))

    @property
    def sizes(self):
        return self._sizes

    def ordered_indices(self):
        return np.arange(len(self), dtype=np.int64)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        if self.supports_prefetch:
            self.dataset.prefetch(indices)
