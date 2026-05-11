"""Encoder-decoder language modeling task with prefix-split inputs.

Reads a standard fairseq monolingual binarised corpus (same format as
``language_modeling``), packs it into length-T blocks, and splits each block
into encoder source (first k tokens) and decoder target (remaining tokens).
The resulting samples are consumed unchanged by the standard
``transformer`` encoder-decoder architecture with ``cross_entropy`` criterion.
"""

import logging
import os
from dataclasses import dataclass, field

from fairseq import utils
from fairseq.data import TokenBlockDataset, data_utils
from fairseq.data.prefix_suffix_dataset import PrefixSuffixDataset
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.language_modeling import (
    LanguageModelingConfig,
    LanguageModelingTask,
)


logger = logging.getLogger(__name__)


@dataclass
class EncoderDecoderLanguageModelingConfig(LanguageModelingConfig):
    encoder_prefix_fraction: float = field(
        default=0.5,
        metadata={
            "help": "fraction of each token block fed to the encoder; the "
            "remainder becomes the decoder target (k = round(fraction * T))"
        },
    )


@register_task("encoder_decoder_language_modeling",
               dataclass=EncoderDecoderLanguageModelingConfig)
class EncoderDecoderLanguageModelingTask(LanguageModelingTask):
    """LM-on-encoder-decoder: same data as ``language_modeling`` but each block
    is split into a (source, target) pair so the model is a standard
    encoder-decoder transformer.
    """

    @classmethod
    def setup_task(cls, args, **kwargs):
        return super().setup_task(args, **kwargs)

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def load_dataset(self, split: str, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(f"Dataset not found: {split} ({split_path})")

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=False,
            use_plasma_view=self.args.use_plasma_view,
            split_path=split_path,
            plasma_path=self.args.plasma_path,
        )

        self.datasets[split] = PrefixSuffixDataset(
            dataset,
            vocab=self.dictionary,
            prefix_fraction=self.args.encoder_prefix_fraction,
        )

    def build_model(self, args, from_checkpoint=False):
        # Bypass LanguageModelingTask.build_model, which assumes a decoder-only
        # model with a ``supported_targets`` attribute. Use the generic
        # FairseqTask path instead.
        return LegacyFairseqTask.build_model(self, args, from_checkpoint=from_checkpoint)

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        raise NotImplementedError(
            "Inference dataset not implemented for encoder_decoder_language_modeling"
        )
