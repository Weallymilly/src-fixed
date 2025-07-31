from __future__ import annotations
from collections import OrderedDict
import random

import numpy as np
import torch
import pandas as pd

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel
from esm import BatchConverter, pretrained


class RandomCropBatchConverter(BatchConverter):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    For sequences over max_len, randomly crop a window.
    """

    def __init__(self, alphabet: Alphabet, max_len: int):
        super().__init__(alphabet)
        self.max_len = max_len

    def __call__(self, raw_batch: list[tuple[str, str]]) -> tuple:
        cropped_batch = [(label, self._crop(seq)) for label, seq in raw_batch]
        return super().__call__(cropped_batch)

    def _crop(self, seq: str) -> str:
        if len(seq) <= self.max_len:
            return seq
        start_idx = np.random.choice(len(seq) - self.max_len + 1)
        return seq[start_idx: start_idx+self.max_len]


class CosupRandomCropBatchConverter(RandomCropBatchConverter):
    def __call__(self, raw_batch: list[tuple[str, str]]) -> tuple:
        datatype = [tup[0] for tup in raw_batch]
        assert len(set(datatype)) == 1
        datatype = datatype[0]
        raw_batch = [tup[1] for tup in raw_batch]
        return datatype, super().__call__(raw_batch)


class CSVBatchedDataset(FastaBatchedDataset):
    @classmethod
    def from_file(cls, csv_file: str) -> CSVBatchedDataset:
        df = pd.read_csv(csv_file)
        return cls(df.log_fitness.values, df.seq.values)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> CSVBatchedDataset:
        return cls(df.log_fitness.values, df.seq.values)


class CosupervisionDataset(object):
    def __init__(self, fasta_batched_dataset: FastaBatchedDataset, csv_batched_dataset: CSVBatchedDataset):
        self.datasets = OrderedDict()
        self.datasets['unsup'] = fasta_batched_dataset
        self.datasets['sup'] = csv_batched_dataset
        self.lens = OrderedDict()
        for k, d in self.datasets.items():
            self.lens[k] = len(d)

    def __len__(self) -> int:
        return sum(self.lens.values())

    def __getitem__(self, idx: int) -> tuple:
        cumlen = 0
        for k in self.datasets.keys():
            if idx < cumlen + self.lens[k]:
                return k, self.datasets[k].__getitem__(idx - cumlen)
            cumlen += self.lens[k]

    def _offset_indices(self, batches, offset):
        return [[idx + offset for idx in batch] for batch in batches]

    def _get_batch_indices(self, per_dataset_batches):
        batches = []
        cumlen = 0
        for k in self.datasets.keys():
            batches += self._offset_indices(per_dataset_batches[k], cumlen)
            cumlen += self.lens[k]
        random.shuffle(batches)
        return batches

    def get_split_batch_indices(self, toks_per_batch, extra_toks_per_seq=0,
        val_split=0.2):
        train_per_dataset_batches = OrderedDict()
        val_per_dataset_batches = OrderedDict()
        for k, d in self.datasets.items():
            batches = d.get_batch_indices(toks_per_batch, extra_toks_per_seq)
            random.shuffle(batches)
            split = int(np.floor(val_split * len(batches)))
            train_batches, val_batches = batches[split:], batches[:split]
            train_per_dataset_batches[k] = train_batches
            val_per_dataset_batches[k] = val_batches
        return self._get_batch_indices(train_per_dataset_batches
            ), self._get_batch_indices(val_per_dataset_batches)


class MaskedFastaBatchedDataset(FastaBatchedDataset):
    """
    For each sequence, mask all the mutated positions in one data entry.
    """
    def __init__(self, sequence_labels: list[float], sequence_strs: list[str], mask_positions: list[list[int]] | None = None):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)
        if mask_positions is not None:
            self.mask_positions = list(mask_positions)

    @classmethod
    def from_file(cls, fasta_file: str, wt: str) -> MaskedFastaBatchedDataset:
        ds = super().from_file(fasta_file)
        sequence_labels, sequence_strs = ds.sequence_labels, ds.sequence_strs
        mask_positions = []
        for s in sequence_strs:
            # +1 for start token
            positions = [pos+1 for pos in range(len(wt)) if s[pos] != wt[pos]]
            mask_positions.append(positions)
        return cls(sequence_labels, sequence_strs, mask_positions)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, wt: str) -> MaskedFastaBatchedDataset:
        sequence_labels, sequence_strs = df.log_fitness.values, df.seq.values
        mask_positions = []
        for s in sequence_strs:
            # +1 for start token
            positions = [pos+1 for pos in range(len(wt)) if s[pos] != wt[pos]]
            mask_positions.append(positions)
        return cls(sequence_labels, sequence_strs, mask_positions)

    def __getitem__(self, idx: int) -> tuple[float, str, list[int]]:
        return self.sequence_labels[idx], self.sequence_strs[idx], self.mask_positions[idx]


class PLLFastaBatchedDataset(MaskedFastaBatchedDataset):
    """Batched dataset specialized for computing pseudo log likelihoods.
    For each sequence, mask each of the mutated positions as a data entry.
    """
    @classmethod
    def from_file(cls, fasta_file: str, wt: str) -> PLLFastaBatchedDataset:
        ds = super().from_file(fasta_file, wt)
        pll_sequence_labels = []
        pll_sequence_strs = []
        pll_mask_positions = []
        for i in range(len(ds.sequence_strs)):
            s = ds.sequence_strs[i]
            l = ds.sequence_labels[i]
            m = ds.mask_positions[i]
            if s == wt:
                pll_sequence_labels.append(l)
                pll_sequence_strs.append(s)
                pll_mask_positions.append(1)   # arbitrary choice of pos
                continue
            for pos in m:
                pll_sequence_labels.append(l)
                pll_sequence_strs.append(s)
                pll_mask_positions.append(pos)
        return cls(pll_sequence_labels, pll_sequence_strs, pll_mask_positions)


class MaskedBatchConverter(BatchConverter):
    """Batch converter to be used with MaskedFastaBatchedDataset."""

    def __call__(self, raw_batch: list[tuple[float, str, list[int]]]) -> tuple:
        _raw_batch = [(l, s) for l, s, p in raw_batch]
        mask_pos = [p for l, s, p in raw_batch]
        mask_pos = torch.tensor(mask_pos).long()
        labels, strs, tokens = super().__call__(_raw_batch)
        return labels, strs, tokens, mask_pos


class PLLBatchConverter(MaskedBatchConverter):
    pass


def random_mask_tokens(inputs: torch.Tensor, alphabet: Alphabet, mlm_probability: float = 0.15) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling.
    Among the 15% masks: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    device = inputs.device
    # We sample a few tokens in each sequence for MLM training
    # (with probability `mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability,
            device=device)
    special_tokens_mask = (inputs == alphabet.padding_idx)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8,
        device=device)).bool() & masked_indices
    inputs[indices_replaced] = alphabet.mask_idx

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1,
        device=device)).bool() & masked_indices & ~indices_replaced
    random_AAs = torch.randint(len(alphabet.prepend_toks),
            len(alphabet.standard_toks), labels.shape,
            dtype=torch.long, device=device)
    inputs[indices_random] = random_AAs[indices_random]

    # The rest of the time (10% of the time)
    # we keep the masked input tokens unchanged
    return inputs, labels, masked_indices