"""Dataset and DataLoader utilities."""


import torch
from torch.utils.data import DataLoader, Dataset

from transliteration.data.vocab import Vocab


class TransliterationDataset(Dataset):
    """Character-level transliteration dataset."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_len: int = 64,
    ):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.pairs[idx]
        src_ids = self.src_vocab.encode(src, add_special=False, max_len=self.max_len)
        tgt_ids = self.tgt_vocab.encode(tgt, add_special=True, max_len=self.max_len)
        return torch.LongTensor(src_ids), torch.LongTensor(tgt_ids)


def collate_fn(batch, pad_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad batch to max length and return padded tensors + source lengths."""
    srcs, tgts = zip(*batch)
    src_max = max(len(s) for s in srcs)
    tgt_max = max(len(t) for t in tgts)
    src_pad = torch.full((len(srcs), src_max), pad_idx, dtype=torch.long)
    tgt_pad = torch.full((len(tgts), tgt_max), pad_idx, dtype=torch.long)
    src_lens = torch.LongTensor([len(s) for s in srcs])
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, : len(s)] = s
        tgt_pad[i, : len(t)] = t
    return src_pad, tgt_pad, src_lens


def create_dataloaders(
    train_pairs: list[tuple[str, str]],
    dev_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    batch_size: int = 128,
    max_len: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/dev/test DataLoaders."""
    train_loader = DataLoader(
        TransliterationDataset(train_pairs, src_vocab, tgt_vocab, max_len),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=src_vocab.pad_idx),
    )
    dev_loader = DataLoader(
        TransliterationDataset(dev_pairs, src_vocab, tgt_vocab, max_len),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx=src_vocab.pad_idx),
    )
    test_loader = DataLoader(
        TransliterationDataset(test_pairs, src_vocab, tgt_vocab, max_len),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx=src_vocab.pad_idx),
    )
    return train_loader, dev_loader, test_loader
