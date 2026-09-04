"""Vocabulary utilities for character-level transliteration."""

from typing import Optional

SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


class Vocab:
    """Character vocabulary with special tokens."""

    def __init__(
        self,
        char2idx: dict[str, int],
        idx2char: dict[int, str],
    ):
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.pad_idx = char2idx["<PAD>"]
        self.sos_idx = char2idx["<SOS>"]
        self.eos_idx = char2idx["<EOS>"]
        self.unk_idx = char2idx["<UNK>"]

    def __len__(self) -> int:
        return len(self.char2idx)

    def encode(
        self, text: str, add_special: bool = True, max_len: Optional[int] = None
    ) -> list[int]:
        ids = [self.char2idx.get(c, self.unk_idx) for c in text]
        if max_len:
            ids = ids[: max_len - (2 if add_special else 1)]
        if add_special:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        else:
            ids = ids + [self.eos_idx]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        specials = {self.pad_idx, self.sos_idx, self.eos_idx} if skip_special else set()
        return "".join(self.idx2char[i] for i in ids if i not in specials)


def build_vocab(pairs: list[tuple[str, str]]) -> tuple[Vocab, Vocab]:
    """Build source and target vocabularies from pairs."""
    src_chars: set[str] = set()
    tgt_chars: set[str] = set()
    for src, tgt in pairs:
        src_chars.update(src)
        tgt_chars.update(tgt)

    src_tokens = SPECIAL_TOKENS + sorted(src_chars)
    tgt_tokens = SPECIAL_TOKENS + sorted(tgt_chars)

    src2idx = {c: i for i, c in enumerate(src_tokens)}
    tgt2idx = {c: i for i, c in enumerate(tgt_tokens)}
    idx2src = {i: c for c, i in src2idx.items()}
    idx2tgt = {i: c for c, i in tgt2idx.items()}

    src_vocab = Vocab(src2idx, idx2src)
    tgt_vocab = Vocab(tgt2idx, idx2tgt)
    return src_vocab, tgt_vocab
