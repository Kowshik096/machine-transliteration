"""Explicit architecture presets matching the trained checkpoints.

Do NOT use a single universal hyperparameter set: each architecture was
trained with a different configuration.  These presets are the canonical
source of truth; ``app.py``, ``scripts/train.py`` and ``scripts/eval.py``
must resolve defaults from here.
"""

from typing import Any

import torch

# Known-good configurations (verified against *_best.pt/ckpt shapes).
BASELINE_PRESET: dict[str, Any] = {
    "arch": "rnn",
    "emb_dim": 64,
    "hidden_dim": 256,
    "num_layers": 2,
    "cell_type": "lstm",
    "bidirectional": False,
    "dropout": 0.2,
}

ATTENTION_PRESET: dict[str, Any] = {
    "arch": "attention",
    "emb_dim": 64,
    "hidden_dim": 256,
    "num_layers": 1,
    "cell_type": "lstm",
    "bidirectional": True,
    "dropout": 0.1,
}

TRANSFORMER_PRESET: dict[str, Any] = {
    "arch": "transformer",
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dim_ff": 256,
    "dropout": 0.1,
}

ARCH_PRESETS: dict[str, dict[str, Any]] = {
    "rnn": BASELINE_PRESET,
    "baseline": BASELINE_PRESET,
    "attention": ATTENTION_PRESET,
    "transformer": TRANSFORMER_PRESET,
}


def get_preset(arch: str) -> dict[str, Any]:
    """Return a copy of the preset for ``arch``.

    Raises ValueError for unknown architectures.
    """
    key = arch.lower()
    if key not in ARCH_PRESETS:
        raise ValueError(
            f"Unknown architecture: {arch!r}. Expected one of {sorted(ARCH_PRESETS)}."
        )
    return dict(ARCH_PRESETS[key])


def build_model(
    arch: str,
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: torch.device,
    src_pad_idx: int = 0,
    **overrides: Any,
) -> torch.nn.Module:
    """Build a model for ``arch`` starting from its preset.

    Any hyperparameter in the preset may be overridden via kwargs.
    """
    from transliteration.models.attention import AttentionSeq2Seq, Decoder, Encoder
    from transliteration.models.seq2seq import DecoderRNN, EncoderRNN, Seq2Seq
    from transliteration.models.transformer import TransformerSeq2Seq

    preset = get_preset(arch)
    preset.update(overrides or {})
    arch_key = preset.get("arch", arch.lower())

    if arch_key == "rnn":
        rnn_encoder = EncoderRNN(
            src_vocab_size,
            preset["emb_dim"],
            preset["hidden_dim"],
            num_layers=preset["num_layers"],
            cell_type=preset["cell_type"],
            bidirectional=preset["bidirectional"],
            dropout=preset["dropout"],
        )
        rnn_decoder = DecoderRNN(
            tgt_vocab_size,
            preset["emb_dim"],
            preset["hidden_dim"],
            num_layers=preset["num_layers"],
            cell_type=preset["cell_type"],
            dropout=preset["dropout"],
        )
        return Seq2Seq(rnn_encoder, rnn_decoder, device).to(device)
    if arch_key == "attention":
        attn_encoder = Encoder(
            src_vocab_size,
            preset["emb_dim"],
            preset["hidden_dim"],
            num_layers=preset["num_layers"],
            cell_type=preset["cell_type"],
            bidirectional=True,
            dropout=preset["dropout"],
        )
        attn_decoder = Decoder(
            tgt_vocab_size,
            preset["emb_dim"],
            preset["hidden_dim"],
            enc_dim=preset["hidden_dim"],
            num_layers=preset["num_layers"],
            cell_type=preset["cell_type"],
            dropout=preset["dropout"],
        )
        return AttentionSeq2Seq(
            attn_encoder, attn_decoder, src_pad_idx=src_pad_idx
        ).to(device)
    if arch_key == "transformer":
        return TransformerSeq2Seq(
            src_vocab_size,
            tgt_vocab_size,
            d_model=preset["d_model"],
            nhead=preset["nhead"],
            num_layers=preset["num_layers"],
            dim_ff=preset["dim_ff"],
            dropout=preset["dropout"],
            pad_idx=src_pad_idx,
        ).to(device)
    raise ValueError(f"Unknown architecture: {arch!r}")
