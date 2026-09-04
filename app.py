#!/usr/bin/env python3
"""Streamlit demo for transliteration."""

from pathlib import Path

import streamlit as st
import torch

from transliteration.data.vocab import build_vocab
from transliteration.models.attention import AttentionSeq2Seq, Decoder, Encoder
from transliteration.models.presets import ARCH_PRESETS, get_preset
from transliteration.models.seq2seq import DecoderRNN, EncoderRNN, Seq2Seq
from transliteration.models.transformer import TransformerSeq2Seq
from transliteration.utils.checkpoints import load_model_state_dict
from transliteration.utils.decode import decode_with_fallback

# Page config
st.set_page_config(
    page_title="Telugu Transliteration",
    page_icon="🔤",
    layout="centered",
)


@st.cache_resource
def load_model_and_vocab(checkpoint_path: str, arch: str, lang: str, data_dir: str,
                         emb_dim: int, hidden_dim: int, d_model: int, nhead: int,
                         num_layers: int, dim_ff: int, cell_type: str, bidirectional: bool,
                         dropout: float):
    """Load model, vocab, and device (cached)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data for vocab
    data_path = Path(data_dir)
    pairs: dict[str, list] = {"train": [], "dev": [], "test": []}
    for split in ["train", "dev"]:
        fname = data_path / f"{lang}.translit.sampled.{split}.tsv"
        if fname.exists():
            with open(fname, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        native, roman = parts[0].strip(), parts[1].strip()
                        if roman and native:
                            pairs[split].append((roman, native))

    src_vocab, tgt_vocab = build_vocab(pairs["train"] + pairs["dev"])

    # Build model from explicit per-architecture presets. Caller-provided
    # hyperparameters override the preset so retrained variants still load.
    app_model: torch.nn.Module
    if arch == "rnn":
        rnn_encoder = EncoderRNN(
            len(src_vocab), emb_dim, hidden_dim,
            num_layers=num_layers, cell_type=cell_type,
            bidirectional=bidirectional, dropout=dropout,
        )
        rnn_decoder = DecoderRNN(
            len(tgt_vocab), emb_dim, hidden_dim,
            num_layers=num_layers, cell_type=cell_type,
            dropout=dropout,
        )
        app_model = Seq2Seq(rnn_encoder, rnn_decoder, device)
    elif arch == "attention":
        attn_encoder = Encoder(
            len(src_vocab), emb_dim, hidden_dim,
            num_layers=num_layers, cell_type=cell_type,
            bidirectional=True, dropout=dropout,
        )
        attn_decoder = Decoder(
            len(tgt_vocab), emb_dim, hidden_dim,
            enc_dim=hidden_dim, num_layers=num_layers,
            cell_type=cell_type, dropout=dropout,
        )
        app_model = AttentionSeq2Seq(
            attn_encoder, attn_decoder, src_pad_idx=src_vocab.pad_idx
        )
    elif arch == "transformer":
        app_model = TransformerSeq2Seq(
            len(src_vocab), len(tgt_vocab),
            d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_ff=dim_ff,
            dropout=dropout, pad_idx=src_vocab.pad_idx,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}. Expected one of {sorted(ARCH_PRESETS)}.")

    # Handles both raw state_dicts (*_best.pt) and Lightning envelopes
    # (baseline_best.ckpt with model.* prefix).
    load_model_state_dict(app_model, checkpoint_path, map_location=device)
    app_model.to(device)
    app_model.eval()

    return app_model, src_vocab, tgt_vocab, device


def preset_defaults(arch: str) -> dict:
    """Return sidebar defaults matching the trained checkpoint for arch."""
    try:
        return get_preset(arch)
    except ValueError:
        return get_preset("attention")


def main():
    st.title("🔤 Telugu Transliteration Demo")
    st.caption("Roman → Telugu script character-level transliteration")

    # Sidebar config
    with st.sidebar:
        st.header("Model Config")
        arch = st.selectbox("Architecture", ["attention", "rnn", "transformer"], index=0)
        preset = preset_defaults(arch)
        # Checkpoint default matches the selected architecture's trained file.
        default_ckpt = {
            "attention": "attention_best.pt",
            "rnn": "baseline_best.ckpt",
            "transformer": "transformer_best.pt",
        }[arch]
        checkpoint = st.text_input("Checkpoint path", default_ckpt)
        lang = st.selectbox("Language", ["te", "hi"], index=0)
        data_dir = st.text_input("Data directory", "data")

        st.divider()
        st.subheader("Hyperparameters (must match checkpoint)")
        st.caption(f"Defaults shown are the {arch} preset: {preset}")
        emb_dim = st.number_input(
            "Embedding dim",
            value=int(preset.get("emb_dim", 64)),
            min_value=16, max_value=512,
        )
        hidden_dim = st.number_input(
            "Hidden dim",
            value=int(preset.get("hidden_dim", 256)),
            min_value=16, max_value=512,
        )
        d_model = st.number_input(
            "Transformer d_model",
            value=int(preset.get("d_model", 128)),
            min_value=32, max_value=512,
        )
        nhead = st.number_input(
            "Attention heads",
            value=int(preset.get("nhead", 4)),
            min_value=1, max_value=16,
        )
        num_layers = st.number_input(
            "Layers",
            value=int(preset.get("num_layers", 1 if arch != "transformer" else 3)),
            min_value=1, max_value=6,
        )
        dim_ff = st.number_input(
            "FF dim", value=int(preset.get("dim_ff", 256)),
            min_value=64, max_value=2048,
        )
        cell_type = st.selectbox(
            "Cell type", ["lstm", "gru", "rnn"],
            index=["lstm", "gru", "rnn"].index(str(preset.get("cell_type", "lstm"))),
        )
        bidirectional = st.checkbox(
            "Bidirectional encoder",
            value=bool(preset.get("bidirectional", True)),
        )
        dropout = st.slider(
            "Dropout", 0.0, 0.5, float(preset.get("dropout", 0.1)), 0.05,
        )

        beam_size = st.slider("Beam size", 1, 10, 4)
        if arch == "transformer":
            st.info("Transformer uses greedy decoding; beam_size > 1 falls back to greedy.")
        max_len = st.number_input("Max output length", value=64, min_value=16, max_value=256)

    # Load model
    try:
        model, src_vocab, tgt_vocab, device = load_model_and_vocab(
            checkpoint, arch, lang, data_dir,
            emb_dim, hidden_dim, d_model, nhead,
            num_layers, dim_ff, cell_type, bidirectional, dropout,
        )
        st.success(f"✅ Model loaded on {device}")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    # Input
    st.subheader("Input (Roman)")
    user_input = st.text_area(
        "Type Telugu in Roman script",
        placeholder="e.g., namaste, dhanyavaadala, mumbai",
        height=100,
    )

    if st.button("Transliterate", type="primary", use_container_width=True) and user_input:
        with st.spinner("Generating..."):
            result, fell_back = decode_with_fallback(
                model, user_input.strip(), src_vocab, tgt_vocab, device,
                beam_size, max_len,
            )
            if fell_back:
                st.warning(
                    "Beam search is not implemented for the Transformer; "
                    "used greedy decoding instead (beam_size=1)."
                )

        st.subheader("Output (Telugu)")
        st.code(result, language=None)

        # Show character-by-character
        st.caption("Character mapping:")
        cols = st.columns(min(len(result), 10))
        for i, (r, t) in enumerate(zip(user_input.strip(), result)):
            if i < 10:
                cols[i].metric(r, t)

    st.divider()
    st.markdown("""
    **Model Details:**
    - **Architecture**: Seq2Seq with Bahdanau Attention (BiLSTM encoder + LSTM decoder)
    - **Dataset**: Aksharantar Telugu (51,200 train / 4,096 dev / 4,096 test)
    - **Best Results**: 49.6% word accuracy / 82.6% char accuracy (beam=4)
    - **Framework**: PyTorch
    """)


if __name__ == "__main__":
    main()
