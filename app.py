#!/usr/bin/env python3
"""Streamlit demo for transliteration."""

import argparse
from pathlib import Path

import streamlit as st
import torch

from transliteration.data.vocab import build_vocab, Vocab
from transliteration.data.dataset import TransliterationDataset, collate_fn
from transliteration.models.seq2seq import Seq2Seq, EncoderRNN, DecoderRNN
from transliteration.models.attention import AttentionSeq2Seq, Encoder, Decoder
from transliteration.models.transformer import TransformerSeq2Seq
from transliteration.utils.decode import greedy_decode, beam_search_decode


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
    pairs = {"train": [], "dev": [], "test": []}
    for split in ["train", "dev"]:
        fname = data_path / f"{lang}.translit.sampled.{split}.tsv"
        if fname.exists():
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        native, roman = parts[0].strip(), parts[1].strip()
                        if roman and native:
                            pairs[split].append((roman, native))

    src_vocab, tgt_vocab = build_vocab(pairs["train"] + pairs["dev"])

    # Build model
    if arch == "rnn":
        encoder = EncoderRNN(
            len(src_vocab), emb_dim, hidden_dim,
            num_layers=num_layers, cell_type=cell_type,
            bidirectional=bidirectional, dropout=dropout,
        )
        decoder = DecoderRNN(
            len(tgt_vocab), emb_dim, hidden_dim,
            num_layers=num_layers, cell_type=cell_type,
            dropout=dropout,
        )
        model = Seq2Seq(encoder, decoder, device)
    elif arch == "attention":
        encoder = Encoder(
            len(src_vocab), emb_dim, hidden_dim,
            num_layers=num_layers, cell_type=cell_type,
            bidirectional=True, dropout=dropout,
        )
        decoder = Decoder(
            len(tgt_vocab), emb_dim, hidden_dim,
            enc_dim=hidden_dim, num_layers=num_layers,
            cell_type=cell_type, dropout=dropout,
        )
        model = AttentionSeq2Seq(encoder, decoder, src_pad_idx=src_vocab.pad_idx)
    elif arch == "transformer":
        model = TransformerSeq2Seq(
            len(src_vocab), len(tgt_vocab),
            d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_ff=dim_ff,
            dropout=dropout, pad_idx=src_vocab.pad_idx,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model, src_vocab, tgt_vocab, device


def main():
    st.title("🔤 Telugu Transliteration Demo")
    st.caption("Roman → Telugu script character-level transliteration")

    # Sidebar config
    with st.sidebar:
        st.header("Model Config")
        checkpoint = st.text_input("Checkpoint path", "attention_best.pt")
        arch = st.selectbox("Architecture", ["attention", "rnn", "transformer"], index=0)
        lang = st.selectbox("Language", ["te", "hi"], index=0)
        data_dir = st.text_input("Data directory", "data")

        st.divider()
        st.subheader("Hyperparameters (must match checkpoint)")
        emb_dim = st.number_input("Embedding dim", value=64, min_value=16, max_value=512)
        hidden_dim = st.number_input("Hidden dim", value=256, min_value=16, max_value=512)
        d_model = st.number_input("Transformer d_model", value=128, min_value=32, max_value=512)
        nhead = st.number_input("Attention heads", value=4, min_value=1, max_value=16)
        num_layers = st.number_input("Layers", value=1, min_value=1, max_value=6)
        dim_ff = st.number_input("FF dim", value=256, min_value=64, max_value=2048)
        cell_type = st.selectbox("Cell type", ["lstm", "gru", "rnn"], index=0)
        bidirectional = st.checkbox("Bidirectional encoder", value=True)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)

        beam_size = st.slider("Beam size", 1, 10, 4)
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
    user_input = st.text_area("Type Telugu in Roman script", placeholder="e.g., namaste, dhanyavaadala, mumbai", height=100)

    if st.button("Transliterate", type="primary", use_container_width=True) and user_input:
        with st.spinner("Generating..."):
            if beam_size > 1:
                result = beam_search_decode(model, user_input.strip(), src_vocab, tgt_vocab, device, beam_size, max_len)
            else:
                result = greedy_decode(model, user_input.strip(), src_vocab, tgt_vocab, device, max_len)

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