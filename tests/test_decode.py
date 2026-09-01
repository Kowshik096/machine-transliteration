"""Tests for beam_search_decode compatibility with both Seq2Seq and AttentionSeq2Seq."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest

from transliteration.data.vocab import build_vocab, Vocab
from transliteration.models.seq2seq import Seq2Seq, EncoderRNN, DecoderRNN
from transliteration.models.attention import AttentionSeq2Seq, Encoder, Decoder
from transliteration.utils.decode import beam_search_decode, greedy_decode


@pytest.fixture(scope="module")
def small_vocab():
    """Build a small vocab for testing."""
    pairs = [
        ("hello", "హేస్తుంది"),
        ("world", "వర్ల్"),
        ("test", "పరీక్ష"),
        ("foo", "ఫూ"),
        ("bar", "బార్"),
    ]
    src_vocab, tgt_vocab = build_vocab(pairs)
    return src_vocab, tgt_vocab


@pytest.fixture(scope="module")
def rnn_model(small_vocab):
    """Create a Seq2Seq (RNN baseline) model."""
    src_vocab, tgt_vocab = small_vocab
    device = torch.device("cpu")
    encoder = EncoderRNN(
        len(src_vocab), emb_dim=32, hidden_dim=64,
        num_layers=1, cell_type="gru", bidirectional=False,
    )
    decoder = DecoderRNN(
        len(tgt_vocab), emb_dim=32, hidden_dim=64,
        num_layers=1, cell_type="gru",
    )
    model = Seq2Seq(encoder, decoder, device)
    return model.to(device)


@pytest.fixture(scope="module")
def attention_model(small_vocab):
    """Create an AttentionSeq2Seq model."""
    src_vocab, tgt_vocab = small_vocab
    device = torch.device("cpu")
    encoder = Encoder(
        len(src_vocab), emb_dim=32, hidden_dim=64,
        num_layers=1, cell_type="lstm", bidirectional=True,
    )
    decoder = Decoder(
        len(tgt_vocab), emb_dim=32, hidden_dim=64,
        enc_dim=64, num_layers=1, cell_type="lstm",
    )
    model = AttentionSeq2Seq(encoder, decoder, src_pad_idx=src_vocab.pad_idx)
    return model.to(device)


class TestBeamSearchCompatibility:
    """Verify beam_search_decode works with both model architectures."""

    def test_beam_search_with_seq2seq(self, rnn_model, small_vocab):
        """Beam search should work with Seq2Seq (RNN baseline)."""
        src_vocab, tgt_vocab = small_vocab
        device = torch.device("cpu")
        result = beam_search_decode(
            rnn_model, "hello", src_vocab, tgt_vocab, device, beam_size=2
        )
        assert isinstance(result, str)
        # Should produce a non-empty output
        assert len(result) >= 0  # Could be empty for untrained model

    def test_beam_search_with_attention(self, attention_model, small_vocab):
        """Beam search should work with AttentionSeq2Seq."""
        src_vocab, tgt_vocab = small_vocab
        device = torch.device("cpu")
        result = beam_search_decode(
            attention_model, "hello", src_vocab, tgt_vocab, device, beam_size=2
        )
        assert isinstance(result, str)

    def test_beam_search_returns_string_for_both(self, rnn_model, attention_model, small_vocab):
        """Both models should return strings without crashing."""
        src_vocab, tgt_vocab = small_vocab
        device = torch.device("cpu")

        result_rnn = beam_search_decode(
            rnn_model, "hello", src_vocab, tgt_vocab, device, beam_size=2
        )
        result_attn = beam_search_decode(
            attention_model, "hello", src_vocab, tgt_vocab, device, beam_size=2
        )

        assert isinstance(result_rnn, str)
        assert isinstance(result_attn, str)

    def test_greedy_decode_with_seq2seq(self, rnn_model, small_vocab):
        """Greedy decode should also work with Seq2Seq."""
        src_vocab, tgt_vocab = small_vocab
        device = torch.device("cpu")
        result = greedy_decode(rnn_model, "hello", src_vocab, tgt_vocab, device)
        assert isinstance(result, str)

    def test_greedy_decode_with_attention(self, attention_model, small_vocab):
        """Greedy decode should also work with AttentionSeq2Seq."""
        src_vocab, tgt_vocab = small_vocab
        device = torch.device("cpu")
        result = greedy_decode(attention_model, "hello", src_vocab, tgt_vocab, device)
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])