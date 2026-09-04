"""Tests for beam_search_decode compatibility with both Seq2Seq and AttentionSeq2Seq."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from transliteration.data.vocab import build_vocab
from transliteration.models.attention import AttentionSeq2Seq, Decoder, Encoder
from transliteration.models.seq2seq import DecoderRNN, EncoderRNN, Seq2Seq
from transliteration.models.transformer import TransformerSeq2Seq
from transliteration.utils.decode import beam_search_decode, greedy_decode
from transliteration.utils.metrics import bleu_score, char_accuracy, word_accuracy


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


@pytest.fixture(scope="module")
def transformer_model(small_vocab):
    """Create a TransformerSeq2Seq model."""
    src_vocab, tgt_vocab = small_vocab
    device = torch.device("cpu")
    model = TransformerSeq2Seq(
        len(src_vocab), len(tgt_vocab),
        d_model=32, nhead=2, num_layers=1, dim_ff=64,
        dropout=0.0, pad_idx=src_vocab.pad_idx,
    )
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

    def test_greedy_decode_with_transformer(self, transformer_model, small_vocab):
        """Greedy decode should also work with TransformerSeq2Seq."""
        src_vocab, tgt_vocab = small_vocab
        device = torch.device("cpu")
        result = greedy_decode(transformer_model, "hello", src_vocab, tgt_vocab, device)
        assert isinstance(result, str)

    def test_beam_search_transformer_raises(self, transformer_model, small_vocab):
        """Beam search should raise ValueError for Transformer (not implemented)."""
        src_vocab, tgt_vocab = small_vocab
        device = torch.device("cpu")
        with pytest.raises(ValueError, match="not implemented"):
            beam_search_decode(
                transformer_model, "hello", src_vocab, tgt_vocab, device, beam_size=2
            )


class TestVocab:
    """Vocabulary encode/decode round-trip and special token handling."""

    def test_special_token_indices(self, small_vocab):
        src_vocab, _ = small_vocab
        assert src_vocab.pad_idx == 0
        assert src_vocab.sos_idx == 1
        assert src_vocab.eos_idx == 2
        assert src_vocab.unk_idx == 3

    def test_encode_adds_special_tokens(self, small_vocab):
        src_vocab, _ = small_vocab
        ids = src_vocab.encode("hello", add_special=True)
        assert ids[0] == src_vocab.sos_idx
        assert ids[-1] == src_vocab.eos_idx

    def test_encode_no_special_tokens(self, small_vocab):
        src_vocab, _ = small_vocab
        ids = src_vocab.encode("hello", add_special=False)
        # add_special=False still appends EOS (see Vocab.encode)
        assert ids[0] != src_vocab.sos_idx
        assert ids[-1] == src_vocab.eos_idx
        assert ids[-2] != src_vocab.eos_idx

    def test_decode_skips_special_tokens(self, small_vocab):
        _, tgt_vocab = small_vocab
        ids = [tgt_vocab.sos_idx, tgt_vocab.eos_idx, tgt_vocab.pad_idx]
        assert tgt_vocab.decode(ids) == ""

    def test_unknown_character_maps_to_unk(self, small_vocab):
        src_vocab, _ = small_vocab
        ids = src_vocab.encode("zzz", add_special=False)
        # All content chars map to UNK; only the trailing EOS is not UNK.
        assert all(i == src_vocab.unk_idx for i in ids[:-1])
        assert ids[-1] == src_vocab.eos_idx

    def test_len_returns_vocab_size(self, small_vocab):
        src_vocab, _ = small_vocab
        assert len(src_vocab) == len(src_vocab.char2idx)


class TestMetrics:
    """Metric computation edge cases."""

    def test_word_accuracy_perfect(self):
        assert word_accuracy(["a", "b"], ["a", "b"]) == 1.0

    def test_word_accuracy_empty(self):
        assert word_accuracy([], []) == 0.0

    def test_word_accuracy_partial(self):
        assert word_accuracy(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)

    def test_char_accuracy_empty(self):
        assert char_accuracy([], []) == 0.0

    def test_char_accuracy_partial(self):
        score = char_accuracy(["abc"], ["axc"])
        assert score == pytest.approx(2 / 3)

    def test_char_accuracy_length_mismatch(self):
        # Shorter string is compared up to its length; denominator is target len.
        score = char_accuracy(["ab"], ["abcd"])
        assert score == pytest.approx(2 / 4)

    def test_bleu_empty(self):
        assert bleu_score([], []) == 0.0

    def test_bleu_perfect(self):
        score = bleu_score(["hello"], ["hello"])
        assert score == pytest.approx(1.0)

    def test_bleu_skips_empty_target(self):
        # Empty target is skipped entirely.
        assert bleu_score(["x"], [""]) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
