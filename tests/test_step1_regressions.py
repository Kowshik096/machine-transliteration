"""Step 1 regression tests for Critical/High blockers.

Covers: data package trackability, Lightning vs raw checkpoint loading,
baseline preset compatibility, Transformer autoregressive decoding,
EOS/max_len behavior, and Transformer beam fallback.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn

from transliteration.data.vocab import Vocab, build_vocab
from transliteration.models.presets import (
    ARCH_PRESETS,
    get_preset,
)
from transliteration.models.presets import build_model as build_preset_model
from transliteration.models.seq2seq import DecoderRNN, EncoderRNN, Seq2Seq
from transliteration.utils.checkpoints import (
    is_lightning_checkpoint,
    load_model_state_dict,
    normalize_state_dict,
)
from transliteration.utils.decode import (
    beam_search_decode,
    decode_with_fallback,
    greedy_decode,
    resolve_beam_size,
)


def _tiny_vocab() -> tuple[Vocab, Vocab]:
    pairs = [("ab", "xy"), ("bc", "yz"), ("ca", "zx")]
    return build_vocab(pairs)


class TestDataPackage:
    def test_data_package_importable(self) -> None:
        import transliteration.data as data_pkg

        assert hasattr(data_pkg, "TransliterationDataset")
        assert hasattr(data_pkg, "build_vocab")
        assert hasattr(data_pkg, "Vocab")

    def test_dataset_collate_import(self) -> None:
        from transliteration.data import TransliterationDataset
        from transliteration.data.dataset import collate_fn

        src_v, tgt_v = _tiny_vocab()
        ds = TransliterationDataset([("ab", "xy")], src_v, tgt_v)
        src, tgt = ds[0]
        assert src.numel() > 0 and tgt.numel() > 0
        src_pad, tgt_pad, src_lens = collate_fn([(src, tgt)])
        assert src_pad.size(0) == 1 and src_lens.item() == len(src)


class TestCheckpointNormalization:
    def test_raw_state_dict_passthrough(self) -> None:
        raw = {"encoder.embedding.weight": torch.zeros(2, 2)}
        assert normalize_state_dict(raw) == raw

    def test_lightning_prefix_stripped(self) -> None:
        raw = {
            "epoch": 9,
            "state_dict": {
                "model.encoder.embedding.weight": torch.zeros(2, 2),
                "model.decoder.fc_out.weight": torch.zeros(3, 2),
            },
        }
        assert is_lightning_checkpoint(raw) is True
        norm = normalize_state_dict(raw)
        assert set(norm) == {
            "encoder.embedding.weight",
            "decoder.fc_out.weight",
        }

    def test_is_lightning_false_for_raw(self) -> None:
        assert is_lightning_checkpoint({"encoder.weight": torch.zeros(1)}) is False

    def test_load_raw_state_dict(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        device = torch.device("cpu")
        src_v, tgt_v = _tiny_vocab()
        enc = EncoderRNN(len(src_v), 8, 16, num_layers=1, cell_type="gru")
        dec = DecoderRNN(len(tgt_v), 8, 16, num_layers=1, cell_type="gru")
        model = Seq2Seq(enc, dec, device)
        ckpt = tmp_path / "raw.pt"
        torch.save(model.state_dict(), ckpt)
        enc2 = EncoderRNN(len(src_v), 8, 16, num_layers=1, cell_type="gru")
        dec2 = DecoderRNN(len(tgt_v), 8, 16, num_layers=1, cell_type="gru")
        model2 = Seq2Seq(enc2, dec2, device)
        # Perturb to prove loading restores values.
        with torch.no_grad():
            for p in model2.parameters():
                p.add_(1.0)
        load_model_state_dict(model2, ckpt, map_location="cpu")
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    def test_load_lightning_prefixed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        device = torch.device("cpu")
        src_v, tgt_v = _tiny_vocab()
        enc = EncoderRNN(len(src_v), 8, 16, num_layers=1, cell_type="gru")
        dec = DecoderRNN(len(tgt_v), 8, 16, num_layers=1, cell_type="gru")
        model = Seq2Seq(enc, dec, device)
        raw_sd = model.state_dict()
        lightning = {
            "epoch": 9,
            "global_step": 100,
            "pytorch-lightning_version": "2.6.5",
            "state_dict": {f"model.{k}": v for k, v in raw_sd.items()},
            "loops": {},
        }
        ckpt = tmp_path / "lightning.ckpt"
        torch.save(lightning, ckpt)
        enc2 = EncoderRNN(len(src_v), 8, 16, num_layers=1, cell_type="gru")
        dec2 = DecoderRNN(len(tgt_v), 8, 16, num_layers=1, cell_type="gru")
        model2 = Seq2Seq(enc2, dec2, device)
        load_model_state_dict(model2, ckpt, map_location="cpu")
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)


class TestPresets:
    def test_baseline_preset_matches_checkpoint(self) -> None:
        preset = get_preset("rnn")
        assert preset["emb_dim"] == 64
        assert preset["hidden_dim"] == 256
        assert preset["num_layers"] == 2
        assert preset["bidirectional"] is False
        assert preset["cell_type"] == "lstm"

    def test_attention_preset(self) -> None:
        preset = get_preset("attention")
        assert (preset["emb_dim"], preset["hidden_dim"]) == (64, 256)
        assert preset["num_layers"] == 1
        assert preset["bidirectional"] is True

    def test_transformer_preset_three_layers(self) -> None:
        preset = get_preset("transformer")
        assert preset["d_model"] == 128
        assert preset["nhead"] == 4
        assert preset["num_layers"] == 3
        assert preset["dim_ff"] == 256

    def test_unknown_arch_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Unknown architecture"):
            get_preset("nope")

    def test_build_preset_models(self) -> None:
        device = torch.device("cpu")
        for arch in ("rnn", "attention", "transformer"):
            model = build_preset_model(arch, 12, 14, device, src_pad_idx=0)
            assert isinstance(model, nn.Module)
        assert "rnn" in ARCH_PRESETS and "transformer" in ARCH_PRESETS


class _CountingTransformer(nn.Module):
    """Deterministic mock Transformer exposing incremental prefixes.

    - Has a ``transformer`` attribute so ``_is_transformer`` is True.
    - Records every target-prefix length seen.
    - Predicts token id 4 twice, then EOS, so autoregressive decoding
      terminates quickly. A single-forward dummy of length 64 would see
      ``tgt_len == 64`` and (by design) predict non-EOS everywhere,
      producing the old 64x repeated-token failure.
    """

    def __init__(self, tgt_vocab: Vocab):
        super().__init__()
        self.transformer = nn.Identity()
        self.tgt_vocab = tgt_vocab
        self.seen_lengths: list[int] = []

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        self.seen_lengths.append(int(tgt.size(1)))
        batch, length = tgt.shape
        vocab_size = len(self.tgt_vocab)
        logits = torch.full((batch, length, vocab_size), -10.0)
        if length == 64:
            # Old code path: single SOS+PAD*63 dummy -> all non-EOS.
            logits[:, :, 4] = 10.0
        elif length <= 2:
            logits[:, -1, 4] = 10.0
        else:
            logits[:, -1, self.tgt_vocab.eos_idx] = 10.0
        return logits


class TestTransformerAutoregressive:
    def test_greedy_is_autoregressive_not_single_dummy(self) -> None:
        _, tgt_v = _tiny_vocab()
        src_v, _ = _tiny_vocab()
        device = torch.device("cpu")
        model = _CountingTransformer(tgt_v)
        out = greedy_decode(model, "ab", src_v, tgt_v, device, max_len=64)
        # New code feeds prefixes 1,2,3...; old code made a single call of 64.
        assert len(model.seen_lengths) > 1
        assert model.seen_lengths[0] == 1
        assert model.seen_lengths == sorted(model.seen_lengths)
        # Must terminate via EOS, not emit 64 repeated tokens.
        assert len(out) <= 64
        assert not (len(out) == 64 and len(set(out)) == 1)
        assert len(out) == 2  # two non-EOS steps (id 4 twice) then EOS

    def test_greedy_respects_max_len(self) -> None:
        class NeverEos(nn.Module):
            def __init__(self, tgt_vocab: Vocab):
                super().__init__()
                self.transformer = nn.Identity()
                self.tgt_vocab = tgt_vocab

            def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
                batch, length = tgt.shape
                logits = torch.full((batch, length, len(self.tgt_vocab)), -10.0)
                logits[:, -1, 4] = 10.0
                return logits

        src_v, tgt_v = _tiny_vocab()
        device = torch.device("cpu")
        out = greedy_decode(
            NeverEos(tgt_v), "ab", src_v, tgt_v, device, max_len=10
        )
        assert len(out) <= 10
        assert "<PAD>" not in out and "<SOS>" not in out

    def test_greedy_immediate_eos_gives_empty(self) -> None:
        class ImmediateEos(nn.Module):
            def __init__(self, tgt_vocab: Vocab):
                super().__init__()
                self.transformer = nn.Identity()
                self.tgt_vocab = tgt_vocab

            def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
                batch, length = tgt.shape
                logits = torch.full((batch, length, len(self.tgt_vocab)), -10.0)
                logits[:, -1, self.tgt_vocab.eos_idx] = 10.0
                return logits

        src_v, tgt_v = _tiny_vocab()
        device = torch.device("cpu")
        out = greedy_decode(
            ImmediateEos(tgt_v), "ab", src_v, tgt_v, device, max_len=64
        )
        assert out == ""

    def test_beam_still_raises_for_transformer(self) -> None:
        import pytest

        src_v, tgt_v = _tiny_vocab()
        device = torch.device("cpu")
        with pytest.raises(ValueError, match="not implemented"):
            beam_search_decode(
                _CountingTransformer(tgt_v), "ab", src_v, tgt_v, device, beam_size=2
            )

    def test_beam_fallback_does_not_raise(self) -> None:
        src_v, tgt_v = _tiny_vocab()
        device = torch.device("cpu")
        model = _CountingTransformer(tgt_v)
        effective, fell_back = resolve_beam_size(model, 4)
        assert (effective, fell_back) == (1, True)
        text, fell_back2 = decode_with_fallback(
            model, "ab", src_v, tgt_v, device, beam_size=4
        )
        assert fell_back2 is True
        assert isinstance(text, str)

    def test_beam_no_fallback_for_rnn(self) -> None:
        src_v, tgt_v = _tiny_vocab()
        device = torch.device("cpu")
        enc = EncoderRNN(len(src_v), 8, 16, num_layers=1, cell_type="gru")
        dec = DecoderRNN(len(tgt_v), 8, 16, num_layers=1, cell_type="gru")
        model = Seq2Seq(enc, dec, device)
        effective, fell_back = resolve_beam_size(model, 4)
        assert (effective, fell_back) == (4, False)
        text, fell_back2 = decode_with_fallback(
            model, "ab", src_v, tgt_v, device, beam_size=2
        )
        assert fell_back2 is False
        assert isinstance(text, str)
