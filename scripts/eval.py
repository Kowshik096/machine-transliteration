#!/usr/bin/env python3
"""Evaluation script for transliteration models."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Ensure non-ASCII output (e.g. Telugu script) is not mangled on cp1252 consoles.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    except Exception:
        pass

from transliteration.data.dataset import TransliterationDataset, collate_fn
from transliteration.data.vocab import Vocab, build_vocab
from transliteration.models.attention import AttentionSeq2Seq, Decoder, Encoder
from transliteration.models.presets import ARCH_PRESETS, get_preset
from transliteration.models.seq2seq import DecoderRNN, EncoderRNN, Seq2Seq
from transliteration.models.transformer import TransformerSeq2Seq
from transliteration.utils.checkpoints import load_model_state_dict
from transliteration.utils.decode import decode_with_fallback
from transliteration.utils.metrics import compute_all_metrics


def load_pairs_from_tsv(data_dir: Path, lang: str = "te") -> tuple[list, list, list]:
    splits: dict[str, list] = {"train": [], "dev": [], "test": []}
    for split in ["train", "dev", "test"]:
        fname = data_dir / f"{lang}.translit.sampled.{split}.tsv"
        if not fname.exists():
            raise FileNotFoundError(f"Dataset file not found: {fname}")
        with open(fname, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    native, roman = parts[0].strip(), parts[1].strip()
                    if roman and native:
                        splits[split].append((roman, native))
    return splits["train"], splits["dev"], splits["test"]


def build_model(args, src_vocab: Vocab, tgt_vocab: Vocab, device: torch.device):
    preset = get_preset(args.arch)
    emb_dim = args.emb_dim if args.emb_dim is not None else preset.get("emb_dim", 64)
    hidden_dim = (
        args.hidden_dim if args.hidden_dim is not None else preset.get("hidden_dim", 256)
    )
    d_model = args.d_model if args.d_model is not None else preset.get("d_model", 128)
    nhead = args.nhead if args.nhead is not None else preset.get("nhead", 4)
    num_layers = (
        args.num_layers if args.num_layers is not None else preset.get("num_layers", 1)
    )
    dim_ff = args.dim_ff if args.dim_ff is not None else preset.get("dim_ff", 256)
    cell_type = (
        args.cell_type if args.cell_type is not None else preset.get("cell_type", "lstm")
    )
    bidirectional = (
        args.bidirectional
        if args.bidirectional is not None
        else preset.get("bidirectional", True)
    )
    dropout = args.dropout if args.dropout is not None else preset.get("dropout", 0.1)
    if args.arch == "rnn":
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
        return Seq2Seq(rnn_encoder, rnn_decoder, device).to(device)
    if args.arch == "attention":
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
        return AttentionSeq2Seq(
            attn_encoder, attn_decoder, src_pad_idx=src_vocab.pad_idx
        ).to(device)
    if args.arch == "transformer":
        return TransformerSeq2Seq(
            len(src_vocab), len(tgt_vocab),
            d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_ff=dim_ff,
            dropout=dropout, pad_idx=src_vocab.pad_idx,
        ).to(device)
    raise ValueError(
        f"Unknown architecture: {args.arch}. Expected one of {sorted(ARCH_PRESETS)}."
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate transliteration model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Data directory with TSV files"
    )
    parser.add_argument(
        "--lang", type=str, default="te", choices=["te", "hi"], help="Language code"
    )
    parser.add_argument(
        "--arch", type=str, default="attention",
        choices=["rnn", "attention", "transformer"],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    # Arch-specific hyperparameters default to None so per-architecture
    # presets (see transliteration.models.presets) fill them in.
    parser.add_argument("--emb-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dim-ff", type=int, default=None)
    parser.add_argument(
        "--cell-type", type=str, default=None, choices=["lstm", "gru", "rnn"]
    )
    parser.add_argument(
        "--bidirectional", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--beam-size", type=int, default=4, help="Beam size (1=greedy)"
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data_dir = Path(args.data_dir)
    train_pairs, dev_pairs, test_pairs = load_pairs_from_tsv(data_dir, args.lang)
    print(f"Train: {len(train_pairs)}  Dev: {len(dev_pairs)}  Test: {len(test_pairs)}")

    # Build vocab (must match training)
    src_vocab, tgt_vocab = build_vocab(train_pairs + dev_pairs)
    print(f"Source vocab: {len(src_vocab)}  Target vocab: {len(tgt_vocab)}")

    # Build model and load checkpoint (raw or Lightning envelope).
    model = build_model(args, src_vocab, tgt_vocab, device)
    load_model_state_dict(model, args.checkpoint, map_location=device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    if args.interactive:
        print("\nInteractive mode - type Roman text to transliterate (Ctrl+C to exit)")
        while True:
            try:
                src = input("\n> ").strip()
                if not src:
                    continue
                pred, fell_back = decode_with_fallback(
                    model, src, src_vocab, tgt_vocab, device, args.beam_size
                )
                if fell_back:
                    print("  (beam search unavailable for Transformer; used greedy)")
                print(f"  → {pred}")
            except (KeyboardInterrupt, EOFError):
                break
        return

    # Batch evaluation on test set
    test_dataset = TransliterationDataset(test_pairs, src_vocab, tgt_vocab)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx=src_vocab.pad_idx),
    )

    preds, targets = [], []
    for src_batch, tgt_batch, src_lens in tqdm(test_loader, desc="Evaluating"):
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        src_lens = src_lens.to(device)

        for i in range(src_batch.size(0)):
            src_ids = src_batch[i].tolist()
            src_text = src_vocab.decode(src_ids)
            tgt_ids = tgt_batch[i].tolist()
            tgt_text = tgt_vocab.decode(tgt_ids)

            pred, _ = decode_with_fallback(
                model, src_text, src_vocab, tgt_vocab, device, args.beam_size
            )

            preds.append(pred)
            targets.append(tgt_text)

    metrics = compute_all_metrics(preds, targets)
    print("\nTest Metrics:")
    print(f"  Word Accuracy:  {metrics['word_accuracy']:.4f} ({metrics['word_accuracy']*100:.2f}%)")
    print(f"  Char Accuracy:  {metrics['char_accuracy']:.4f} ({metrics['char_accuracy']*100:.2f}%)")
    print(f"  BLEU:           {metrics['bleu']:.4f}")

    # Show some examples
    print("\nSample predictions:")
    for i in range(min(10, len(preds))):
        status = "ok" if preds[i] == targets[i] else "XX"
        pred_str = preds[i][:20]
        target_str = targets[i][:20]
        print(f"  [{status}] {pred_str:20s} | {target_str}")


if __name__ == "__main__":
    main()
