#!/usr/bin/env python3
"""Evaluation script for transliteration models."""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transliteration.data.vocab import build_vocab, Vocab
from transliteration.data.dataset import TransliterationDataset, collate_fn
from transliteration.models.seq2seq import Seq2Seq, EncoderRNN, DecoderRNN
from transliteration.models.attention import AttentionSeq2Seq, Encoder, Decoder
from transliteration.models.transformer import TransformerSeq2Seq
from transliteration.utils.metrics import compute_all_metrics
from transliteration.utils.decode import greedy_decode, beam_search_decode


def load_pairs_from_tsv(data_dir: Path, lang: str = "te") -> Tuple[List, List, List]:
    splits = {"train": [], "dev": [], "test": []}
    for split in ["train", "dev", "test"]:
        fname = data_dir / f"{lang}.translit.sampled.{split}.tsv"
        if not fname.exists():
            raise FileNotFoundError(f"Dataset file not found: {fname}")
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    native, roman = parts[0].strip(), parts[1].strip()
                    if roman and native:
                        splits[split].append((roman, native))
    return splits["train"], splits["dev"], splits["test"]


def build_model(args, src_vocab: Vocab, tgt_vocab: Vocab, device: torch.device):
    if args.arch == "rnn":
        encoder = EncoderRNN(
            len(src_vocab), args.emb_dim, args.hidden_dim,
            num_layers=args.num_layers, cell_type=args.cell_type,
            bidirectional=args.bidirectional, dropout=args.dropout,
        )
        decoder = DecoderRNN(
            len(tgt_vocab), args.emb_dim, args.hidden_dim,
            num_layers=args.num_layers, cell_type=args.cell_type,
            dropout=args.dropout,
        )
        model = Seq2Seq(encoder, decoder, device)
    elif args.arch == "attention":
        encoder = Encoder(
            len(src_vocab), args.emb_dim, args.hidden_dim,
            num_layers=args.num_layers, cell_type=args.cell_type,
            bidirectional=True, dropout=args.dropout,
        )
        decoder = Decoder(
            len(tgt_vocab), args.emb_dim, args.hidden_dim,
            enc_dim=args.hidden_dim, num_layers=args.num_layers,
            cell_type=args.cell_type, dropout=args.dropout,
        )
        model = AttentionSeq2Seq(encoder, decoder, src_pad_idx=src_vocab.pad_idx)
    elif args.arch == "transformer":
        model = TransformerSeq2Seq(
            len(src_vocab), len(tgt_vocab),
            d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers, dim_ff=args.dim_ff,
            dropout=args.dropout, pad_idx=src_vocab.pad_idx,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate transliteration model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory with TSV files")
    parser.add_argument("--lang", type=str, default="te", choices=["te", "hi"], help="Language code")
    parser.add_argument("--arch", type=str, default="attention", choices=["rnn", "attention", "transformer"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dim-ff", type=int, default=256)
    parser.add_argument("--cell-type", type=str, default="lstm", choices=["lstm", "gru", "rnn"])
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beam-size", type=int, default=4, help="Beam size (1=greedy)")
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

    # Build model and load checkpoint
    model = build_model(args, src_vocab, tgt_vocab, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    if args.interactive:
        print("\nInteractive mode - type Roman text to transliterate (Ctrl+C to exit)")
        while True:
            try:
                src = input("\n> ").strip()
                if not src:
                    continue
                if args.beam_size > 1:
                    pred = beam_search_decode(model, src, src_vocab, tgt_vocab, device, args.beam_size)
                else:
                    pred = greedy_decode(model, src, src_vocab, tgt_vocab, device)
                print(f"  → {pred}")
            except (KeyboardInterrupt, EOFError):
                break
        return

    # Batch evaluation on test set
    test_dataset = TransliterationDataset(test_pairs, src_vocab, tgt_vocab)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, pad_idx=src_vocab.pad_idx))

    preds, targets = [], []
    for src_batch, tgt_batch, src_lens in tqdm(test_loader, desc="Evaluating"):
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        src_lens = src_lens.to(device)

        for i in range(src_batch.size(0)):
            src_ids = src_batch[i].tolist()
            src_text = src_vocab.decode(src_ids)
            tgt_ids = tgt_batch[i].tolist()
            tgt_text = tgt_vocab.decode(tgt_ids)

            if args.beam_size > 1:
                pred = beam_search_decode(model, src_text, src_vocab, tgt_vocab, device, args.beam_size)
            else:
                pred = greedy_decode(model, src_text, src_vocab, tgt_vocab, device)

            preds.append(pred)
            targets.append(tgt_text)

    metrics = compute_all_metrics(preds, targets)
    print(f"\nTest Metrics:")
    print(f"  Word Accuracy:  {metrics['word_accuracy']:.4f} ({metrics['word_accuracy']*100:.2f}%)")
    print(f"  Char Accuracy:  {metrics['char_accuracy']:.4f} ({metrics['char_accuracy']*100:.2f}%)")
    print(f"  BLEU:           {metrics['bleu']:.4f}")

    # Show some examples
    print("\nSample predictions:")
    for i in range(min(10, len(preds))):
        status = "✓" if preds[i] == targets[i] else "✗"
        print(f"  {status} {preds[i]:20s} | {targets[i]}")


if __name__ == "__main__":
    main()