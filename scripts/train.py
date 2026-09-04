#!/usr/bin/env python3
"""Training script for transliteration models."""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transliteration.data.dataset import TransliterationDataset, collate_fn
from transliteration.data.vocab import Vocab, build_vocab
from transliteration.models.attention import AttentionSeq2Seq, Decoder, Encoder
from transliteration.models.presets import ARCH_PRESETS, get_preset
from transliteration.models.seq2seq import DecoderRNN, EncoderRNN, Seq2Seq
from transliteration.models.transformer import TransformerSeq2Seq
from transliteration.utils.checkpoints import safe_torch_load


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pairs_from_tsv(data_dir: Path, lang: str = "te") -> tuple[list, list, list]:
    """Load train/dev/test pairs from TSV files."""
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
    """Build model based on architecture, starting from explicit presets."""
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


def train_epoch(model, loader, optimizer, criterion, device, teacher_forcing, clip=1.0):
    model.train()
    total_loss = 0.0
    for src, tgt, src_lens in tqdm(loader, desc="Training", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        src_lens = src_lens.to(device)
        optimizer.zero_grad()
        if isinstance(model, TransformerSeq2Seq):
            output = model(src, tgt)
        else:
            output = model(src, tgt, src_lens, teacher_forcing_ratio=teacher_forcing)
        loss = criterion(
            output[:, 1:].reshape(-1, output.size(-1)),
            tgt[:, 1:].reshape(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, teacher_forcing):
    model.eval()
    total_loss = 0.0
    for src, tgt, src_lens in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_lens = src_lens.to(device)
        if isinstance(model, TransformerSeq2Seq):
            output = model(src, tgt)
        else:
            output = model(src, tgt, src_lens, teacher_forcing_ratio=teacher_forcing)
        loss = criterion(
            output[:, 1:].reshape(-1, output.size(-1)),
            tgt[:, 1:].reshape(-1),
        )
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train transliteration model")
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
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
    parser.add_argument("--teacher-forcing", type=float, default=0.5)
    parser.add_argument("--save-path", type=str, default="checkpoints/best.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {args.arch}")

    # Load data
    data_dir = Path(args.data_dir)
    train_pairs, dev_pairs, test_pairs = load_pairs_from_tsv(data_dir, args.lang)
    print(f"Train: {len(train_pairs)}  Dev: {len(dev_pairs)}  Test: {len(test_pairs)}")

    # Build vocab
    src_vocab, tgt_vocab = build_vocab(train_pairs + dev_pairs)
    print(f"Source vocab: {len(src_vocab)}  Target vocab: {len(tgt_vocab)}")

    # Create loaders
    train_dataset = TransliterationDataset(train_pairs, src_vocab, tgt_vocab)
    dev_dataset = TransliterationDataset(dev_pairs, src_vocab, tgt_vocab)
    test_dataset = TransliterationDataset(test_pairs, src_vocab, tgt_vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=src_vocab.pad_idx),
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx=src_vocab.pad_idx),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx=src_vocab.pad_idx),
    )

    # Build model
    model = build_model(args, src_vocab, tgt_vocab, device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_idx)

    # Training loop
    best_val_loss = float("inf")
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, args.teacher_forcing
        )
        val_loss = evaluate(model, dev_loader, criterion, device, 0.0)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  Saved best model to {args.save_path}")

    # Final evaluation on test
    model.load_state_dict(safe_torch_load(args.save_path, map_location=device))
    test_loss = evaluate(model, test_loader, criterion, device, 0.0)
    print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
