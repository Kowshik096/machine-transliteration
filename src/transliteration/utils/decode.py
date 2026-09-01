"""Decoding utilities: greedy and beam search."""

import heapq
from typing import List, Tuple, Union

import torch

from transliteration.data.vocab import Vocab


@torch.no_grad()
def greedy_decode(
    model: torch.nn.Module,
    src_text: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int = 64,
    src_pad_idx: int = 0,
) -> str:
    """Greedy decoding for any seq2seq model."""
    model.eval()
    src_ids = src_vocab.encode(src_text, add_special=False, max_len=max_len)
    src_tensor = torch.LongTensor([src_ids]).to(device)
    src_lens = torch.LongTensor([len(src_ids)]).to(device)

    # Handle different model forward signatures
    if hasattr(model, "forward") and "src_lens" in model.forward.__code__.co_varnames:
        # RNN-based models with src_lens
        tgt_dummy = torch.LongTensor([[tgt_vocab.sos_idx] + [tgt_vocab.pad_idx] * (max_len - 1)]).to(device)
        outputs = model(src_tensor, tgt_dummy, src_lens, teacher_forcing_ratio=0.0)
    else:
        # Transformer
        tgt_dummy = torch.LongTensor([[tgt_vocab.sos_idx] + [tgt_vocab.pad_idx] * (max_len - 1)]).to(device)
        outputs = model(src_tensor, tgt_dummy)

    ids = outputs.argmax(-1).squeeze(0).tolist()
    res = []
    for id_ in ids:
        if id_ == tgt_vocab.eos_idx:
            break
        if id_ in (tgt_vocab.sos_idx, tgt_vocab.pad_idx):
            continue
        res.append(tgt_vocab.idx2char.get(id_, "?"))
    return "".join(res)


@torch.no_grad()
def beam_search_decode(
    model: torch.nn.Module,
    src_text: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    beam_size: int = 4,
    max_len: int = 64,
) -> str:
    """Beam search decoding for RNN-based seq2seq models.

    Works with both Seq2Seq (no attention) and AttentionSeq2Seq (Bahdanau attention).
    """
    model.eval()

    if not hasattr(model, "encoder") or not hasattr(model, "decoder"):
        raise ValueError("Beam search only implemented for RNN-based models with encoder/decoder")

    src_ids = src_vocab.encode(src_text, add_special=False, max_len=max_len)
    src_tensor = torch.LongTensor([src_ids]).to(device)
    src_lens = torch.LongTensor([len(src_ids)]).to(device)
    src_mask = (src_tensor == src_vocab.pad_idx)

    enc_outputs, enc_hidden = model.encoder(src_tensor, src_lens)

    # Handle different method names for initializing decoder hidden state
    if hasattr(model, "_init_decoder_hidden"):
        dec_hidden = model._init_decoder_hidden(enc_hidden)
    elif hasattr(model, "_init_decoder_hidden_from_encoder"):
        dec_hidden = model._init_decoder_hidden_from_encoder(enc_hidden)
    else:
        raise AttributeError(
            "Model must have _init_decoder_hidden or _init_decoder_hidden_from_encoder"
        )

    # Determine if decoder expects attention context (has attention module)
    has_attention = hasattr(model.decoder, "attention")

    beam = [(0.0, [tgt_vocab.sos_idx], dec_hidden)]
    completed = []

    for _ in range(max_len):
        candidates = []
        for score, tokens, hidden in beam:
            if tokens[-1] == tgt_vocab.eos_idx:
                completed.append((score, tokens))
                continue
            inp = torch.LongTensor([tokens[-1]]).to(device)
            if has_attention:
                decoder_out = model.decoder(inp, hidden, enc_outputs, src_mask)
                logits, new_hidden, _ = decoder_out
            else:
                logits, new_hidden = model.decoder(inp, hidden)
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            topk_logp, topk_idx = torch.topk(log_probs, beam_size)
            for k in range(beam_size):
                new_score = score + topk_logp[k].item()
                new_tokens = tokens + [topk_idx[k].item()]
                if isinstance(new_hidden, tuple):
                    new_h = (new_hidden[0].clone(), new_hidden[1].clone())
                else:
                    new_h = new_hidden.clone()
                candidates.append((new_score, new_tokens, new_h))

        beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        if all(tokens[-1] == tgt_vocab.eos_idx for _, tokens, _ in beam):
            completed.extend([(s, t) for s, t, _ in beam])
            break

    if not completed:
        completed = [(s, t) for s, t, _ in beam]

    best_tokens = max(completed, key=lambda x: x[0])[1]
    return tgt_vocab.decode(best_tokens)


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    loader,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    beam_size: int = 1,
) -> dict:
    """Evaluate model on a DataLoader."""
    from transliteration.utils.metrics import compute_all_metrics

    model.eval()
    preds, targets = [], []

    for src_batch, tgt_batch, src_lens in loader:
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
        src_lens = src_lens.to(device)

        for i in range(src_batch.size(0)):
            src_ids = src_batch[i].tolist()
            src_text = src_vocab.decode(src_ids)
            tgt_ids = tgt_batch[i].tolist()
            tgt_text = tgt_vocab.decode(tgt_ids)

            if beam_size > 1:
                pred = beam_search_decode(model, src_text, src_vocab, tgt_vocab, device, beam_size)
            else:
                pred = greedy_decode(model, src_text, src_vocab, tgt_vocab, device)

            preds.append(pred)
            targets.append(tgt_text)

    return compute_all_metrics(preds, targets)