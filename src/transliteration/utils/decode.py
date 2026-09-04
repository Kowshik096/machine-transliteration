"""Decoding utilities: greedy and beam search."""


import torch

from transliteration.data.vocab import Vocab


def _is_transformer(model: torch.nn.Module) -> bool:
    """Return True for the Transformer architecture (different forward signature).

    Detected via duck typing: only TransformerSeq2Seq exposes a
    ``transformer`` (nn.Transformer) submodule.
    """
    return hasattr(model, "transformer")


def _make_tgt_dummy(tgt_vocab: Vocab, max_len: int, device: torch.device) -> torch.Tensor:
    return torch.LongTensor(
        [[tgt_vocab.sos_idx] + [tgt_vocab.pad_idx] * (max_len - 1)]
    ).to(device)


def _greedy_decode_transformer(
    model: torch.nn.Module,
    src_tensor: torch.Tensor,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int,
) -> str:
    """Genuine autoregressive greedy decoding for the Transformer.

    1. Start the target prefix with ``<SOS>``.
    2. Forward ``model(src, prefix)`` and take the last-step argmax.
    3. Append the predicted token to the prefix.
    4. Repeat until ``<EOS>`` or ``max_len`` tokens are generated.
    """
    generated = [tgt_vocab.sos_idx]
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor([generated]).to(device)
        logits = model(src_tensor, tgt_tensor)
        next_id = int(logits[0, -1].argmax().item())
        if next_id == tgt_vocab.eos_idx:
            break
        generated.append(next_id)
        # generated includes SOS; stop once we reach max_len total ids.
        if len(generated) >= max_len:
            break
    res = []
    for id_ in generated:
        if id_ == tgt_vocab.eos_idx:
            break
        if id_ in (tgt_vocab.sos_idx, tgt_vocab.pad_idx):
            continue
        res.append(tgt_vocab.idx2char.get(id_, "?"))
    return "".join(res)


def resolve_beam_size(model: torch.nn.Module, beam_size: int) -> tuple[int, bool]:
    """Return ``(effective_beam_size, fell_back)`` for a model.

    The Transformer has no beam-search implementation, so any
    ``beam_size > 1`` falls back to greedy (beam 1).  Callers should
    surface ``fell_back`` to the user instead of raising.
    """
    if _is_transformer(model) and beam_size > 1:
        return 1, True
    return beam_size, False


def decode_with_fallback(
    model: torch.nn.Module,
    src_text: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    beam_size: int = 1,
    max_len: int = 64,
) -> tuple[str, bool]:
    """Decode, falling back to greedy for the Transformer.

    Returns ``(text, fell_back)`` where ``fell_back`` is True when a
    requested ``beam_size > 1`` was ignored for Transformer compatibility.
    """
    effective, fell_back = resolve_beam_size(model, beam_size)
    if effective > 1:
        return (
            beam_search_decode(
                model, src_text, src_vocab, tgt_vocab, device, effective, max_len
            ),
            fell_back,
        )
    return (
        greedy_decode(model, src_text, src_vocab, tgt_vocab, device, max_len),
        fell_back,
    )


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
    if _is_transformer(model):
        # Genuine autoregressive decoding: feed each predicted token back.
        return _greedy_decode_transformer(model, src_tensor, tgt_vocab, device, max_len)
    if hasattr(model, "forward") and "src_lens" in model.forward.__code__.co_varnames:
        # RNN-based models with src_lens
        tgt_dummy = _make_tgt_dummy(tgt_vocab, max_len, device)
        outputs = model(src_tensor, tgt_dummy, src_lens, teacher_forcing_ratio=0.0)
    else:
        raise ValueError(f"Unsupported model forward signature for {type(model).__name__}")

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

    if _is_transformer(model):
        raise ValueError(
            "Beam search is not implemented for the Transformer architecture. "
            "Use beam_size=1 (greedy_decode) instead."
        )

    if not hasattr(model, "encoder") or not hasattr(model, "decoder"):
        raise ValueError("Beam search only implemented for RNN-based models with encoder/decoder")

    src_ids = src_vocab.encode(src_text, add_special=False, max_len=max_len)
    src_tensor = torch.LongTensor([src_ids]).to(device)
    src_lens = torch.LongTensor([len(src_ids)]).to(device)
    src_mask = (src_tensor == src_vocab.pad_idx)

    enc_outputs, enc_hidden = model.encoder(src_tensor, src_lens)  # type: ignore[operator]

    # Handle different method names for initializing decoder hidden state
    if hasattr(model, "_init_decoder_hidden"):
        dec_hidden = model._init_decoder_hidden(enc_hidden)  # type: ignore[operator]
    elif hasattr(model, "_init_decoder_hidden_from_encoder"):
        dec_hidden = model._init_decoder_hidden_from_encoder(enc_hidden)  # type: ignore[operator]
    else:
        raise AttributeError(
            "Model must have _init_decoder_hidden or _init_decoder_hidden_from_encoder"
        )

    # Determine if decoder expects attention context (has attention module)
    has_attention = hasattr(model.decoder, "attention")  # type: ignore[operator]

    beam: list = [(0.0, [tgt_vocab.sos_idx], dec_hidden)]
    completed: list = []

    for _ in range(max_len):
        candidates = []
        for score, tokens, hidden in beam:
            if tokens[-1] == tgt_vocab.eos_idx:
                completed.append((score, tokens))
                continue
            inp = torch.LongTensor([tokens[-1]]).to(device)
            if has_attention:
                decoder_out = model.decoder(inp, hidden, enc_outputs, src_mask)  # type: ignore[operator]
                logits, new_hidden, _ = decoder_out
            else:
                logits, new_hidden = model.decoder(inp, hidden)  # type: ignore[operator]
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

            pred, _ = decode_with_fallback(
                model, src_text, src_vocab, tgt_vocab, device, beam_size
            )

            preds.append(pred)
            targets.append(tgt_text)

    return compute_all_metrics(preds, targets)
