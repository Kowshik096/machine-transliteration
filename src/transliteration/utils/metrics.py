"""Evaluation metrics for transliteration."""


from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def word_accuracy(preds: list[str], targets: list[str]) -> float:
    """Exact word match accuracy."""
    if not preds:
        return 0.0
    correct = sum(p == t for p, t in zip(preds, targets))
    return correct / len(preds)


def char_accuracy(preds: list[str], targets: list[str]) -> float:
    """Character-level accuracy (position-wise up to min length)."""
    if not preds:
        return 0.0
    total_correct = 0
    total_chars = 0
    for p, t in zip(preds, targets):
        min_len = min(len(p), len(t))
        total_correct += sum(p[i] == t[i] for i in range(min_len))
        total_chars += len(t)
    return total_correct / total_chars if total_chars > 0 else 0.0


def bleu_score(preds: list[str], targets: list[str]) -> float:
    """Sentence-level BLEU with smoothing on character n-grams."""
    if not preds:
        return 0.0
    smooth = SmoothingFunction().method1
    scores = []
    for p, t in zip(preds, targets):
        if len(t) == 0:
            continue
        scores.append(sentence_bleu([list(t)], list(p), smoothing_function=smooth))
    return sum(scores) / len(scores) if scores else 0.0


def compute_all_metrics(
    preds: list[str], targets: list[str]
) -> dict:
    """Compute all metrics at once."""
    return {
        "word_accuracy": word_accuracy(preds, targets),
        "char_accuracy": char_accuracy(preds, targets),
        "bleu": bleu_score(preds, targets),
    }
