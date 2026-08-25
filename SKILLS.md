# Skills & Interview Preparation Guide

This document covers the deep learning concepts, engineering practices, and project-specific questions you should master to confidently discuss this transliteration benchmarking project in interviews.

---

## Project Overview

```text
1. RNN (LSTM/GRU/SimpleRNN) without attention  → baseline seq2seq
2. RNN + Bahdanau/additive attention           → does explicit alignment help?
3. Transformer                                 → how much does removing recurrence help?
```

All three models run on the same **Dakshina Hindi** dataset with identical train/dev/test splits and evaluation metrics.

---

## Core Deep Learning Concepts

### Seq2seq / Encoder-Decoder

- Maps variable-length input to variable-length output.
- Encoder compresses input into a fixed-size context vector; decoder generates output from it.
- The fixed-size bottleneck is the key limitation that motivated attention.

**Likely questions:**
- Why is a single context vector a problem for long sequences?
- What happens if input and output vocabularies are different?

### Teacher Forcing

- During training, feed ground-truth token as decoder input instead of model's own prediction.
- Speeds convergence but creates **exposure bias**: at inference, the model never sees its own mistakes during training.

**Likely questions:**
- What is exposure bias? How can you mitigate it?
- What happens with `teacher_forcing_ratio = 0.0` during training?

### Additive (Bahdanau) Attention

Score function:

```
score(h_t, e_i) = v^T · tanh(W_enc · e_i + W_dec · h_t)
```

- The decoder's current hidden state (`h_t`) is compared against every encoder output (`e_i`).
- Softmax over scores gives alignment weights.
- A weighted sum of encoder outputs forms the context vector.

**Key properties:**
- Sequential — one attention computation per output step.
- Cross-attention: decoder attending to encoder outputs.
- Fixes the information bottleneck of vanilla seq2seq.

**Likely questions:**
- Derive or explain the additive score function.
- Why use tanh before the dot product with v?
- How do you visualize attention weights?

### Self-Attention (Transformer)

Score function (scaled dot-product):

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

- Every token position produces Query, Key, Value projections.
- Each token attends to all other tokens simultaneously (parallel computation).
- No RNN needed — stacked self-attention layers replace recurrence.

**Types of attention inside a Transformer:**
1. **Encoder self-attention** — each input char attends to other input chars.
2. **Masked decoder self-attention** — each target char attends only to previous target chars (prevents looking ahead).
3. **Cross-attention** — decoder attends to encoder outputs (like Bahdanau but computed differently).

**Likely questions:**
- Explain Q, K, V intuitively. What does "query" mean here?
- Why divide by √d_k?
- Why mask future positions in the decoder?
- What is multi-head attention and why is it better than one large head?

### Positional Encoding

- Transformers have no inherent notion of order (unlike RNNs).
- Sinusoidal positional encodings are added to embeddings:
  ```
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  ```

**Likely questions:**
- Why do Transformers need positional encoding but RNNs don't?
- What are learned vs sinusoidal encodings? Trade-offs?

### Beam Search vs Greedy Decoding

| | Greedy | Beam Search |
|---|---|---|
| Picks | Best token at each step | Top-k sequences tracked simultaneously |
| Speed | Fast | Slower (k× more work) |
| Quality | Can miss better overall sequences | Often finds higher-probability sequences |
| Risk | Local optimum | Can still fail on long sequences |

**Likely questions:**
- When would beam search produce worse results than greedy?
- How does beam width affect latency and quality?

### Loss Functions

- **CrossEntropyLoss** with `ignore_index=PAD` — skips padded positions so they don't contribute to loss/gradient.
- Character-level models have small vocab (~70–80), making training stable.

**Likely questions:**
- Why set `ignore_index` instead of manually masking?
- How would you handle class imbalance if some characters are very rare?

### Gradient Clipping

- RNNs are prone to exploding gradients due to repeated multiplication through timesteps.
- `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` caps global gradient norm.

**Likely questions:**
- Why are RNNs more susceptible to exploding gradients than feedforward networks?
- What's the difference between gradient clipping by value vs by norm?

---

## Architecture Comparison Cheat Sheet

| | RNN alone | RNN + Bahdanau | Transformer |
|---|---|---|---|
| Computation | Sequential | Sequential | Parallel |
| Attention type | None | Cross-attention per step | Self + masked self + cross |
| Long-range deps | Limited by gradient flow | Better via direct access to all encoder states | Direct connection between any two positions |
| Positional info | Implicit via RNN order | Implicit via RNN order | Explicit positional encodings |
| Speed per epoch | Slow | Slowest (attention adds overhead) | Fastest on GPU |
| Parameters (typical) | ~2–5M | ~5–10M | ~1–3M (small config) |

---

## Engineering Skills

### Version Control
- Clean commit messages.
- Feature branches for experiments.
- `.gitignore` for checkpoints, data, and notebook outputs.

### Experiment Tracking
- W&B (already used in baseline notebook).
- Log hyperparameters, metrics, and artifacts consistently.

### Configuration Management
- Move hardcoded values into YAML configs or CLI arguments.
- Makes experiments reproducible.

### Testing
- Unit-test preprocessing, vocabulary building, padding, and metric computation.
- Use tiny synthetic datasets for fast iteration.

### Code Organization
- Refactor notebooks into `src/` modules when ready.
- Separate data loading, model definitions, training loops, and evaluation.

---

## Project-Specific Interview Questions

Be prepared to answer these about YOUR implementation:

### Data & Preprocessing
1. Why character-level rather than subword (BPE)?
2. How did you handle `<UNK>` characters at test time?
3. Why pad sequences? What's your max length and why?
4. How did you build vocabularies — from train only or train + dev?

### Model Design
5. What embedding size did you choose and why?
6. How many layers in encoder vs decoder?
7. Why bidirectional encoder but unidirectional decoder?
8. How do you initialize decoder hidden state from a bidirectional encoder?

### Training
9. What teacher forcing ratio did you use and why?
10. How did you prevent overfitting?
11. What optimizer and learning rate? Did you tune them?
12. How long did training take per epoch?

### Evaluation
13. Report your word accuracy, character accuracy, and BLEU.
14. How does character accuracy differ conceptually from word accuracy?
15. Show examples where the model fails. Can you categorize errors?
16. How would you improve results with more time/compute?

### Attention Analysis
17. Can you visualize what the attention layer learned?
18. Do attention weights align sensibly with expected character correspondences?
19. How does attention behave on longer words?

### Transformer-Specific
20. Why is your Transformer faster/slower per epoch than your LSTM?
21. How many parameters does each model have?
22. If you had 10× more data, which architecture would benefit most?

---

## Recommended Learning Resources

### Conceptual Understanding
- **The Illustrated Transformer** — Jay Alammar (best visual explanation of Q/K/V)
- **The Illustrated Seq2Seq** — Jay Alammar
- *Attention Is All You Need* (Vaswani et al., 2017) — read multiple times
- Stanford CS224N — Lecture slides and videos on NMT and attention

### Practical Implementation
- PyTorch official seq2seq tutorial
- TorchText sentiment/translation tutorials
- LabML.ai annotated Transformer implementation

### Interview Prep General
- *Deep Learning Interviews* (Shlomo Kashani)
- LeetCode ML tag problems
- Glassdoor interview questions for ML Engineer roles

---

## Resume Bullet Suggestions

After running experiments and filling in real numbers, frame your work as:

> "Benchmarked character-level seq2seq architectures (LSTM, GRU, Bahdanau attention, Transformer) for Hindi transliteration on the Dakshina dataset, achieving X% word accuracy with beam-search decoding."

Or for engineering focus:

> "Built reproducible PyTorch training pipelines with configurable data paths, experiment tracking via W&B, and automated evaluation of word accuracy, character accuracy, and BLEU across three architecture families."
