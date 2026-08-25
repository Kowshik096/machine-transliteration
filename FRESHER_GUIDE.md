# Machine Transliteration — Fresher's Guide

> You have zero experience with this project? Start here. By the end of this guide you can build, train, and explain all three models in an interview.

---

## 1. What This Project Does (in Plain English)

**Input:** romanized Hindi word typed in English letters — e.g. `namaste`
**Output:** the same word in Devanagari script — `नमस्ते`

This is called **transliteration** (not translation). The meaning stays the same, only the writing system changes. We treat it as a **character-level sequence-to-sequence** problem: the model reads one character at a time and writes one character at a time.

**Dataset:** Dakshina Hindi — 51,200 train / 4,096 dev / 4,096 test word pairs. Each TSV line is `<Devanagari>\t<Roman>\t<1>`. The `data/` folder is built from `aksharantar_sampled.zip` via `convert_aksharantar.py`.

---

## 2. Skills You Need (Grouped by Priority)

### A. Programming (must-have)
| Skill | Why it matters here | How to learn it |
|---|---|---|
| **Python** — lists, dicts, file I/O, classes | Everything is Python | Python docs + any beginner course |
| **NumPy** — arrays, basic ops | Vocabulary building, data handling | `numpy` quickstart |
| **PyTorch** — `nn.Module`, `Embedding`, `RNN/LSTM/GRU`, `DataLoader` | All three models are PyTorch | PyTorch official 60-min blitz tutorial |

### B. Machine Learning Fundamentals
- Train / dev / test splits, overfitting, loss curves
- Optimizers (Adam), learning rate, weight decay, dropout
- Checkpointing (save best model by validation score)
- Reproducibility — fixing random seeds (`random`, `numpy`, `torch`)

### C. Deep Learning / NLP Core
- **Seq2Seq / Encoder-Decoder** — encoder compresses input, decoder generates output step-by-step
- **Teacher forcing** — feed the true next character during training (faster but causes *exposure bias*)
- **Padding & masking** — `PAD` tokens to batch variable-length words, `ignore_index=PAD` in loss
- **CrossEntropyLoss** — per-character classification

### D. Attention & Transformers (project-specific)
- **Bahdanau (additive) attention:** `score(h_t, e_i) = vᵀ tanh(W_enc·e_i + W_dec·h_t)` — decoder attends over all encoder outputs each step, fixing the fixed-size bottleneck
- **Transformer:** self-attention, cross-attention, positional encodings, multi-head — no recurrence, fully parallel
- **Greedy vs beam search** — best-token-per-step vs tracking top-k sequences
- **BLEU, word accuracy, character accuracy** — the three evaluation metrics used here

### E. Tools (nice to have)
- **Jupyter / Kaggle** — interactive notebook workflow (GPU T4)
- **Weights & Biases (W&B)** — experiment tracking, sweep-based hyperparameter search
- **Git** — version control, `.gitignore` for `data/`, `*.pt`, `*.zip`

---

## 3. Roadmap — Zero to Finished (in Order)

### Phase 0 — Environment (30 min)
```bash
pip install -r requirements.txt          # torch, wandb, nltk, tqdm
# + pytorch-lightning (installed inside the baseline notebook)
python convert_aksharantar.py --zip aksharantar_sampled.zip --outdir data
```

### Phase 1 — Read the Code (1–2 days)
1. Open `seq2seq_pytorch_attention.ipynb` — smallest clean example (13 cells, no sweep)
2. `seq2seq_transformer.ipynb` — see how `nn.Transformer` replaces the RNNs
3. `seq2seq_pytorch_without_attention.ipynb` — full sweep + Lightning pipeline

For each, trace the flow: **TSV → vocab → Dataset → DataLoader → model → training loop → greedy decode → metrics**.

### Phase 2 — Run the Easy Notebooks (1 day, GPU)
- **Attention:** upload 3 TSVs from `data/` when prompted, Run All (20 epochs, ~1–2 h on T4). Expected: ~35% word / 70% char.
- **Transformer:** same. Expected: ~34% word / 67% char.
- Both save `attention_best.pt` / `transformer_best.pt`.

### Phase 3 — Run the Sweep Notebook (1 day, Kaggle recommended)
- Kaggle: New Notebook → Import `seq2seq_pytorch_without_attention.ipynb`, attach dataset, GPU T4 x2, Internet On, add `WANDB_API_KEY` secret + env-var cell.
- Save Version → Save & Run All (background, ~6–14 h for 10 runs). The final cell auto-fetches the best run from W&B, downloads its checkpoint, and prints test metrics.

### Phase 4 — Understand & Extend (ongoing)
- Fill `README.md` results table.
- Try one improvement: raise attention dropout 0.1→0.3, or implement beam search, or visualize attention weights.

---

## 4. Interview Questions & Answers

### Data & Preprocessing

**Q1. Why character-level and not subword (BPE)?**
A. Hindi has 50+ Devanagari characters but word vocab is large and transliteration is highly character-aligned (e.g. `sh` → `श`). Character-level gives a tiny vocab (~30 Roman, ~68 Devanagari), fast training, and handles unseen words via composition.

**Q2. How is the vocabulary built? What about unseen characters at test time?**
A. Count every character in `train + dev`, add special tokens `<PAD> <SOS> <EOS> <UNK>`, then map each char to an index. Any char not in the vocab becomes `<UNK>` at test time.

**Q3. Why pad sequences?**
A. Batches require a uniform shape. `collate_fn` pads each batch to its longest sequence with `<PAD>`; `CrossEntropyLoss(ignore_index=PAD)` ensures padded positions contribute zero gradient.

**Q4. Why build vocab from train+dev and not train only?**
A. Including dev slightly enriches the vocabulary without leaking test labels. The trade-off is minor dev-set leakage, accepted here for fair comparison across the three architectures (all use the same splits).

### Model Design

**Q5. Why bidirectional encoder but unidirectional decoder?**
A. The encoder sees the whole word at once, so it can use future characters. The decoder must generate left-to-right, one character at a time — it cannot look ahead.

**Q6. How is the decoder's initial hidden state set from a bidirectional encoder?**
A. Encoder hidden is `(num_layers*2, batch, hidden)`. We reshape to `(num_layers, 2, batch, hidden)` and **sum the two directions** → `(num_layers, batch, hidden)`. For LSTM we do this for both `h` and `c`.

**Q7. What does Bahdanau attention compute?**
A. Score `vᵀ tanh(W_enc·e_i + W_dec·h_t)` for every encoder output `e_i` against the current decoder hidden `h_t`. Softmax gives attention weights, their weighted sum is the context vector fed into the decoder.

**Q8. How is the Transformer different?**
A. It drops recurrence entirely. Each token attends to every other token via scaled dot-product `softmax(QKᵀ/√d_k)V`, in three flavors: encoder self-attention, masked decoder self-attention, and cross-attention. Positional encodings (sinusoidal) inject order information that RNNs get for free.

**Q9. What embedding/hidden sizes were used and why?**
A. Baseline sweep searched emb 16/32/64/256 and hidden 16–256; the winning attention run used emb 64, hidden 256, 1 layer, LSTM. Transformer used d_model 128, 4 heads, 3 layers. Small values suffice because character vocab is tiny.

### Training

**Q10. What is teacher forcing and exposure bias?**
A. During training we feed the ground-truth character as the decoder's next input (with probability e.g. 0.5–1.0) — faster convergence but at test time the model only sees its own predictions, so errors compound. Mitigations: schedule-sampling, scheduled teacher-forcing decay, or beam search at inference.

**Q11. How did you prevent overfitting?**
A. Dropout (0.1–0.3), weight decay (1e-5), early stopping (patience 5 on val_bleu), and picking the checkpoint with best validation score.

**Q12. Why gradient clipping?**
A. RNNs unrolled over timesteps can produce exploding gradients (repeated multiplication). `clip_grad_norm_(..., 1.0)` caps the global gradient norm.

**Q13. Why different batch sizes / learning rates?**
A. The sweep searched batch 32/64/128 and lr 1e-4–1e-2. The best baseline run used bs 64, lr ~1.3e-03.

### Evaluation

**Q14. How do word accuracy, character accuracy, and BLEU differ?**
A. Word accuracy = exact full-word match. Character accuracy = per-character match up to `min(len(pred), len(truth))` divided by truth length. BLEU = n-gram overlap with smoothing (sentence_bleu on character lists) — rewards partial word matches.

**Q15. Why is BLEU useful alongside accuracy?**
A. Word accuracy is harsh (one wrong character → 0). BLEU gives credit for near-correct transliterations, reflecting usability better.

**Q16. Greedy vs beam search — when does beam search hurt?**
A. Beam search keeps top-k hypotheses, usually better but slower (k× work). It can hurt when beam hypotheses share a prefix error and prune away a correct late divergence, or when the model is poorly calibrated.

### Attention & Transformer Analysis

**Q17. Can you visualize attention? Do weights align sensibly?**
A. Yes — plot `attn_weights` (batch × src_len) per output step. Ideally `श` lights up near `sh`, `आ` near `aa`, etc. Misalignment on longer words signals the model relying on language-model priors.

**Q18. Why need positional encoding in Transformers but not RNNs?**
A. RNNs process sequentially, so order is implicit in the recurrence. Transformers process all positions in parallel via attention, so position must be injected explicitly (sinusoidal or learned embeddings).

**Q19. If you had 10× more data, which architecture benefits most?**
A. Likely the Transformer — its parallelism and lack of recurrent bottlenecks scale better with data and parameters. Attention-augmented RNNs plateau earlier.

### Engineering

**Q20. How did you make runs reproducible?**
A. Fixed seed (`SEED=42`) across `random`, `numpy`, `torch`, `torch.cuda` in every notebook.

**Q21. How did you handle the Kaggle session limits?**
A. The baseline sweep uses W&B — runs and checkpoints sync to wandb.ai. Even if a session dies, the sweep lives on the server and can be resumed with `wandb.agent(sweep_id, ...)` or its best checkpoint can be downloaded directly for testing. Progress bars were disabled to prevent log-flood kills.

**Q22. Why is data/ gitignored?**
A. The 3 TSVs are regenerable from `aksharantar_sampled.zip` via `convert_aksharantar.py`. The zip and `*.pt` checkpoints are large binaries — gitignored to keep the repo small; W&B stores the true model registry.

---

## 5. Quick Self-Check Before an Interview

- [ ] Can you derive `vᵀ tanh(W·e + U·h)` on a whiteboard?
- [ ] Can you explain Q/K/V in one sentence each?
- [ ] Can you walk through one batch: shapes of `src_pad`, `enc_outputs`, `logits`?
- [ ] Can you report your three BLEU/accuracy numbers and say which model won and why?
- [ ] Can you describe one failure example from the test set and hypothesize the cause?

---

## 6. Suggested Next Steps (if you have more time)

1. Increase attention dropout 0.1 → 0.3 and re-train.
2. Implement beam search (k=4) for the attention/transformer models and re-evaluate.
3. Plot attention heatmaps for 10 test words.
4. Train on another Dakshina language (Telugu, Gujarati) and compare.
5. Containerize with Docker or export to ONNX for deployment.

---

*Evidence for this guide: `SKILLS.md` (deeper theory), `README.md` (results), wandb.ai project `kowshikpolisetty9-iit-kharagpur/Machine Transliteration`, and the `_tested.ipynb` notebooks with stored outputs.*
