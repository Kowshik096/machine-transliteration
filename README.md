# Machine Transliteration — Seq2Seq Models

Character-level encoder–decoder models for transliteration — now on **Telugu (Latin → Telugu script)** — built from the [Aksharantar dataset](https://ai4bharat.iitm.ac.in) with the same Dakshina-compatible pipeline. Hindi support remains via `--lang hi`.

All models are implemented in **PyTorch**. This is a personal project forked from a Hindi baseline and extended to Telugu with full sweeps, beam search, and attention heatmaps.

## Models

| Notebook | Architecture |
|----------|-------------|
| `seq2seq_pytorch_without_attention.ipynb` | LSTM/GRU/RNN encoder–decoder (Lightning) |
| `seq2seq_pytorch_attention.ipynb` | LSTM/GRU/RNN encoder–decoder + Bahdanau attention |
| `seq2seq_transformer.ipynb` | Transformer encoder–decoder |

All three notebooks use the same **Aksharantar Telugu** train/dev/test splits (51,200 / 4,096 / 4,096) for fair comparison. Hindi splits are available via the same converter with `--lang hi`.

## Setup

### 1. Install dependencies

Install the package in editable mode (also installs all dependencies and the `dev` extras for `ruff`/`pytest`):

```bash
pip install -e ".[dev]"
```

For the RNN baseline notebook, you will also need PyTorch Lightning:

```bash
pip install pytorch-lightning
```

### 2. Prepare the dataset

Tracked source (`src/`, `scripts/`, `app.py`, `convert_aksharantar.py`,
`pyproject.toml`) is sufficient for a clean clone. Generated/external
artifacts are intentionally NOT committed:

- `data/*.tsv` — generated via `convert_aksharantar.py` (ignored `/data/`).
- `aksharantar_sampled.zip` — raw Aksharantar export, obtain separately (ignored `*.zip`).
- `*_best.pt` / `*_best.ckpt` — trained weights, obtain separately (ignored `*.pt`/`*.ckpt`).
- `results.zip`, `_results_unzipped/`, `wandb/`, `archive/` — experiment outputs.

**Option A — Convert the Aksharantar zip (recommended, Telugu default):**

```bash
python convert_aksharantar.py --zip aksharantar_sampled.zip --outdir data --lang te
export DAKSHINA_DATA_DIR="$(pwd)/data"
# For Hindi instead: --lang hi
```

**Option B — Download the original Dakshina dataset:**

Download from https://github.com/google-research-datasets/dakshina and extract the Hindi lexicon files:

```
te.translit.sampled.train.tsv   # Telugu (default)
te.translit.sampled.dev.tsv
te.translit.sampled.test.tsv
# or hi.translit.sampled.*.tsv  # Hindi with --lang hi
```

Place them in a folder and set:

```bash
export DAKSHINA_DATA_DIR="/path/to/dakshina/hi/lexicons"
```

Or edit `DATA_DIR` at the top of each notebook.

### Colab

1. Upload the three TSV files to Google Drive.
2. Default path: `/content/drive/MyDrive/dakshina_dataset_v1.0/dakshina_dataset_v1.0/hi/lexicons`
3. Open a notebook, mount Drive when prompted, then run all cells.

### Kaggle

1. Upload the three TSV files as a Kaggle Dataset.
2. Set:
   ```python
   os.environ["DAKSHINA_DATA_DIR"] = "/kaggle/input/<your-dataset-name>"
   ```
   before running the data-loading cell.

### Local

```bash
export DAKSHINA_DATA_DIR="$HOME/data/dakshina/hi/lexicons"
jupyter lab
```

## Weights & Biases

All three notebooks now support W&B sweeps. The RNN baseline notebook has a full 11-parameter Bayes sweep; attention and Transformer notebooks have their own sweep configs (see `sweep_config` cells). Attention adds heatmap visualization via `plot_attention_heatmap()`, and all three support `beam_search_decode`.

```bash
wandb login
```

## Evaluation Metrics

- **Word Accuracy** — exact match between predicted and reference target-script word.
- **Character Accuracy** — position-wise character match.
- **BLEU** — sentence-level BLEU with smoothing (RNN baseline notebook).

## Results

### Telugu (current, Aksharantar te.* — 51,200 / 4,096 / 4,096) — updated 2026-08-30 from Colab + Kaggle + Evals

| Model                    | Greedy Word (%) | Greedy Char (%) | Beam 4 Word (%) | Beam 4 Char (%) | BLEU   |
|--------------------------|----------------|----------------|----------------|----------------|--------|
| RNN baseline (PyTorch)   | 45.80 (46.58*) | 80.35 (80.39*) | **47.29** | **81.25** | **0.6979** |
| BiRNN + Bahdanau attention | 48.56 | 82.66 | **49.61** | 82.61 | — |
| Transformer              | 43.38 | 79.23 | **43.87** | 79.36 | — |

* 46.58/80.39 is Kaggle direct `seq2seq_pytorch_without_attention_tested.ipynb` test (beam 2); 45.80/80.35 is `seq2seq_evals.ipynb` greedy re-eval for fair comparison (same `baseline_best.ckpt`).

**Notes (Telugu, 2026-08-30):**
- RNN baseline: Kaggle sweep `tbkdzfq2` best `val_bleu 0.7365` → test `Word 0.4658 | Char 0.8039 | BLEU 0.6979` greedy (`seq2seq_pytorch_without_attention_tested.ipynb`, `baseline_best.ckpt` 6,946,773 bytes, `results.zip` 35,823,835 bytes, `lstm 2 layers, emb 64, hidden 256, no-bidir, dropout 0.2, lr 7.05e-03, bs 64, tf 0.7`). Evals re-run `seq2seq_evals_tested.ipynb:15` (468,583 bytes): greedy `0.4580/0.8035` → beam 4 `0.4729/0.8125`.
- Attention: greedy `0.4856/0.8266` from `seq2seq_pytorch_attention_tested.ipynb:1617` (20 epochs, `attention_best.pt` 5,433,394 bytes, BiLSTM + Bahdanau `score=vᵀ tanh(W_enc·e + W_dec·h)`, ~1.36M params, best `Val 0.6229` @13/20) → beam 4 `0.4961/0.8261` from `seq2seq_evals_tested.ipynb:13`.
- Transformer: greedy `0.4338/0.7923` from `seq2seq_transformer_tested.ipynb:1421` (20 epochs, `transformer_best.pt` 4,361,707 bytes, `d_model=128, 4 heads, 3 layers`, ~1.02M params, best `Val 0.2862` @20/20) → beam 4 `0.4387/0.7936` from `seq2seq_evals_tested.ipynb:14`.
- Evals: `seq2seq_evals.ipynb` (17 cells, 44,870 bytes) → `seq2seq_evals_tested.ipynb` (468,583 bytes) runs all 3 models greedy vs beam 4 on `test 4096` + `plot_attention_heatmap` for 10 words (`Cell 16`). BLEU for attention/transformer was not logged in those runs; recompute locally via `nltk sentence_bleu` if needed.
- All runs `SEED=42`.

### Hindi (previous, Dakshina hi.* — 51,200 / 4,096 / 4,096) — for reference

| Model                    | Word Acc (%) | Char Acc (%) | BLEU   |
|--------------------------|-------------|-------------|--------|
| RNN baseline (PyTorch)   | 33.81       | 66.64       | 0.5649 |
| BiRNN + Bahdanau attention | 35.52     | 70.03       | 0.5962 |
| Transformer              | 34.33       | 67.41       | 0.5814 |

Notes (Hindi):
- RNN baseline: GRU/LSTM/RNN sweep via W&B + PyTorch Lightning (best val_bleu **0.6166**: LSTM, 3 layers, emb 256, hidden 128, no-bidir, lr 1.35e-03, bs 64, tf 1.0).
- Attention model: bidirectional LSTM encoder + Bahdanau additive-attention decoder, 1.36M params (`seq2seq_pytorch_attention_tested.ipynb`).
- Transformer: d_model=128, 4 heads, 3 layers, 1.02M params (`seq2seq_transformer_tested.ipynb`).

> All 3 Telugu runs + beam search (k=4) + heatmaps completed via `seq2seq_evals_tested.ipynb` — optional next is dropout tuning / additional languages.

### What I built on top of the Hindi baseline

- **Telugu pipeline:** `convert_aksharantar.py` turns the Aksharantar export into the Dakshina-format splits (51,200 / 4,096 / 4,096).
- **RNN-variant sweep:** a 10-trial W&B Bayes sweep over 11 hyperparameters, including cell type (LSTM / GRU / vanilla RNN), depth, embedding and hidden sizes, bidirectionality, dropout, learning rate and teacher forcing (`wandb.agent(..., count=10)` in `seq2seq_pytorch_without_attention.ipynb`).
- **Result:** BiLSTM + Bahdanau attention reaches **49.61%** word accuracy with beam 4, ahead of the LSTM baseline (47.29%, +2.3 points) and the Transformer (43.87%, +5.7 points).
- **Transformer decoding fix:** inference used to run a single forward pass over a fixed 64-token target, so the model never saw its own predictions and emitted the same character 64 times. `greedy_decode` now decodes autoregressively: it feeds each predicted token back and stops at `<EOS>` (`src/transliteration/utils/decode.py`). `tests/test_step1_regressions.py` pins this with a mock Transformer that records every prefix length it sees.
- **Engineering:** 40 offline pytest tests (`pytest tests/`), CI running `ruff` and the tests on every push (`.github/workflows/ci.yml`), a Docker image (`Dockerfile`), and a Streamlit demo (`app.py`).

## Repository Structure

```
├── README.md
├── SKILLS.md                                # concepts & interview prep notes
├── pyproject.toml                           # build system, deps, ruff config
├── requirements.txt                         # pinned runtime deps
├── convert_aksharantar.py                   # builds data/*.tsv from aksharantar_sampled.zip
├── aksharantar_sampled.zip                  # raw multi-language data (not committed)
├── data/                                    # generated TSVs (not committed, `te.*` default)
│   ├── te.translit.sampled.train.tsv   # Telugu (51k/4k/4k)
│   ├── te.translit.sampled.dev.tsv
│   └── te.translit.sampled.test.tsv
├── src/transliteration/                     # package source
│   ├── data/        # vocab, dataset, collate
│   ├── models/      # seq2seq, attention, transformer
│   └── utils/       # decode (greedy/beam), metrics
├── scripts/
│   ├── train.py     # training entry point (argparse)
│   └── eval.py      # evaluation entry point (argparse)
├── app.py           # Streamlit demo
├── tests/           # pytest tests
├── seq2seq_pytorch_without_attention.ipynb  # RNN baseline + W&B sweeps
├── seq2seq_pytorch_attention.ipynb          # BiRNN encoder + Bahdanau attention
├── seq2seq_transformer.ipynb                # Transformer encoder-decoder
├── *_tested.ipynb                           # copies with stored outputs of full runs
└── *.pt                                     # trained checkpoints (not committed)
```

## Checkpoints & Architecture Presets

Each architecture uses its own preset (see `src/transliteration/models/presets.py`);
do not share one universal config:

- `rnn` (baseline `baseline_best.ckpt`, Lightning envelope): 2-layer uni LSTM, emb 64, hidden 256, dropout 0.2.
- `attention` (`attention_best.pt`, raw state_dict): 1-layer BiLSTM, emb 64, hidden 256, dropout 0.1.
- `transformer` (`transformer_best.pt`, raw state_dict): d_model 128, 4 heads, 3 layers, dim_ff 256, dropout 0.1.

`app.py`, `scripts/train.py`, and `scripts/eval.py` resolve defaults from these
presets. The Transformer always uses autoregressive greedy decoding; requesting
`beam_size > 1` falls back to greedy with an explicit message (beam search
remains RNN-only). The baseline loader accepts both raw state_dicts and
Lightning `{"state_dict": {"model.*": ...}}` envelopes.
