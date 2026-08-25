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

```bash
pip install -r requirements.txt
```

For the RNN baseline notebook, you will also need PyTorch Lightning:

```bash
pip install pytorch-lightning
```

### 2. Prepare the dataset

**Option A — Convert the included Aksharantar zip (recommended, Telugu default):**

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

- **Word Accuracy** — exact match between predicted and reference Devanagari word.
- **Character Accuracy** — position-wise character match.
- **BLEU** — sentence-level BLEU with smoothing (RNN baseline notebook).

## Results

### Telugu (current, Aksharantar te.* — 51,200 / 4,096 / 4,096)
Fresh Telugu results will be filled after the new sweeps. The table below is the **previous Hindi run** (kept for reference until Telugu finishes).

| Model                    | Word Acc (%) | Char Acc (%) | BLEU   |
|--------------------------|-------------|-------------|--------|
| RNN baseline (PyTorch)   | 33.81       | 66.64       | 0.5649 |
| BiRNN + Bahdanau attention | 35.52     | 70.03       | 0.5962 |
| Transformer              | 34.33       | 67.41       | 0.5814 |

Notes (Hindi):
- RNN baseline: GRU/LSTM/RNN sweep via W&B + PyTorch Lightning (best val_bleu **0.6166**: LSTM, 3 layers, emb 256, hidden 128, no-bidir, lr 1.35e-03, bs 64, tf 1.0).
- Attention model: bidirectional LSTM encoder + Bahdanau additive-attention decoder, 1.36M params (`seq2seq_pytorch_attention_tested.ipynb`).
- Transformer: d_model=128, 4 heads, 3 layers, 1.02M params (`seq2seq_transformer_tested.ipynb`).
- BLEU computed with the same `nltk.translate.bleu_score.sentence_bleu` (method1 smoothing, per-word character list) as the baseline notebook; mid-table BLEU values recomputed locally on the stored checkpoints.
- All runs use fixed random seed 42 for reproducibility.

> Telugu sweeps (6 runs each, beam search, heatmaps) are in progress — the table above will be replaced with `te.*` greedy + beam numbers once they finish.

## Repository Structure

```
├── README.md
├── SKILLS.md                                # concepts & interview prep notes
├── requirements.txt
├── convert_aksharantar.py                   # builds data/*.tsv from aksharantar_sampled.zip
├── aksharantar_sampled.zip                  # raw multi-language data (not committed)
├── data/                                    # generated TSVs (not committed, `te.*` default)
│   ├── te.translit.sampled.train.tsv   # Telugu (51k/4k/4k)
│   ├── te.translit.sampled.dev.tsv
│   └── te.translit.sampled.test.tsv
├── seq2seq_pytorch_without_attention.ipynb  # RNN baseline + W&B sweeps
├── seq2seq_pytorch_attention.ipynb          # BiRNN encoder + Bahdanau attention
├── seq2seq_transformer.ipynb                # Transformer encoder-decoder
├── *_tested.ipynb                           # copies with stored outputs of full runs
└── *.pt                                     # trained checkpoints (not committed)
```
