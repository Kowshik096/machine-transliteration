# Machine Transliteration — Seq2Seq Models

Character-level encoder–decoder models for Hindi transliteration (Latin → Devanagari) using the [Dakshina dataset](https://github.com/google-research-datasets/dakshina).

All models are implemented in **PyTorch**.

## Models

| Notebook | Architecture |
|----------|-------------|
| `seq2seq_pytorch_without_attention.ipynb` | LSTM/GRU/RNN encoder–decoder (Lightning) |
| `seq2seq_pytorch_attention.ipynb` | LSTM/GRU/RNN encoder–decoder + Bahdanau attention |
| `seq2seq_transformer.ipynb` | Transformer encoder–decoder |

All three notebooks use the same **Dakshina Hindi** train/dev/test splits for fair comparison.

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

**Option A — Convert the included Aksharantar zip (recommended):**

```bash
python convert_aksharantar.py --zip aksharantar_sampled.zip --outdir data
export DAKSHINA_DATA_DIR="$(pwd)/data"
```

**Option B — Download the original Dakshina dataset:**

Download from https://github.com/google-research-datasets/dakshina and extract the Hindi lexicon files:

```
hi.translit.sampled.train.tsv
hi.translit.sampled.dev.tsv
hi.translit.sampled.test.tsv
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

## Weights & Biases (Optional)

The RNN baseline notebook uses W&B for sweep-based hyperparameter tuning. The attention and Transformer notebooks train directly without W&B. If you want sweep tuning for attention/Transformer, add your own sweep config.

```bash
wandb login
```

## Evaluation Metrics

- **Word Accuracy** — exact match between predicted and reference Devanagari word.
- **Character Accuracy** — position-wise character match.
- **BLEU** — sentence-level BLEU with smoothing (RNN baseline notebook).

## Results

Full Dakshina Hindi test set (4,096 words), greedy decoding, best val-loss checkpoint, 20 epochs, batch size 128.

| Model                    | Word Acc (%) | Char Acc (%) | BLEU   |
|--------------------------|-------------|-------------|--------|
| RNN baseline (PyTorch)   | 33.81       | 66.64       | 0.5649 |
| BiRNN + Bahdanau attention | 35.52     | 70.03       | 0.5962 |
| Transformer              | 34.33       | 67.41       | 0.5814 |

Notes:
- RNN baseline: GRU/LSTM/RNN sweep via W&B + PyTorch Lightning (best val_bleu **0.6166**: LSTM, 3 layers, emb 256, hidden 128, no-bidir, lr 1.35e-03, bs 64, tf 1.0).
- Attention model: bidirectional LSTM encoder + Bahdanau additive-attention decoder, 1.36M params (`seq2seq_pytorch_attention_tested.ipynb`).
- Transformer: d_model=128, 4 heads, 3 layers, 1.02M params (`seq2seq_transformer_tested.ipynb`).
- BLEU computed with the same `nltk.translate.bleu_score.sentence_bleu` (method1 smoothing, per-word character list) as the baseline notebook; mid-table BLEU values recomputed locally on the stored checkpoints.
- All runs use fixed random seed 42 for reproducibility.

## Repository Structure

```
├── README.md
├── SKILLS.md                                # concepts & interview prep notes
├── requirements.txt
├── convert_aksharantar.py                   # builds data/*.tsv from aksharantar_sampled.zip
├── aksharantar_sampled.zip                  # raw multi-language data (not committed)
├── data/                                    # generated Hindi TSVs (not committed)
│   ├── hi.translit.sampled.train.tsv
│   ├── hi.translit.sampled.dev.tsv
│   └── hi.translit.sampled.test.tsv
├── seq2seq_pytorch_without_attention.ipynb  # RNN baseline + W&B sweeps
├── seq2seq_pytorch_attention.ipynb          # BiRNN encoder + Bahdanau attention
├── seq2seq_transformer.ipynb                # Transformer encoder-decoder
├── *_tested.ipynb                           # copies with stored outputs of full runs
└── *.pt                                     # trained checkpoints (not committed)
```
