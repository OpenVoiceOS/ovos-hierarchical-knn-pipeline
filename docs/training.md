# Training a Custom Index

This guide walks through building a FAISS index from your own intent dataset. Use it when:

- You want to add languages the default model does not cover.
- You have domain-specific skills whose utterances differ a lot from the pre-built training set.
- You want to prune the index to only the skills you use (a smaller index gives faster inference).

---

## Pipeline overview

```text
raw_intents.csv
      │
      ▼  preprocess_data.py   : normalize labels, fill placeholders
      │
      ▼  balance_dataset.py   : (optional) cap large classes, augment small ones
      │
      ▼  encode_dataset.py    : (optional) pre-compute embeddings once
      │
      ▼  build_index.py       : build FAISS IVF+PQ index → index_dir/
```

---

## Training dependencies

```bash
pip install -r train/requirements.txt
```

This installs extra packages needed only for training: `model2vec`, `transformers`, `huggingface_hub`, `faker`, `pandas`, `scikit-learn`, `google-generativeai`.

---

## Input data format

The raw CSV must have at least two columns:

| Column | Content |
|---|---|
| `utterance` | The example sentence (for example `"play some jazz"`) |
| `label` | The intent label (for example `"ocp:play"` or `"ocp.play"`) |

An optional `lang` column (ISO-639-1 code, for example `"en"`) enables language filtering in `build_index.py`.

---

## Step 1: Preprocess

Normalize label formats and fill `{placeholder}` slots with Faker-generated values:

```bash
python train/preprocess_data.py \
    dataset/raw_intents.csv \
    dataset/intents_clean.csv
```

This script:
- Converts dots to colons in labels (`ocp.play` becomes `ocp:play`)
- Ensures labels follow the `domain:intent` format
- Replaces `{city}`, `{artist}`, `{number}`, and similar placeholders with realistic fake values using Faker
- Adds an `entity` column listing the slot types found in each row

---

## Step 2: Balance (optional)

Balance the class distribution to improve classifier performance on rare intents:

```bash
python train/balance_dataset.py \
    --input   dataset/intents_clean.csv \
    --output  dataset/intents_balanced.parquet \
    --model   /path/to/granite-97m-r2 \
    --encoder-file model.onnx
```

This script:
- Caps very large classes using FAISS k-means (keeps the most diverse examples)
- Downsamples mid-tier classes to a target count
- Augments micro-tier classes with Faker slot-filling and, optionally, Gemini LLM paraphrases
- Pre-computes embeddings and writes them to the output Parquet file

To use Gemini augmentation, set the `GEMINI_API_KEY` environment variable.

The output Parquet file contains the original columns plus `dim_0 … dim_N` embedding columns. You can pass this file directly to `build_index.py`.

---

## Step 3: Encode (optional, recommended)

If you skipped `balance_dataset.py`, pre-compute embeddings so `build_index.py` does not need to re-encode on every build:

```bash
python train/encode_dataset.py \
    --input   dataset/intents_clean.csv \
    --output  dataset/intents_encoded.parquet \
    --model   /path/to/granite-97m-r2 \
    --encoder-file model.onnx
```

The output Parquet has the same schema as the `balance_dataset.py` output and can be passed directly to `build_index.py`.

---

## Step 4: Build the index

```bash
python train/build_index.py \
    --dataset   dataset/intents_balanced.parquet \
    --index-dir models/my-index \
    --model     /path/to/granite-97m-r2
```

When the input is a plain CSV (no pre-computed embeddings), pass `--encoder-file` to select the ONNX variant to use for encoding:

```bash
python train/build_index.py \
    --dataset      dataset/intents_clean.csv \
    --index-dir    models/my-index \
    --model        /path/to/granite-97m-r2 \
    --encoder-file model.onnx
```

### `build_index.py` options

| Option | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Path to CSV or Parquet file. |
| `--index-dir` | *(required)* | Directory to write index artifacts to. |
| `--model` | `models/granite-97m-onnx` | Path to the embedding model directory. |
| `--encoder-file` | `model.onnx` | ONNX file used for encoding (CSV only). |
| `--langs` | 11 EU languages | Comma-separated list of language codes to include. |
| `--min-count` | `10` | Minimum examples per label; the script drops rarer labels. |
| `--k` | `5` | Neighbors retrieved per Wu-Lin estimation. |
| `--n` | `4` | Top-n domains passed from L1 to L2 search. |
| `--nlist` | `1024` | Number of IVF centroids. |
| `--pq-m` | `16` | PQ sub-quantizers (bytes per vector). |
| `--nprobe` | `32` | IVF cells probed at search time. |
| `--margin` | `0.1` | Shell half-width for adaptive neighborhood. |
| `--gamma` | `1.0` | Distance decay rate (0 means uniform). |

---

## Using your custom index

Point `index_dir` at the output directory in `mycroft.conf`:

```json
{
  "intents": {
    "ovos_hierarchical_knn_pipeline": {
      "index_dir": "/path/to/models/my-index"
    }
  }
}
```

Or load it directly in Python:

```python
from ovos_hierarchical_knn_pipeline.classifier import HierarchicalPairKNNClassifier

clf = HierarchicalPairKNNClassifier.from_disk("/path/to/models/my-index")
print(clf.predict(["play some jazz"]))
```

---

## Advanced: diversity and multimodality analysis

Before building the final index, two analysis scripts can help improve data quality.

### `create_diverse_subset.py`

Selects the most semantically diverse sentences per label using a greedy farthest-point (maximin) algorithm:

```bash
python train/create_diverse_subset.py \
    --input   dataset/intents_encoded.parquet \
    --output  dataset/intents_diverse.parquet \
    --max-per-label 200
```

Use this script to trim very large classes down to a compact, high-coverage subset before building the index.

### `detect_multimodality.py`

Runs k-means sub-clustering within each label to detect semantic islands: labels whose training examples split into distinct groups.

```bash
python train/detect_multimodality.py \
    --input  dataset/intents_encoded.parquet
```

The report identifies labels that may need splitting (for example `timer:set_timer` covers both `"set a timer"` and `"start the countdown"` clusters) or additional examples to bridge the gap.

---

## Publishing to HuggingFace

To share your custom index:

```bash
pip install huggingface_hub
huggingface-cli login
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='/path/to/models/my-index',
    repo_id='your-username/your-index-name',
    repo_type='model',
)
"
```

Then set `hf_repo_id` in `mycroft.conf` to `"your-username/your-index-name"`.

---
[← API Reference](api-reference.md) · [Home](index.md) · [Encoders →](encoders.md)
