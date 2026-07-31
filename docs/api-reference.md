# API Reference

This document covers the public Python API for the classifier and encoder abstractions. It is for developers who want to use these components outside of the OVOS plugin system, build custom pipelines, or run batch inference.

---

## `HierarchicalPairKNNClassifier`

```python
from ovos_hierarchical_knn_pipeline.classifier import HierarchicalPairKNNClassifier
```

### Constructor

```python
HierarchicalPairKNNClassifier(
    classes: list[str],
    k: int = 10,
    n: int = 5,
    sep: str = ":",
    nlist: int = 1024,
    pq_m: int = 16,
    nprobe: int = 32,
    model_path: str = "m2v-labse",
    gamma: float = 1.0,
    tau: float = 0.05,
    margin: float = 0.10,
    anchor_to_global: bool = True,
    renormalize: bool = True,
    encoder_file: str | None = None,
)
```

| Parameter | Description |
|---|---|
| `classes` | All intent label strings that will appear in the index. Format: `"domain:intent"`. |
| `k` | Number of nearest neighbors retrieved per k-NN search. Wu-Lin uses a dynamic effective k based on the margin, so this is an upper bound. |
| `n` | Number of top domains passed from L1 to L2. |
| `sep` | Separator character between hierarchy levels. |
| `nlist` | IVF quantizer centroids. Ignored for small datasets (flat index used instead). |
| `pq_m` | PQ sub-quantizers (bytes per vector). Lower means faster, less accurate. |
| `nprobe` | IVF cells probed at search time. Higher means more accurate, slower. |
| `model_path` | Path to the encoder model directory. |
| `gamma` | Distance decay rate for Wu-Lin weighting. `0` means uniform (ignore distance). |
| `tau` | Exact-match distance threshold. Nearest neighbors closer than this win 100%. |
| `margin` | Shell half-width for the adaptive neighborhood. |
| `anchor_to_global` | Anchor margin to the global nearest neighbor across all domains. |
| `renormalize` | Re-scale probabilities to sum to 1 after filtering. |
| `encoder_file` | ONNX filename override. Auto-detected when `None`. |

---

### Class methods

#### `from_disk`

```python
@classmethod
def from_disk(cls, index_dir: str | Path) -> HierarchicalPairKNNClassifier
```

Load a pre-built index from a local directory. This restores all hyperparameters from `meta.pkl`.

```python
clf = HierarchicalPairKNNClassifier.from_disk("/opt/ovos-knn-index")
```

---

#### `from_pretrained`

```python
@classmethod
def from_pretrained(
    cls,
    repo_id: str = "fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2",
    cache_dir: str | None = None,
) -> HierarchicalPairKNNClassifier
```

Download a pre-built index from HuggingFace and load it. The snapshot stays in the local cache after the first download.

```python
clf = HierarchicalPairKNNClassifier.from_pretrained()
```

---

### Instance methods

#### `build`

```python
def build(
    self,
    documents: list[str] | None,
    labels: list[str],
    index_dir: str | Path,
    embeddings: np.ndarray | None = None,
    batch_size: int = 512,
) -> None
```

Build the FAISS index from training data and save all artifacts to `index_dir`.

- Pass `documents` (raw text) to have the encoder compute embeddings.
- Pass `embeddings` (pre-computed float32, shape `[N, dim]`, L2-normalized) to skip encoding. `documents` can be `None` in this case.
- Pass both to override encoding with pre-computed embeddings while keeping text metadata.

```python
clf = HierarchicalPairKNNClassifier(classes=all_labels, model_path="/opt/granite-97m")
clf.build(
    documents=texts,
    labels=labels,
    index_dir="/opt/my-index",
)
```

---

#### `predict_proba`

```python
def predict_proba(
    self,
    documents: list[str] | np.ndarray,
    level: int | None = None,
    batch_size: int = 512,
) -> list[dict[str, float]]
```

Return a probability distribution over intent labels for each input.

- `documents`: list of strings (auto-encoded) or float32 numpy array of shape `[N, dim]` (pre-encoded).
- `level`: maximum hierarchy depth to predict. `None` means full depth. `1` means domain only.
- Returns: a list of dicts mapping label string to probability float.

```python
results = clf.predict_proba(["play some jazz", "what time is it"])
# [{"ocp:play": 0.87, "timer:get_time": 0.05, ...}, ...]
```

---

#### `predict`

```python
def predict(
    self,
    documents: list[str] | np.ndarray,
    level: int | None = None,
) -> list[str]
```

Return the top-1 label for each input.

```python
labels = clf.predict(["play some jazz"])
# ["ocp:play"]
```

---

#### `set_active_domains`

```python
def set_active_domains(self, domains: list[str]) -> None
```

Restrict L1 search to the given list of domains. L2 search inherits the restriction.

Pass an empty list to reset to the full domain set.

```python
clf.set_active_domains(["weather", "timer"])
results = clf.predict_proba(["what time is it"])
# Only weather and timer intents are considered
```

---

#### `get_depth`

```python
def get_depth(self) -> int
```

Return the number of hierarchy levels in the label format (for example `2` for `domain:intent`).

---

### Properties

#### `encoder`

```python
@property
def encoder(self) -> AnyEncoder
```

Lazily initialized encoder. Auto-detected from the `model_path` directory. See [Encoders](encoders.md).

---

## `HierarchicalKNNIntentPipeline`

```python
from ovos_hierarchical_knn_pipeline import HierarchicalKNNIntentPipeline
```

This class is the OVOS pipeline plugin. The OVOS plugin manager normally instantiates it. You can also instantiate it directly for testing.

### Constructor

```python
HierarchicalKNNIntentPipeline(bus: MessageBusClient, config: dict | None = None)
```

[Configuration Reference](configuration.md) describes the config keys.

### Methods

These follow the `ConfidenceMatcherPipeline` interface from `ovos-workshop`.

| Method | Fires when |
|---|---|
| `match_high(utterances, lang, message)` | Confidence ≥ `conf_high` |
| `match_medium(utterances, lang, message)` | Confidence ≥ `conf_medium` |
| `match_low(utterances, lang, message)` | Confidence ≥ `conf_low` |

Each method:
- Takes `utterances: list[str]`, `lang: str`, `message: Message`
- Returns `IntentHandlerMatch | None`

`IntentHandlerMatch` fields:

| Field | Content |
|---|---|
| `match_type` | Intent label string (for example `"ocp:play"`) |
| `match_data` | `{"utterance": str, "confidence": float}` |
| `skill_id` | Skill package name derived from the domain |
| `utterance` | The matched utterance string |

---

## Standalone usage example

```python
from ovos_hierarchical_knn_pipeline.classifier import HierarchicalPairKNNClassifier

# Load the default pre-built index
clf = HierarchicalPairKNNClassifier.from_pretrained()

# Restrict to the domains of installed skills
clf.set_active_domains(["weather", "timer", "ocp"])

# Batch inference
utterances = [
    "play some jazz music",
    "what is the weather tomorrow",
    "set a timer for 5 minutes",
]
for utt, probs in zip(utterances, clf.predict_proba(utterances)):
    top = max(probs, key=probs.get)
    print(f"{utt!r:40s} → {top} ({probs[top]:.2f})")
```

---
[← Architecture](architecture.md) · [Home](index.md) · [Training →](training.md)
