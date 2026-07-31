# Testing

The test suite covers the pipeline plugin, the classifier, and end-to-end intent matching.

---

## Running the tests

```bash
pip install pytest
pytest tests/
```

The full suite, excluding live tests, runs without a GPU or internet connection. The classifier tests use small synthetic datasets. The pipeline tests mock the OVOS message bus.

---

## Test files

### `tests/test_pipeline.py`

Unit tests for `HierarchicalKNNIntentPipeline`. Coverage includes:

- Plugin initialization and config loading
- Intent manifest querying (Adapt and Padatious)
- Intent synchronization on skill load and unload
- Filtering utterances to registered intents only
- Confidence threshold gating (high, medium, low)
- Probability renormalization
- Special label gating (`ocp:play`, `common_query:common_query`, `stop:stop`)

The tests mock the classifier, so they run in milliseconds without loading the real model.

### `tests/test_classifier.py`

Unit tests for `HierarchicalPairKNNClassifier`. Coverage includes:

- Classifier initialization and label parsing
- Hierarchy level decomposition
- Wu-Lin probability estimation (basic and adaptive forms)
- End-to-end build, save, load, and predict workflow
- Active domain filtering (`set_active_domains`)
- Dimension mismatch detection

These tests use a small synthetic dataset (10 classes, about 50 examples) built in memory.

### `tests/test_ovoscope_e2e.py`

End-to-end tests using the `ovoscope` MiniCroft harness. The tests mock the real model download, so they run fully offline. They check:

- `match_high` returns a match above `conf_high`
- `match_medium` returns a match above `conf_medium`
- `match_low` returns a match above `conf_low`
- Below-threshold utterances return `None`

### `tests/test_live_fixture.py`

An optional live end-to-end test that downloads the real model from HuggingFace and evaluates it against an en-US intent fixture from `ovos-localize`.

Enable it by setting `OVOSCOPE_LIVE=1`:

```bash
OVOSCOPE_LIVE=1 pytest tests/test_live_fixture.py -v
```

The test allows up to 20% drift on the fixture accuracy. It fails only if fewer than 80% of fixture utterances match correctly.

---

## Fixture data

`tests/fixtures/en_us_intents.jsonl` is a JSONL file with utterances and expected intent labels, sourced from the `ovos-localize` en-US dataset. Each line looks like this:

```json
{"label": "ovos-skill-moviemaster.openvoiceos:movie.top.intent", "utterance": "what are the highest rated films out now"}
```

---

## Writing new tests

Use the `HierarchicalPairKNNClassifier` directly with a tiny synthetic dataset:

```python
import numpy as np
from ovos_hierarchical_knn_pipeline.classifier import HierarchicalPairKNNClassifier

labels = ["music:play", "music:stop", "weather:get_weather"] * 20
texts  = ["play jazz"] * 20 + ["stop music"] * 20 + ["weather tomorrow"] * 20

clf = HierarchicalPairKNNClassifier(classes=labels, model_path="m2v-labse")
clf.build(documents=texts, labels=labels, index_dir="/tmp/test-index")
clf2 = HierarchicalPairKNNClassifier.from_disk("/tmp/test-index")

result = clf2.predict(["play some music"])
assert result[0] == "music:play"
```

For pipeline-level tests, use `ovoscope` or mock the FAISS index and encoder as in `test_ovoscope_e2e.py`.

---

## CI

Tests run automatically on push through GitHub Actions. CI excludes the live fixture test, because it needs network access and about 560 MB of model data. To add it to CI, set the `OVOSCOPE_LIVE` secret in your repository settings.

---
[← Encoders](encoders.md) · [Home](index.md) · [Troubleshooting →](troubleshooting.md)
