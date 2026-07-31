[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/ovos-hierarchical-knn-pipeline)

# OVOS Hierarchical KNN Intent Pipeline

An intent matching pipeline for [OpenVoiceOS (OVOS)](https://openvoiceos.org). It uses a two-stage hierarchical k-NN classifier backed by IBM Granite embeddings and a FAISS index.

The plugin classifies natural language utterances into intent labels using a pre-built [FAISS](https://github.com/facebookresearch/faiss) index. The intent labels come from the system (Adapt, Padatious, and plugin-specific labels). The plugin considers only intents from loaded skills and ignores labels from unregistered intents. Use this pipeline when other deterministic engines fail to give a high-confidence match.

---

## Features

* Two-stage hierarchical search: domain-level, then intent-level, using Wu-Lin pairwise probability estimation
* Integration with OVOS pipelines
* Multilingual support (en, pt, es, fr, it, de, nl, ca, gl, da, eu)
* IBM Granite Embedding 97M Multilingual R2 encoder (quantized ONNX, about 94 MB)
* FAISS IVF+PQ index for low-memory inference on edge devices
* Syncs Adapt and Padatious intents at runtime
* Considers only intents from loaded skills and ignores unregistered labels
* Domain pre-filtering: the search scope is limited to the domains of loaded skills

The quantized encoder (`model_quint8_avx2.onnx`) needs an AVX2-capable CPU. The total footprint (index and encoder) is about 560 MB.

---

## Installation

Install the plugin with `pip`:

```bash
pip install ovos-hierarchical-knn-pipeline
```

---

## Configuration

Add the following to your `mycroft.conf`. When `index_dir` is omitted, the plugin downloads the default pre-built index from HuggingFace on first run.

```json
{
  "intents": {
    "pipeline": [
      "ovos-hierarchical-knn-pipeline-high",
      "adapt-high",
      "padatious-high",
      "ovos-hierarchical-knn-pipeline-medium",
      "adapt-medium",
      "padatious-medium",
      "ovos-hierarchical-knn-pipeline-low",
      "adapt-low",
      "padatious-low",
      "fallback-low"
    ],
    "ovos_hierarchical_knn_pipeline": {
      "conf_high": 0.7,
      "conf_medium": 0.5,
      "conf_low": 0.15,
      "ignore_intents": []
    }
  }
}
```

To use a local copy of the index (this gives a faster startup and needs no internet connection):

```json
{
  "intents": {
    "ovos_hierarchical_knn_pipeline": {
      "index_dir": "/path/to/local/index"
    }
  }
}
```

| Key | Default | Description |
|---|---|---|
| `index_dir` | *(auto-download)* | Path to a local index directory. When omitted, the plugin downloads the index from HuggingFace. |
| `hf_repo_id` | `fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2` | HuggingFace repo to download when `index_dir` is not set. |
| `hf_cache_dir` | HF default cache | Local directory for the downloaded snapshot. |
| `conf_high` | `0.7` | Minimum confidence for `match_high`. |
| `conf_medium` | `0.5` | Minimum confidence for `match_medium`. |
| `conf_low` | `0.15` | Minimum confidence for `match_low`. |
| `ignore_intents` | `[]` | Intent labels to suppress. |
| `renormalize` | `false` | Re-scale surviving probabilities to sum to 1 after filtering. The classifier already renormalizes internally, so turning this on re-scales a second time over the registered intents only. |

The FAISS index is pre-built on a fixed dataset and cannot learn new skills dynamically. Skills not covered by the index are still reachable through the Adapt and Padatious stages of the pipeline.

---

## Usage

The `HierarchicalKNNIntentPipeline` class integrates with the OVOS intent system. It performs these steps:

1. Receives an utterance (text).
2. Encodes the utterance with the IBM Granite ONNX encoder.
3. Runs a two-stage hierarchical k-NN search, first across skill domains, then across intents within the top-scoring domains.
4. Filters out intents that are not part of the loaded skills.
5. Returns a match for the highest-confidence intent from the list of valid intents.

---

## Tips

* Tune `conf_high`, `conf_medium`, and `conf_low` to control the confidence threshold at each pipeline stage.
* Use the `ignore_intents` list to filter out specific problematic intents from predictions.
* The plugin syncs Adapt and Padatious intents automatically at runtime, over the OVOS message bus.
* Set `renormalize: true` to make probabilities sum to 1 after unregistered intents are filtered out.

A pre-built index is available on HuggingFace: [fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2](https://huggingface.co/fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2).

---

## Building a custom index

To train on a different skill set or language combination, see [docs/training.md](docs/training.md) or [train/README.md](train/README.md) for the full pipeline:

```
preprocess_data.py  →  balance_dataset.py  →  encode_dataset.py  →  build_index.py
```

---

## Documentation

Full documentation, from installation to building custom indexes, lives in the [`docs/`](docs/) directory:

| Guide | Summary |
|---|---|
| [Getting Started](docs/getting-started.md) | Install, configure, and run the plugin |
| [Configuration Reference](docs/configuration.md) | Every config key explained |
| [Architecture](docs/architecture.md) | How the classifier and pipeline work |
| [API Reference](docs/api-reference.md) | Public classes and methods |
| [Training a Custom Index](docs/training.md) | Build your own index from scratch |
| [Encoders](docs/encoders.md) | Encoder implementations and auto-detection |
| [Testing](docs/testing.md) | Unit tests, E2E tests, live fixtures |
| [Troubleshooting](docs/troubleshooting.md) | Common problems and fixes |

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## Credits

Developed by [**femelo**](https://github.com/femelo) for
[OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
