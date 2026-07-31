# OVOS Hierarchical KNN Pipeline — Documentation

An intent matching pipeline for [OpenVoiceOS (OVOS)](https://openvoiceos.org). It uses a two-stage hierarchical k-NN classifier backed by IBM Granite embeddings and a FAISS vector index.

---

## Contents

| Guide | Audience | Summary |
|---|---|---|
| [Getting Started](getting-started.md) | Users | Install, configure, and run the plugin |
| [Configuration Reference](configuration.md) | Users | Every config key explained |
| [Architecture](architecture.md) | Developers | How the classifier and pipeline work |
| [API Reference](api-reference.md) | Developers | Public classes and methods |
| [Training a Custom Index](training.md) | Developers | Build your own index from scratch |
| [Encoders](encoders.md) | Developers | Encoder implementations and auto-detection |
| [Testing](testing.md) | Developers | Unit tests, E2E tests, live fixtures |
| [Troubleshooting](troubleshooting.md) | Everyone | Common problems and fixes |

---

## Quick facts

- **Languages:** English, Portuguese, Spanish, French, Italian, German, Dutch, Catalan, Galician, Danish, Basque
- **Encoder:** IBM Granite Embedding 97M Multilingual R2 (quantized ONNX, about 94 MB)
- **Index format:** FAISS IVF+PQ (about 233 MB)
- **Total footprint:** about 320 MB RAM, about 345 MB storage (with quint8-quantized ONNX)
- **CPU requirement:** AVX2 (for the default quantized encoder)
- **License:** Apache 2.0
- **HuggingFace model:** [fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2](https://huggingface.co/fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2)
