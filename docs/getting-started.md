# Getting Started

This guide gets you from zero to a running OVOS Hierarchical KNN Pipeline in a few minutes.

---

## Prerequisites

- Python 3.10 or later
- An OVOS installation (ovos-core, ovos-workshop 0.1.7 or later)
- An AVX2-capable CPU (needed for the default quantized encoder; most x86-64 CPUs since about 2013 have it)
- About 600 MB of free disk space (encoder about 94 MB, index about 466 MB)
- Internet access on first run, to download the pre-built index from HuggingFace

To check whether your CPU supports AVX2:

```bash
grep -m1 avx2 /proc/cpuinfo && echo "AVX2 supported" || echo "AVX2 NOT found"
```

If AVX2 is not available, see [Troubleshooting — No AVX2](troubleshooting.md#no-avx2).

---

## Installation

```bash
pip install ovos-hierarchical-knn-pipeline
```

This installs the plugin and all runtime dependencies (`faiss-cpu`, `onnxruntime`, `tokenizers`, `numpy`, `tqdm`).

---

## Basic configuration

Open (or create) `~/.config/mycroft/mycroft.conf` and add this section:

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
      "conf_low": 0.15
    }
  }
}
```

On first start, the plugin downloads the pre-built index from HuggingFace (about 560 MB). Later starts load from the local HuggingFace cache.

---

## Offline / air-gapped setup

Download the model manually and point `index_dir` at it:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2',
    local_dir='/opt/ovos-knn-index'
)
"
```

Then in `mycroft.conf`:

```json
{
  "intents": {
    "ovos_hierarchical_knn_pipeline": {
      "index_dir": "/opt/ovos-knn-index"
    }
  }
}
```

---

## Verify the plugin is loaded

After starting OVOS, look for this log line:

```
INFO  HierarchicalKNNIntentPipeline  Loaded index from /path/to/index
```

You can also emit a test utterance over the message bus:

```python
from ovos_bus_client import MessageBusClient
from ovos_bus_client.message import Message

bus = MessageBusClient()
bus.run_in_thread()
bus.emit(Message("recognizer_loop:utterance", {
    "utterances": ["play some jazz"],
    "lang": "en-US"
}))
```

---

## Pipeline position

The KNN pipeline can sit at any position relative to Adapt and Padatious. The layout above places it before the deterministic engines at each confidence tier, so it gets first refusal on ambiguous utterances while still deferring to exact-match engines when they fire.

You can also place it after all deterministic stages, so it fires only when nothing else matched:

```json
"pipeline": [
  "adapt-high",
  "padatious-high",
  "adapt-medium",
  "padatious-medium",
  "adapt-low",
  "padatious-low",
  "ovos-hierarchical-knn-pipeline-low",
  "fallback-low"
]
```

See [Configuration Reference](configuration.md) for all options.

---
[Home](index.md) · [Configuration →](configuration.md)
