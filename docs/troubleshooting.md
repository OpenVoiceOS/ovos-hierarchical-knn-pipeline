# Troubleshooting

---

## No AVX2 support {#no-avx2}

**Symptom:** `onnxruntime` raises an `Illegal instruction` error or fails to load `model_quint8_avx2.onnx`.

**Cause:** The default quantized encoder needs AVX2 CPU instructions. Most x86-64 CPUs since about 2013 have them, but some older hardware and ARM devices do not.

**Fix:** Force the non-AVX2 quantized variant:

```json
{
  "intents": {
    "ovos_hierarchical_knn_pipeline": {
      "index_dir": "/path/to/index"
    }
  }
}
```

Then, when you load the classifier manually, specify the fallback encoder:

```python
clf = HierarchicalPairKNNClassifier.from_disk("/path/to/index")
clf._encoder = load_encoder("/path/to/model", onnx_filename="model_uint8.onnx")
```

Or rebuild the index with `model_uint8.onnx` as the encoder file and update `meta.pkl` to match.

For ARM devices such as Raspberry Pi, use the float32 variant (`model.onnx`). It is slower but has no special CPU requirements.

---

## Index not found / download fails

**Symptom:** `FileNotFoundError` or a network error on startup.

**Causes and fixes:**

1. No internet access: download the snapshot manually and set `index_dir` (see [Getting Started — offline setup](getting-started.md#offline--air-gapped-setup)).

2. HuggingFace rate limit: the download is large (about 560 MB). Retry after a few minutes, or authenticate with `huggingface-cli login`.

3. Corrupted cache: delete the cache and re-download.

   ```bash
   rm -rf ~/.cache/huggingface/hub/models--fdemelo--ovos-hierarchical-knn-granite-97m-multilingual-r2
   ```

---

## No intents matched / all results below threshold

**Symptom:** Every utterance returns `None` from all three confidence tiers.

**Possible causes:**

1. Skills not loaded yet: the pipeline filters to registered intents only. If no skills have loaded when the utterance arrives, there are no valid intents and nothing matches. Check that skills finish initializing before you send utterances.

2. Skills not in training data: the pre-built index is trained on a fixed skill set. Skills whose intents are absent from the index will not match. Use Adapt or Padatious for those skills, or [build a custom index](training.md).

3. Thresholds too high: lower `conf_low` to `0.05` or `0.0` to always return the top prediction.

4. Wrong language: the index covers 11 European languages. Utterances in other languages will not match reliably.

---

## Intent matched to wrong skill

**Symptom:** The pipeline returns the right intent name but the wrong skill ID.

**Cause:** Skill IDs come from the domain portion of the label (left-hand side of `:`). If two skills share a domain prefix, the pipeline may target the wrong skill.

**Fix:** Add the conflicting intent label to `ignore_intents` and let Adapt or Padatious handle it:

```json
{
  "intents": {
    "ovos_hierarchical_knn_pipeline": {
      "ignore_intents": ["conflicting_domain:conflicting_intent"]
    }
  }
}
```

---

## High memory usage

**Symptom:** OVOS runs out of memory on devices with less than 1 GB of RAM.

**Cause:** The FAISS IVF+PQ index (about 466 MB) and the Granite ONNX encoder (about 94 MB) load into RAM.

**Options:**

1. Build a pruned index covering only your installed skills (a smaller dataset gives a smaller index).
2. Increase swap space on the device.
3. Use a lighter encoder (`StaticModelEncoder` through `model2vec`): build a new index with it and point `model_path` at it.

---

## Pipeline stage fires unexpectedly

**Symptom:** The KNN pipeline matches an utterance that you expected another engine to handle.

**Fix:** Move the KNN stage to a later position in the pipeline list, after the deterministic engines:

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

---

## `renormalize` changes confidence scores unexpectedly

**Symptom:** After you enable `renormalize: true`, confidence scores are higher than expected.

**Explanation:** Renormalization redistributes the probability mass from filtered (unregistered) intents to the surviving intents. If many intents were filtered out, the surviving probability mass is small and gets scaled up significantly. This is correct behavior: the score reflects relative certainty among valid intents.

If you need raw scores for comparison, set `renormalize: false`.

---

## Startup is slow

**Symptom:** OVOS takes 5 or more seconds longer to start after you add the plugin.

**Cause:** The encoder ONNX model loads on the first inference call (lazy init), not at startup. The delay you see is likely the first utterance triggering the model load.

**Fix:** The model loads once and stays in memory, so later inferences are fast. If startup time itself is the issue, it is likely the HuggingFace download on first run. See the offline setup instructions in [Getting Started](getting-started.md).

---
[← Testing](testing.md) · [Home](index.md)
