# Architecture

This document explains how the classifier and pipeline plugin work internally.

---

## Overview

```text
User utterance
      │
      ▼
 GraniteEncoder             ← ONNX model, ~94 MB
      │  float32 embedding (384-dim, L2-normalized)
      ▼
HierarchicalPairKNNClassifier
  ├─ L1 search  ──── top-N domains (skill namespaces)
  └─ L2 search  ──── top-k intents within those domains
      │  Wu-Lin probability distribution
      ▼
 Intent filter              ← only keep loaded-skill intents
      │  optional renormalization
      ▼
 IntentHandlerMatch         ← returned to OVOS
```

---

## Two-phase hierarchy

Labels in the index follow a `domain:intent` format (for example `weather:get_weather`, `ocp:play`). The classifier uses this structure to run two independent k-NN searches instead of one flat search over all intents.

### L1 — Domain search

The classifier queries the full FAISS index for the `k` nearest neighbors of the query embedding, then selects the top-`n` unique domains (left-hand side of `:`) by probability mass. Domain pre-filtering (`set_active_domains`) can restrict L1 to only the domains of currently loaded skills, which reduces noise when few skills are installed.

### L2 — Intent search

For each of the top-`n` domains selected in L1, the classifier searches the same FAISS index again, restricted to training vectors that belong to that domain. It retrieves the top-`k` neighbors within each domain and feeds them into the Wu-Lin estimator to produce a per-intent probability.

Final probabilities are the product of domain probability (L1) and intent probability (L2 given domain).

---

## Wu-Lin probability estimation

The classifier converts raw k-NN distances to probabilities using a pairwise kernel method derived from Wu & Lin (2004).

### Adaptive neighborhood (default)

`predict_proba` uses `_get_probabilities_with_adaptive_neighborhood`. Each neighbor at distance `d` contributes a Gaussian-decayed weight anchored at a reference distance `d_anchor` (typically the nearest-neighbor distance from the unfiltered global search):

```python
sigma = margin / gamma
centered = d - d_anchor
decay = np.exp(-(centered ** 2) / (2 * sigma ** 2))
decay = np.clip(decay, 1e-10, 1.0)
```

The neighbor pool is then further constrained:

1. Exact-match override: if the nearest neighbor distance is at or below `tau` (default 0.05), the nearest class wins 100%.
2. Adaptive margin: the classifier considers only neighbors within `[d_anchor, d_anchor + margin]`. It always includes neighbors closer than `d_anchor` and excludes neighbors beyond `d_anchor + margin`.
3. Dynamic k: the effective neighborhood size scales with the number of active classes and the distance spread, rather than being a fixed hyperparameter.

Anchoring on `d_anchor` from the unfiltered global search keeps the Gaussian's spread and the margin window invariant to domain-restricted candidate pools. `set_active_domains` does not change the shape of the kernel.

### Simple exponential form (`_get_probabilities`)

The library also implements a second, simpler kernel for reference and for callers that do not want the adaptive behavior:

```text
w = exp(-γ · d)
```

This is plain exponential decay with no anchor, no margin, and no dynamic k. `predict_proba` does not use it; the adaptive Gaussian variant above is the default.

### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `tau` | `0.05` | Exact-match threshold. Lower means more selective winner-takes-all. |
| `margin` | `0.10` | Shell half-width. Wider means smoother distributions. |
| `gamma` | `1.0` | Decay rate. Higher means sharper falloff with distance. |
| `anchor_to_global` | `True` | Anchor margin to the global nearest neighbor (not domain-filtered). Prevents bias when domains have very different densities. |

---

## FAISS index

The index is built with `IndexIVFPQ`:

- Quantizer: `IndexFlatIP` (inner product, equivalent to cosine similarity for L2-normalized vectors)
- IVF: 1024 Voronoi cells (`nlist`)
- PQ: 16 sub-quantizers, 8 bits each (`pq_m`)
- Search: 32 cells probed per query (`nprobe`)

For small datasets (fewer than `nlist × 256` vectors), the plugin uses a flat `IndexFlatIP` instead, which is exact but slower for large collections.

Vectors are L2-normalized before insertion, so inner product equals cosine similarity.

---

## Intent filtering and renormalization

After the classifier returns a probability distribution over all labels in the index, the pipeline applies two filters:

1. Registered-intent filter: only labels whose skill is currently loaded (through the Adapt or Padatious manifest) pass through. The pipeline discards unrecognized labels.
2. Ignore list: the pipeline discards labels listed in `ignore_intents`.

Special labels bypass the registered-intent filter but are gated by the session pipeline configuration:

| Label | Required pipeline stage |
|---|---|
| `ocp:play` | `ocp` in session pipeline |
| `common_query:common_query` | `common_query` in session pipeline |
| `stop:stop` | `stop` in session pipeline |

After filtering, if `renormalize=true`, the pipeline re-scales surviving probabilities to sum to 1.

The classifier (`HierarchicalPairKNNClassifier`) renormalizes internally over its full search context. This is intentional and stays on by default. The plugin-level `renormalize` setting, which re-scales again over only the surviving registered intents, defaults to `false`, because re-scaling a second time discards information about how confident the classifier was overall. It forces the visible candidates to sum to 1 even when none of them was a strong match. Enable plugin-level renormalization when you want the visible probabilities to highlight relative differences within the registered set.

---

## Dynamic intent synchronization

The pipeline maintains an allowlist of registered intents that it updates in real time:

- At startup: it queries the Adapt and Padatious services for all currently registered intent names.
- On skill load: it listens for `intent.service.intent-registered` and similar bus messages.
- On skill detach: it removes the skill's intents from the allowlist.

This means intents from skills that load after the pipeline starts become available immediately, and the pipeline blocks intents from unloaded skills immediately.

---

## File structure of a built index

```text
index_dir/
├── index.faiss              ← FAISS IVF+PQ index
├── label_ids.npy            ← label-to-class-ID mapping per hierarchy level
├── class_names.npy          ← class name array per hierarchy level
├── class_to_train_ids.pkl   ← training vector IDs per class (for scoped L2 search)
└── meta.pkl                 ← classifier hyperparameters + model_path
```

The encoder model is stored separately, either inside the same directory or referenced by `meta.pkl → model_path`.

---

## Memory and latency budget (Raspberry Pi 4)

| Component | Size | Load time | Inference latency |
|---|---|---|---|
| Granite ONNX encoder | ~94 MB | ~2 s | ~30 ms/utterance |
| FAISS IVF+PQ index | ~233 MB | ~1 s | <5 ms/query |
| **Total** | **~345 MB storage / ~320 MB RAM**[^total] | **~3 s** | **~35 ms** |

[^total]: The storage figure includes auxiliary files (`meta.pkl`, `label_ids.npy`, `class_names.npy`, `class_to_train_ids.pkl`) on top of the encoder and FAISS index. The full on-disk footprint of the published snapshot is about 560 MB. The RAM figure is what is mapped in memory once the pipeline is loaded.

These figures are approximate, for a Pi 4 with 4 GB RAM. Actual performance varies with the number of loaded skills and the FAISS `nprobe` setting.

---
[← Configuration](configuration.md) · [Home](index.md) · [API Reference →](api-reference.md)
