# Methods and training protocol

[← README](../README.md) · [Training](training.md) · [Evaluation](evaluation.md)

## From a document to a chunk pair

The sampler chooses adjacent, non-overlapping text chunks A and B from the same document. Each chunk is forwarded independently through the frozen backbone, with its own attention context and positions. At the selected layer, token activations are averaged within each chunk to form vectors `mu_A` and `mu_B`.

This independent-forward protocol matters for the Cross objective: the representation of A must not already contain B through a shared forward pass. Replacing it with pooling over one concatenated document changes the experiment.

The shared training stream uses several chunk lengths and weights sampled pairs by their token occurrences. Content hashes determine logical train/validation/test membership so identical normalized content cannot enter different logical splits.

## Sparse encoder

The encoder centers scaled inputs using a learned input bias, applies a linear map and ReLU, then selects activations with BatchTopK. During training, the global budget is approximately `batch_size × K`; individual examples can use different numbers of features. At inference, the saved moving-average threshold selects features independently for each example.

The checkpoint stores the input center, decoder center(s), activation scale, and inference threshold. Keep them with the weights. The input center and output center are separate parameters, which is especially important when the target is an adjacent chunk.

## Mean-Chunk

```text
z_A    = sparse_encode(mu_A)
mu_A^  = W_dec z_A + b_mean
```

Mean-Chunk reconstructs the current chunk's mean activation. The target aggregates evidence across the span, encouraging features to describe the representation of a phrase or passage.

The reverse view uses B as both input and target. In code, this is `mode="mean"` and the model is `BatchTopKSAE`.

## Cross-Chunk

```text
z_A    = sparse_encode(mu_A)
mu_B^  = W_dec z_A + b_cross
```

Cross-Chunk predicts the independently encoded adjacent chunk. The default `DIRECTION_POLICY=both` trains A→B and B→A. The target emphasizes information that can be predicted across nearby spans.

In code, this is `mode="cross"`, also using `BatchTopKSAE`; the training view supplies the adjacent target. A cross-prediction loss is not directly comparable to a self-reconstruction loss. Use a matched attainable reference for the RFVE analysis described in [the evaluation guide](evaluation.md#rfve-and-the-cross-reference).

## Joint-Chunk: a nested shared dictionary

Let the dictionary width be `m` and the cross prefix be `h`, with `0 < h < m`.

```text
z_A    = sparse_encode(mu_A)                 # m coordinates
mu_A^  = W_dec[:, :m] z_A       + b_mean     # full dictionary
mu_B^  = W_dec[:, :h] z_A[:h]   + b_cross    # shared prefix
```

The first `h` dictionary atoms participate in both reconstruction and prediction. The remaining atoms participate only in reconstruction. The two readouts share the decoder columns in the prefix and have separate output biases.

The main task loss combines baseline-normalized mean and cross errors:

```text
L_joint = (SSE_mean / baseline_mean
           + alpha × SSE_cross / baseline_cross) / (1 + alpha)
```

The production trainer also applies its configured auxiliary objective and optimization controls. The supplied nested 9B recipe uses `m=65536`, `h=32768`, and `alpha=0.25`.

Set:

```bash
export SAE_MODES=joint_chunk
export JOINT_CROSS_PREFIX=32768
export JOINT_CHUNK_ALPHA=0.25
```

Two similarly named settings have different roles:

| Setting | Meaning |
| :--- | :--- |
| `SAE_MODES=joint_chunk` | Select the Joint-Chunk architecture/objective |
| `JOINT_CROSS_PREFIX>0` | Select its nested shared decoder layout |
| `JOINT_CROSS_PREFIX=0` | Select the legacy independent two-decoder layout |
| Trainer flag `--joint-modes` | Train multiple requested SAE modes over a shared loader; does not itself select Joint-Chunk |

The class `JointChunkSAE` supports both decoder layouts so older checkpoints remain readable. Record the layout and prefix size when reporting results.

## Baselines

- **Token BatchTopK** (`token`) reconstructs individual token activations.
- **Temporal** (`temporal`) combines token reconstruction with a temporally structured feature objective. Its high-feature fraction, reconstruction weights, contrastive temperature, and loss weight are exposed in the model launchers.

Use the same backbone, layer, sampled occurrences, dictionary width, sparsity settings, and validation protocol when comparing objectives.

## Exact occurrence accounting

A budget of one billion means **one billion sampled token occurrences**. It does not mean one billion unique tokens or one billion chunks. A chunk mean can receive multiple occurrence weights; the loader can deduplicate equal chunk inputs for computation while preserving those weights in the objective and sparsity accounting.

The supplied trainer enforces:

```text
number of steps × global batch size = cache training occurrences
cache rank partition = training world size
```

This is why changing GPU count, cache generation settings, or the training budget requires a compatible cache and configuration. See [resource planning](training.md#resource-planning) before scaling up.

## Source map

| Component | Source |
| :--- | :--- |
| Sparse encoder, decoders, checkpoint API | [`sae.py`](../src/chunk_saes/sae.py) |
| Frozen feature subsets for evaluation | [`frozen_sae.py`](../src/chunk_saes/frozen_sae.py) |
| Independent chunk forward passes | [`forward.py`](../src/chunk_saes/forward.py) |
| Qwen3.5 activation extraction | [`modeling.py`](../src/chunk_saes/modeling.py) |
| Streaming sampling | [`streaming_plan.py`](../src/chunk_saes/streaming_plan.py) |
| Cache tensors and manifests | [`cache.py`](../src/chunk_saes/cache.py) |
| Training objectives and distributed execution | [`train_chunk_saes.py`](../src/train_chunk_saes.py) |
