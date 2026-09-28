# Training Chunk-Level SAEs

[← README](../README.md) · [中文](training_zh.md) · [Methods](methods.md)

## Choose a starting point

| Goal | Entry point | Required resources |
| :--- | :--- | :--- |
| Understand the API and verify installation | `examples/train_synthetic.py` | CPU; no downloads |
| Exercise extraction and real SAE training | `configs/quickstart_2b.env` | Linux, a CUDA GPU, Qwen3.5-2B weights, dataset access |
| Run the supplied 9B research configuration | `configs/paper_9b.env` | Linux, 8 CUDA GPUs, substantial activation storage |

The model-specific shell launchers assume GNU utilities, `flock`, and CUDA. Their unmodified defaults are large-cluster settings. Start with the reduced configuration when bringing up a new machine.

## Environment

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for your driver from [the official installation selector](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -e '.[dev]'
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The source uses Qwen3.5-specific Transformers internals and pins `transformers==5.14.1`. The base package supports the CPU example and standard extraction path. Optional kernel packages are isolated in an extra:

```bash
# Optional, on a compatible Linux CUDA toolchain:
python -m pip install -e '.[cuda]'
export CHUNK_SAES_ENABLE_FLA=1
```

Flash Linear Attention is disabled by default by the extractor; install and enable it only when your environment supports it. The optional `spark` extra is for the code-page data preparation workflow and is not needed for SAE training.

### Local model weights

```bash
hf download Qwen/Qwen3.5-2B-Base --local-dir models/Qwen3.5-2B-Base
```

Use a complete local checkpoint directory. The extractor's efficient truncated loading path expects the model config and safetensors index; incomplete directories can cause loading failures or fall back to a larger full-model load. Model-specific launchers are provided for Qwen3.5-2B, 9B, and 35B-A3B. Supporting another model family requires adapting `chunk_saes/modeling.py`.

## Data and cache protocol

The default corpus is [EleutherAI/the_pile_deduplicated](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated), read through Hugging Face streaming at revision:

```text
fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb
```

The source's physical split is `train`. The pipeline derives logical train/validation/test splits from normalized content hashes, with proportions 80%/10%/10% and split seed 42. Sampling and document order have separate seeds recorded in the cache metadata.

The `corpus` stage verifies stream access; it does not download or materialize the whole corpus. The `cache` stage samples adjacent chunk pairs and independently encodes each chunk. A cache includes token activations, pooled chunk means, pair identities, provenance, and rank manifests.

The standard launcher builds **training and validation caches**. A held-out test cache is an additional extraction step when an evaluation requires it.

## First run: reduced 2B configuration

Use a fresh activated shell when switching between configuration files. Source the file from the repository root; apply any local overrides **after** sourcing it.

```bash
source .venv/bin/activate
source configs/quickstart_2b.env

# Optional local overrides:
# export MODEL=/absolute/path/to/Qwen3.5-2B-Base
# export PIPELINE_ROOT=/absolute/path/to/your/run
# export CUDA_VISIBLE_DEVICES=1

bash run_scripts/train_qwen35_2b_chunk_saes.sh corpus
bash run_scripts/train_qwen35_2b_chunk_saes.sh cache
bash run_scripts/train_qwen35_2b_chunk_saes.sh train
```

This trains `mean,cross` using 32,768 training occurrences, 8,192 validation occurrences, width 4,096, `K=32`, and 128 steps of global batch size 256. It is a small integration run for checking your setup.

To train nested Joint-Chunk on that cache:

```bash
SAE_MODES=joint_chunk JOINT_CROSS_PREFIX=2048 JOINT_CHUNK_ALPHA=0.25 \
RUN_NAME=quickstart_joint_chunk \
bash run_scripts/train_qwen35_2b_chunk_saes.sh train
```

Use a different `RUN_NAME` for a new objective or hyperparameter experiment. The quick-start and research configuration files set `RUN_EVALUATION=0`, because downstream evaluation requires separately prepared artifacts.

### Pipeline stages

| Stage | Action |
| :--- | :--- |
| `corpus` | Check the Hugging Face streaming source |
| `cache` | Build training and validation activation caches |
| `train` | Train requested modes from existing caches |
| `eval` | Run the broader evaluation orchestrator, after its inputs are prepared |
| `all` | Run corpus verification, caching, and training; evaluation runs if enabled |

Running `cache` again with compatible complete artifacts allows the extractor to verify and reuse them. Keep overwrite flags off when resuming an existing run. Read any cache-compatibility error before deciding to regenerate artifacts.

## Research-scale 9B configuration

Start in a fresh activated shell so pilot settings do not carry over.

```bash
hf download Qwen/Qwen3.5-9B-Base --local-dir models/Qwen3.5-9B-Base
source configs/paper_9b.env

bash run_scripts/train_qwen35_9b_chunk_saes.sh corpus
bash run_scripts/train_qwen35_9b_chunk_saes.sh cache
bash run_scripts/train_qwen35_9b_chunk_saes.sh train

RUN_NAME=paper_9b_joint_chunk \
bash run_scripts/train_qwen35_9b_joint_chunk_nested.sh
```

The baseline/Mean/Cross run and the Joint run reuse the same activation caches. The nested Joint launcher explicitly sets an eight-GPU, 32,000-global-batch, 31,250-step run. Use the general model launcher with explicit overrides for a smaller Joint experiment, as in the 2B example above.

### Configuration reference

| Variable | Role | 9B recipe |
| :--- | :--- | :--- |
| `MODEL` | Local Qwen checkpoint directory | `models/Qwen3.5-9B-Base` |
| `LAYER` | Zero-based transformer layer | `21` |
| `GPU_COUNT` | Processes launched by `torchrun` | `8` |
| `CUDA_VISIBLE_DEVICES` | Physical devices available to the run | `0,1,2,3,4,5,6,7` |
| `SAE_MODES` | Comma-separated objectives | `token,temporal,mean,cross` |
| `TARGET_TRAIN_TOKENS` | Sampled training occurrences | `1000000000` |
| `TARGET_VALIDATION_TOKENS` | Sampled validation occurrences | `10000128` |
| `CHUNK_LENGTHS` | Allowed chunk lengths | `32,64,128,256,512` |
| `SAE_WIDTH` / `K` | Dictionary size / average sparsity budget | `65536` / `128` |
| `GLOBAL_BATCH_SIZE` / `TRAIN_STEPS` | Global occurrence batch / updates | `32000` / `31250` |
| `LR` / `WARMUP_STEPS` | Learning rate / warm-up | `1e-4` / `200` |
| `MIN_LR_RATIO` | Learning-rate floor ratio | `0.1` |
| `AUXK_ALPHA` | Auxiliary reconstruction loss weight | `0.0625` |
| `JOINT_CROSS_PREFIX` | Number of shared predictive features | `32768` in the nested Joint launcher |
| `JOINT_CHUNK_ALPHA` | Relative cross-objective weight | `0.25` |
| `PIPELINE_ROOT` | Persistent artifacts and logs | `qwen/qwen35-9b-base/pile_proportional_exact_occurrence` |
| `RUN_NAME` | Training-run subdirectory | A distinct name per experiment |

All model launchers source [`src/train_chunk_saes_common.sh`](../src/train_chunk_saes_common.sh). That file is the complete reference for supported environment variables. CLI details are also available without model downloads:

```bash
python src/extract_streaming_chunk_activations.py --help
python src/train_chunk_saes.py --help
```

### Exact-coverage constraints

```text
TRAIN_STEPS × GLOBAL_BATCH_SIZE = TARGET_TRAIN_TOKENS
GLOBAL_BATCH_SIZE must be divisible by GPU_COUNT
the cache rank partition must match the training world size
```

The budget counts token occurrences. Chunk representations retain their occurrence weights even when repeated inputs are deduplicated internally. Adjust the budget, batch size, and step count together. Regenerate a compatible cache when changing its data protocol or rank partition; reusing the same directory name does not make a cache compatible.

## Checkpoints and monitoring

For each mode, the training directory has the following layout:

```text
${PIPELINE_ROOT}/
├── data/
│   ├── layer${LAYER}_train_cache_exact${TARGET_TRAIN_TOKENS}/
│   └── layer${LAYER}_validation_cache_exact${TARGET_VALIDATION_TOKENS}/
├── logs/
├── runs/                         # TensorBoard events
└── checkpoints/${RUN_NAME}/
    └── mean/                     # Or cross, joint_chunk, token, temporal
        ├── config.json
        ├── run_config.json
        ├── metrics.jsonl
        ├── sae.safetensors
        ├── complete.json
        └── checkpoints/
            ├── best/             # Validation-selected, inference weights
            └── latest/           # Optimizer + training state for resuming
```

`complete.json` marks a completed mode. The final root weights and `checkpoints/best` serve different selection policies; record which you use for evaluation.

```bash
tensorboard --logdir "${PIPELINE_ROOT}/runs"
```

To resume an interrupted run, restore the same environment, run name, cache, modes, dictionary settings, and world size, then:

```bash
RESUME=1 bash run_scripts/train_qwen35_2b_chunk_saes.sh train
```

For 9B, use its corresponding launcher. Resume requires the complete `checkpoints/latest` training state. The best checkpoint is weights-only and cannot restore the optimizer or data position. A completed run may be recognized as complete rather than trained again; choose a new run name for a new experiment.

## Resource planning

**Activation storage dominates at research scale.** The cache retains token activations even when training only Mean/Cross/Joint. A useful lower-order estimate for the token tensor alone is:

```text
sampled occurrences × hidden dimension × bytes per activation
```

For example, `1e9 × 4096 × 2` is approximately **8.2 TB (decimal)** before pooled means, metadata, validation caches, replicas, model weights, and checkpoints. Measure a pilot cache on your system before allocating a full run. Streaming the corpus avoids full local text materialization; it does not eliminate activation storage.

Memory controls are stage-specific:

| Stage / pressure | Controls to reduce |
| :--- | :--- |
| Backbone extraction GPU memory | `FORWARD_BATCH_SIZE`, `FORWARD_TOKEN_BUDGET` |
| Training GPU memory | `GLOBAL_BATCH_SIZE`, `LOADER_GPU_SHARDS`, number of modes trained together |
| Host RAM and prefetch pressure | `LOADER_PREFETCH_SHARDS`, `LOADER_PREFETCH_WORKERS`, `LOADER_PREFETCH_BATCHES` |
| Cache writer memory | `CACHE_SHARD_TOKENS`, `WRITER_BATCH_TOKENS`, pending write settings |
| Dataset startup/shuffle memory | `STREAM_SHUFFLE_BUFFER`, tokenizer batch settings |

Changing global batch size requires a corresponding step-count adjustment to preserve exact coverage. Lowering dictionary width changes the experiment. The original model launchers contain high-throughput extraction/prefetch defaults intended for large-memory GPUs; the reduced config overrides these explicitly. No minimum VRAM or throughput guarantee has been established for every supported model.

## Troubleshooting

| Symptom | Resolution |
| :--- | :--- |
| `flock` missing, GNU `date`/`readlink` option errors | Run the production launchers on Linux. The CPU example is the portable starting point. |
| PyTorch reports CUDA unavailable | Install a driver-compatible CUDA wheel and check device visibility. |
| Qwen3.5 import or config errors | Use the pinned Transformers version and a complete local base-model checkpoint. |
| OOM during cache extraction | Reduce the extraction batch/token budgets; check that truncated model loading is active. |
| OOM during SAE training | Reduce prefetch/on-GPU shards, train fewer modes together, or reduce global batch and adjust steps. |
| Exact-coverage error | Check `steps × global batch`, cache occurrence count, and cache/training world size. |
| Cache manifest mismatch | Match dataset revision, tokenizer/model, layer, splits, sampling settings, and rank layout. Use a new cache directory when the protocol changes. |
| Existing output directory | Resume the matching run with `RESUME=1`, or select a new `RUN_NAME`. |
| Cross RFVE reference missing | Construct a matched dense-reference manifest; see [evaluation](evaluation.md#rfve-and-the-cross-reference). |
| Evaluation missing data or judge outputs | Prepare task-specific artifacts first; training configs leave automatic evaluation disabled. |

## What has been validated

The release preparation checks include the CPU test suite, actual Mean/Cross/Joint optimization on synthetic data, checkpoint reload consistency, Python entry-point help, shell syntax, and package construction. CPU CI repeats those checks. Full Qwen3.5 CUDA extraction and billion-occurrence training need separate validation on the intended compute environment.
