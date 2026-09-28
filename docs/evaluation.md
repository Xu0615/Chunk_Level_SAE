# Evaluation guide

[← README](../README.md) · [Training](training.md) · [Methods](methods.md)

The repository separates feature quality from downstream utility. Evaluate the dictionary itself, then test whether its features support retrieval, transfer, and reasoning. Each task consumes explicit checkpoints and artifacts; preserve their manifests when comparing methods.

## Available components

| Question | Components | Additional inputs |
| :--- | :--- | :--- |
| How much target structure is reconstructed? | [`evals/rfve`](../src/evals/rfve): RFVE, cosine, training health | Completed checkpoints, metrics, activation caches; matched dense Cross reference for RFVE |
| How is the dictionary used? | [`evaluate_dictionary_health.py`](../src/evals/rfve/evaluate_dictionary_health.py), [`dictionary_utilization`](../src/evals/dictionary_utilization) | Checkpoints, held-out activation/evidence artifacts |
| Do features persist across related spans? | Adjacent feature consistency and [`feature_dynamics`](../src/evals/feature_dynamics) | Aligned chunk-pair evidence and feature statistics |
| Are features interpretable or high level? | [`autointerp`](../src/evals/autointerp), high-level analysis in `dictionary_utilization`, [`semantic_invariance`](../src/evals/semantic_invariance) | Prepared evidence; task-specific explanation/judge results and controlled examples |
| Do features link semantically related documents? | [`document_linking`](../src/evals/document_linking) | Evidence data and manifest, model, SAE checkpoints |
| Do features transfer across domains or label budgets? | [`supervised_transfer`](../src/evals/supervised_transfer) | Prepared arXiv benchmark, extracted probe features, labels and splits |
| Do features capture reasoning beyond surface cues? | [`reasoning`](../src/evals/reasoning) | Reasoning examples, selected features, task-specific annotations/judge artifacts |
| What changes under the Joint objective? | [`joint_extension`](../src/evals/joint_extension), [`experiments`](../src/evals/experiments) | Matching baseline/Joint checkpoints and evaluation data |
| How are results assembled for a paper? | [`visualization`](../src/evals/visualization) | Completed task result files in the expected directory layout |

The `evals/steering` and `evals/high_level` namespaces currently contain only package markers. High-level feature analysis has implementations in the modules listed above; a complete standalone causal-steering runner is not included.

The code-page workflow under `evals/code_page` additionally needs its external data and, for Spark preparation, a configured Spark environment. It is separate from the core SAE training path.

## Inspect task interfaces

After installing the repository in editable mode:

```bash
python -m evals.rfve.train_cross_dense_reference --help
python -m evals.rfve.analyze_training_fidelity --help
python -m evals.rfve.evaluate_reconstruction_cosine --help
python -m evals.document_linking.evaluate_lexical_controlled_document_linking --help
python -m evals.supervised_transfer.run_linear_probes --help
```

Prefer these canonical modules to the compatibility wrappers at the top level of `src/`. Different tasks require different schemas; a JSONL file with arbitrary text is not automatically a valid benchmark input.

## RFVE and the Cross reference

Cross-Chunk predicts an independently encoded neighbor. Its attainable predictive fidelity differs from self-reconstruction, so RFVE uses a matched dense predictor as a reference. The reference trainer performs a capacity sweep, selects on validation data, and evaluates on an independent test split.

The ordinary cache stage produces **train and validation** caches. Prepare a compatible logical test cache with `src/extract_streaming_chunk_activations.py --logical-split test`, keeping the model, layer, corpus revision, split seed, chunk lengths, and other protocol settings aligned. Use that command's `--help` for the full extraction interface.

Once all three caches are complete, the reference command has this form. Replace the paths with your artifacts:

```bash
python -m evals.rfve.train_cross_dense_reference \
  --train-cache-dir /path/to/train_cache \
  --validation-cache-dir /path/to/validation_cache \
  --test-cache-dir /path/to/test_cache \
  --output-dir /path/to/cross_dense_reference \
  --monitor-world-size 8 \
  --device cuda:0
```

Set `--monitor-world-size` to the matching cache/experiment world size. For a publication comparison, also pass the Cross SAE's saved `activation_scale` via `--activation-scale` so its numerical conditioning matches the reference training.

Then summarize the completed runs:

```bash
python -m evals.rfve.analyze_training_fidelity \
  --sae-root /path/to/baseline_mean_cross_run \
  --joint-sae-root /path/to/joint_run \
  --checkpoint-selection best \
  --cross-reference-manifest /path/to/cross_dense_reference/manifest.json \
  --output /path/to/evaluation/training_fidelity.json
```

Omit `--joint-sae-root` if no Joint run exists. Supply a run root containing mode subdirectories, rather than a single mode's safetensors file.

Some original launcher settings retain a scalar Cross reference for historical TensorBoard audits. The reporting command rejects a Cross scalar unless `--allow-audit-cross-scalar` is explicitly set. Build a matched manifest for new experiments instead of treating a historical scalar as a newly measured reference.

## Checkpoint selection and comparable results

Record the following with each result:

- Backbone checkpoint, tokenizer, layer, corpus revision, split/sampling seeds, and cache manifest.
- SAE mode, dictionary width, `K`, training budget, and world size.
- For Joint-Chunk, layout, prefix width, alpha, and the decoder readout used.
- Validation-selected (`best`) or final checkpoint policy.
- Benchmark version, feature subset, controls, judge setup where applicable, and uncertainty estimates.

For token-level baselines and chunk-level models, preserve the corresponding input aggregation rules. Use the same held-out source and matching evaluation protocol across methods. The analysis helpers include schema and manifest checks to catch inconsistent artifacts.

## Evaluation orchestrator

`src/eval_chunk_saes_common.sh` coordinates a wider experiment suite and expects a populated artifact workspace. Review its paths and task switches before running the model launcher with the `eval` stage. The quick-start and research training configs set `RUN_EVALUATION=0` so missing evaluation inputs do not interrupt first-time training.

Paper-figure scripts assemble existing result files. Running a plotting script does not execute all underlying experiments. This code release does not include the pretrained dictionaries, full benchmark outputs, or external explanation/judge service required for every paper analysis.
