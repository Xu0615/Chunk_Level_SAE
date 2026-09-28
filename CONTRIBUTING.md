# Contributing

Thank you for helping improve Chunk-Level SAEs. Reproducible changes to training, evaluation, and documentation are welcome.

## Development setup

Use Python 3.12+, install the appropriate PyTorch build, then:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/train_synthetic.py --steps 12 --output-dir outputs/contribution-check
python -m build
```

The example expects an empty output directory. Use a fresh path for each verification run. CPU tests cover model behavior, cache/sample protocols, checkpoints, and evaluation helpers; CUDA-specific cases skip when CUDA is unavailable.

Additional checks used in CI:

```bash
python -m compileall -q src examples
for file in run_scripts/*.sh src/*.sh; do bash -n "$file"; done
python src/train_chunk_saes.py --help
python src/extract_streaming_chunk_activations.py --help
```

## Research code changes

Keep the input/target protocol explicit. Changes to sampling, chunk forward context, occurrence weighting, normalization, BatchTopK selection, or checkpoint semantics can change the scientific comparison. Explain such changes and add focused tests for the behavior they affect.

New evaluation code should live in the relevant package under `src/evals/`. Keep existing compatibility entry points functional when moving modules. Use paths supplied through arguments or environment variables; avoid machine-specific paths, credentials, or unrelated cluster process management.

## Reporting a problem

Include the command, mode, model/layer, environment versions, GPU count, cache world size, occurrence budget, and traceback. For an exact-coverage issue, include global batch size and step count. Share small synthetic reproductions when possible. Do not attach access tokens, private data, model weights, or multi-gigabyte caches.

For a pull request, describe the problem, resulting behavior, and validation performed. Distinguish CPU checks from actual CUDA or distributed experiments, and include benchmark provenance with any performance claim.
