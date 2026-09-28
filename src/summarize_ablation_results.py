"""Compatibility entry point; implementation lives in :mod:``evals.experiments.summarize_ablation_results``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.experiments.summarize_ablation_results", globals())

if __name__ == "__main__":
    _wrapped_module.main()
