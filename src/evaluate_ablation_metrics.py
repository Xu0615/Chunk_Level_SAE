"""Compatibility entry point; implementation lives in :mod:``evals.experiments.evaluate_ablation_metrics``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.experiments.evaluate_ablation_metrics", globals())

if __name__ == "__main__":
    _wrapped_module.main()
