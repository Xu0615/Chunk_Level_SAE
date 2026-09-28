"""Compatibility entry point; implementation lives in :mod:``evals.experiments.train_sequence_cross_ablation``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.experiments.train_sequence_cross_ablation", globals())

if __name__ == "__main__":
    _wrapped_module.main()
