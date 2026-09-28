"""Compatibility entry point for :mod:`evals.rfve.analyze_training_fidelity`."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.rfve.analyze_training_fidelity", globals())

if __name__ == "__main__":
    _wrapped_module.main()
