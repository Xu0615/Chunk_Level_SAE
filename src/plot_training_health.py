"""Compatibility entry point for :mod:`evals.rfve.plot_training_health`."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.rfve.plot_training_health", globals())

if __name__ == "__main__":
    _wrapped_module.main()
