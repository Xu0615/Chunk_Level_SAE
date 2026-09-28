"""Compatibility entry point for :mod:`evals.rfve.evaluate_reconstruction_cosine`."""
from evals._compat import expose as _expose

_wrapped_module = _expose(
    "evals.rfve.evaluate_reconstruction_cosine",
    globals(),
)

if __name__ == "__main__":
    _wrapped_module.main()
