"""Compatibility entry point; implementation lives in :mod:``evals.rfve.train_cross_dense_reference``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.rfve.train_cross_dense_reference", globals())

if __name__ == "__main__":
    _wrapped_module.main()
