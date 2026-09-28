"""Compatibility entry point; implementation lives in :mod:``evals.supervised_transfer.extract_probe_features``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.supervised_transfer.extract_probe_features", globals())

if __name__ == "__main__":
    _wrapped_module.main()
