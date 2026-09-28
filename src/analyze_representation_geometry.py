"""Compatibility entry point; implementation lives in :mod:``evals.supervised_transfer.analyze_representation_geometry``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.supervised_transfer.analyze_representation_geometry", globals())

if __name__ == "__main__":
    _wrapped_module.main()
