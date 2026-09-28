"""Compatibility entry point; implementation lives in :mod:``evals.autointerp.prepare_autointerp_data``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.autointerp.prepare_autointerp_data", globals())

if __name__ == "__main__":
    _wrapped_module.main()
