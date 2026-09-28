"""Compatibility entry point; implementation lives in :mod:``evals.autointerp.materialize_chunk_autointerp_exact100_data``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.autointerp.materialize_chunk_autointerp_exact100_data", globals())

if __name__ == "__main__":
    _wrapped_module.main()
