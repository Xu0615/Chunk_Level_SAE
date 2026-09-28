"""Compatibility entry point; implementation lives in :mod:``evals.tools.benchmark_forward_batch``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.tools.benchmark_forward_batch", globals())

if __name__ == "__main__":
    _wrapped_module.main()
