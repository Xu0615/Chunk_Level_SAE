"""Compatibility entry point; implementation lives in :mod:``evals.supervised_transfer.run_linear_probes``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.supervised_transfer.run_linear_probes", globals())

if __name__ == "__main__":
    _wrapped_module.main()
