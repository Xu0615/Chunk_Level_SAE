"""Compatibility entry point; implementation lives in :mod:``evals.tools.backfill_tensorboard_dashboard``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.tools.backfill_tensorboard_dashboard", globals())

if __name__ == "__main__":
    _wrapped_module.main()
