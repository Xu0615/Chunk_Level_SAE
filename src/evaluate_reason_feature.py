"""Compatibility entry point; implementation lives in :mod:``evals.reasoning.evaluate_reason_feature``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.reasoning.evaluate_reason_feature", globals())

if __name__ == "__main__":
    _wrapped_module.main()
