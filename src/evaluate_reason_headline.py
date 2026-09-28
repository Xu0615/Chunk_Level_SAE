"""Compatibility entry point; implementation lives in :mod:``evals.reasoning.evaluate_reason_headline``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.reasoning.evaluate_reason_headline", globals())

if __name__ == "__main__":
    _wrapped_module.main()
