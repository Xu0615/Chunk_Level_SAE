"""Compatibility entry point; implementation lives in :mod:``evals.reasoning.summarize_reason_feature_eval``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.reasoning.summarize_reason_feature_eval", globals())

if __name__ == "__main__":
    _wrapped_module.main()
