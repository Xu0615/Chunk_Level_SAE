"""Compatibility entry point; implementation lives in :mod:``evals.reasoning.materialize_reason_token_features``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.reasoning.materialize_reason_token_features", globals())

if __name__ == "__main__":
    _wrapped_module.main()
