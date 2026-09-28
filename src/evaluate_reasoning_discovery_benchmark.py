"""Compatibility entry point; implementation lives in :mod:``evals.reasoning.evaluate_reasoning_discovery_benchmark``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.reasoning.evaluate_reasoning_discovery_benchmark", globals())

if __name__ == "__main__":
    _wrapped_module.main()
