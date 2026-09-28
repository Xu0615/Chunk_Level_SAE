"""Compatibility entry point; implementation lives in :mod:``evals.reasoning.judge_reason_scenario_pairs``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.reasoning.judge_reason_scenario_pairs", globals())

if __name__ == "__main__":
    _wrapped_module.main()
