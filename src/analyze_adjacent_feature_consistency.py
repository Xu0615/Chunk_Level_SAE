"""Compatibility entry point; implementation lives in :mod:``evals.dictionary_utilization.analyze_adjacent_feature_consistency``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.dictionary_utilization.analyze_adjacent_feature_consistency", globals())

if __name__ == "__main__":
    _wrapped_module.main()
