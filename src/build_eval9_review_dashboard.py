"""Compatibility entry point; implementation lives in :mod:``evals.code_page.build_eval9_review_dashboard``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.code_page.build_eval9_review_dashboard", globals())

if __name__ == "__main__":
    _wrapped_module.main()
