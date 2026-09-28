"""Compatibility entry point; implementation lives in :mod:``evals.code_page.build_code_page_retrieval_demo``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.code_page.build_code_page_retrieval_demo", globals())

if __name__ == "__main__":
    _wrapped_module.main()
