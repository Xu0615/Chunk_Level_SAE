"""Compatibility entry point; implementation lives in :mod:``evals.code_page.profile_code_page_sample``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.code_page.profile_code_page_sample", globals())

if __name__ == "__main__":
    _wrapped_module.main()
