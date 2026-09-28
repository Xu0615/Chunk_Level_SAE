"""Compatibility entry point; implementation lives in :mod:``evals.code_page.extract_code_page_sae_features``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.code_page.extract_code_page_sae_features", globals())

if __name__ == "__main__":
    _wrapped_module.main()
