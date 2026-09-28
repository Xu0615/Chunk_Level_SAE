"""Compatibility entry point; implementation lives in :mod:``evals.document_linking.evaluate_lexical_controlled_document_linking``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.document_linking.evaluate_lexical_controlled_document_linking", globals())

if __name__ == "__main__":
    _wrapped_module.main()
