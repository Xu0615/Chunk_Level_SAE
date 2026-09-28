"""Compatibility entry point; implementation lives in :mod:``evals.feature_dynamics.plot_cross_feature_semantic_manifold``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.feature_dynamics.plot_cross_feature_semantic_manifold", globals())

if __name__ == "__main__":
    _wrapped_module.main()
