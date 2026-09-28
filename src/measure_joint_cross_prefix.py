"""Compatibility entry point; implementation lives in :mod:``evals.experiments.measure_joint_cross_prefix``."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.experiments.measure_joint_cross_prefix", globals())

if __name__ == "__main__":
    _wrapped_module.main()
