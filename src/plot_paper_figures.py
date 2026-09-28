"""Compatibility entry point for :mod:`evals.visualization.plot_paper_figures`."""
from evals._compat import expose as _expose

_wrapped_module = _expose("evals.visualization.plot_paper_figures", globals())

if __name__ == "__main__":
    _wrapped_module.main()
