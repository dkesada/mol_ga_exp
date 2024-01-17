from .general_ga import run_ga_maximization
from .restart_ga import run_rga_maximization
from .preconfigured_gas import default_ga, restart_ga

__all__ = ["run_ga_maximization", "run_ga_maximization", "default_ga", "restart_ga"]

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("mol_ga")
    except PackageNotFoundError:
        # package is not installed
        pass
except ModuleNotFoundError:
    pass  # Python < 3.8
