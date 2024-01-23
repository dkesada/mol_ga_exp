from .general_ga import run_ga_maximization
from .ga_controller import GAController
from .restart_ga import run_trga_maximization
from .trga_controller import TRGAController
from .preconfigured_gas import default_ga, restart_ga

__all__ = ["run_ga_maximization", "run_trga_maximization", "default_ga", "restart_ga", "GAController", "TRGAController"]

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("mol_ga")
    except PackageNotFoundError:
        # package is not installed
        pass
except ModuleNotFoundError:
    pass  # Python < 3.8
