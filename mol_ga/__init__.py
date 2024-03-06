from .ga_controller import GAController
from .trga_controller import TRGAController

__all__ = ["GAController", "TRGAController"]

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("mol_ga")
    except PackageNotFoundError:
        # package is not installed
        pass
except ModuleNotFoundError:
    pass  # Python < 3.8
