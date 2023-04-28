try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("cev-metrics")
except PackageNotFoundError:
    __version__ = "uninstalled"

from cev_metrics._rust import confusion, confusion_and_neighborhood, neighborhood
