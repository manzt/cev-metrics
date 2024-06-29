"""cev-metrics: A Python package for computing the metrics used in the CEV project."""

import importlib.metadata

__version__ = importlib.metadata.version("cev-metrics")

from cev_metrics._rust import (
    confusion,
    confusion_and_neighborhood,
    graph_stats,
    neighborhood,
)

__all__ = ["confusion", "confusion_and_neighborhood", "neighborhood", "graph_stats"]
