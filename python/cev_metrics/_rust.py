from dataclasses import dataclass

import numpy as np
import pandas as pd

from cev_metrics.cev_metrics import (
    Graph,
    _confusion,
    _confusion_and_neighborhood,
    _neighborhood,
)


def _prepare_xy(df: pd.DataFrame) -> Graph:
    points = df[["x", "y"]].values

    if points.dtype != np.float64:
        points = points.astype(np.float64)

    return Graph(points)


def _prepare_labels(df: pd.DataFrame) -> np.ndarray:
    codes = df["label"].cat.codes.values

    if codes.dtype != np.int16:
        codes = codes.astype(np.int16)

    return codes


def confusion(df: pd.DataFrame):
    """Returns confusion matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns `x`, `y` and `label`. `label` must be a categorical.
    """
    return _confusion(_prepare_xy(df), _prepare_labels(df))


def neighborhood(df: pd.DataFrame, max_depth: int = 1):
    """Returns neighborhood metric.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns `x`, `y` and `label`. `label` must be a categorical.

    max_depth : int, optional
        Maximum depth (or hops) to consider for neighborhood metric. Default is 1.
    """
    return _neighborhood(_prepare_xy(df), _prepare_labels(df))


def confusion_and_neighborhood(df: pd.DataFrame, max_depth: int = 1):
    """Returns confusion and neighborhood metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns `x`, `y` and `label`. `label` must be a categorical.

    max_depth : int, optional
        Maximum depth (or hops) to consider for neighborhood metric. Default is 1.
    """
    return _confusion_and_neighborhood(_prepare_xy(df), _prepare_labels(df), max_depth)


@dataclass
class GraphStats:
    """Statistics from building the Delaunay triangulation.

    Attributes
    ----------
    triangle_count : int
        Number of triangles in the graph.

    ambiguous_circumcircle_count : int
        The number of occurrences where four or more points lie on the same
        circumcircle, resulting in an ambiguous triangulation with
        identical circumcenters.
    """

    triangle_count: int
    ambiguous_circumcircle_count: int


def graph_stats(df: pd.DataFrame):
    """Returns statistics from building the Delaunay triangulation."""
    graph = _prepare_xy(df)
    return GraphStats(
        triangle_count=graph.triangle_count(),
        ambiguous_circumcircle_count=graph.ambiguous_circumcircle_count(),
    )
