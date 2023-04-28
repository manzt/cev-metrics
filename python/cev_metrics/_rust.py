import numpy as np
import pandas as pd

from cev_metrics.cev_metrics import (
    Graph,
    _confusion,
    _confusion_and_neighborhood,
    _neighborhood,
)


def _prepare(df: pd.DataFrame):
    points = df[["x", "y"]].values
    codes = df["label"].cat.codes.values

    if points.dtype != np.float64:
        points = points.astype(np.float64)

    if codes.dtype != np.int16:
        codes = codes.astype(np.int16)

    return Graph(points), codes


def confusion(df: pd.DataFrame):
    """Returns confusion matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns `x`, `y` and `label`. `label` must be a categorical.
    """
    graph, codes = _prepare(df)
    return _confusion(graph, codes)


def neighborhood(df: pd.DataFrame, max_depth: int = 1):
    """Returns neighborhood metric.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns `x`, `y` and `label`. `label` must be a categorical.

    max_depth : int, optional
        Maximum depth (or hops) to consider for neighborhood metric. Default is 1.
    """
    graph, codes = _prepare(df)
    return _neighborhood(graph, codes, max_depth)


def confusion_and_neighborhood(df: pd.DataFrame, max_depth: int = 1):
    """Returns confusion and neighborhood metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns `x`, `y` and `label`. `label` must be a categorical.

    max_depth : int, optional
        Maximum depth (or hops) to consider for neighborhood metric. Default is 1.
    """
    graph, codes = _prepare(df)
    return _confusion_and_neighborhood(graph, codes, max_depth)
