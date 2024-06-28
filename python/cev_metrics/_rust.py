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


def ambiguous_circumcircle_count(df: pd.DataFrame):
    """Returns the number of ambiguous circumcircles found in the Delaunay triangulation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns `x`, `y` and `label`. `label` must be a categorical.
    """
    return _prepare_xy(df).ambiguous_circumcircle_count()
