#!/usr/bin/env python

import numpy as np
import pandas as pd
from cev_metrics import Graph, confusion, neighborhood


def main():
    df = pd.DataFrame(
        {
            "x": [0.0, 0.0, 1.0, 1.0],
            "y": [0.0, 1.0, 1.0, 0.0],
            "label": ["a", "b", "a", "b"],
        }
    ).astype({"label": "category"})
    df.info()

    points = df[["x", "y"]].values
    labels = df["label"].cat.codes.values

    if points.dtype != np.float64:
        points = points.astype(np.float64)

    if labels.dtype != np.int16:
        labels = labels.astype(np.int16)

    graph = Graph(points)
    data = neighborhood(graph, labels)
    print(data)


if __name__ == "__main__":
    main()
