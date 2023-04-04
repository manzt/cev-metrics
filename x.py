#!/usr/bin/env python

import numpy as np
from cev_metrics import Graph

def main():
    # df = pd.read_parquet("../cev-user-study/task2/parquet/T0A.parquet")
    # graph = Graph(df[["x", "y"]].to_numpy())
    # labels = Labels.from_codes(df["label"].cat.codes)
    # confusion = labels.confusion(graph)
    # neighborhood = labels.neighborhood(graph, confusion)
    X = np.array([[0, 0],[0, 1],[1, 1],[1, 0]], dtype=np.float64)
    graph = Graph(X)
    print(graph)
    graph.run()

if __name__ == "__main__":
    main()
