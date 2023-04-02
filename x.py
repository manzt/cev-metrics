#!/usr/bin/env python

import numpy as np
from cev_graph import Graph

def main():
    coords = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ], dtype=np.float64)
    graph = Graph(coords)
    print(graph)

if __name__ == "__main__":
    main()
