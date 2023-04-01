#!/usr/bin/env python

import numpy as np
import cev_graph

def main():
    arr = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ], dtype=np.float64)
    print(arr)
    cev_graph.graph(arr)

if __name__ == "__main__":
    main()
