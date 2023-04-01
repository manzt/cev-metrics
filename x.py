#!/usr/bin/env python

import numpy as np
import cev_graph

def main():
    arr = np.random.rand(100, 2)
    cev_graph.graph(arr)

if __name__ == "__main__":
    main()
