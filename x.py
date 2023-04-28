#!/usr/bin/env python

import pandas as pd
from cev_metrics import confusion_and_neighborhood

# import cev_metrics.py

def main():
    df = pd.DataFrame(
        {
            "x": [0.0, 0.0, 1.0, 1.0],
            "y": [0.0, 1.0, 1.0, 0.0],
            "label": ["a", "b", "a", "b"],
        }
    ).astype({"label": "category"})
    df.info()
    confusion, neighborhood = confusion_and_neighborhood(df)
    print(confusion)
    print(neighborhood)


if __name__ == "__main__":
    main()
