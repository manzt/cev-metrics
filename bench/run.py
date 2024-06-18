import statistics
import timeit
import typing
import platform

import pandas as pd
from cev_metrics import confusion_and_neighborhood


def load_data():
    file = "/Users/manzt/Downloads/unstimulated-pembrolizumab-responder.parquet"
    df = pd.read_parquet(file)
    df = df.rename(columns={"umapX": "x", "umapY": "y", "faustLabels": "label"})
    df["label"] = df["label"].astype("category")
    return df[["x", "y", "label"]]


def bench(func: typing.Callable, args: tuple, kwargs: dict, num_runs: int):
    times = timeit.repeat(lambda: func(*args, **kwargs), repeat=num_runs, number=1)

    mean_time = statistics.mean(times)
    stdev_time = statistics.stdev(times)

    print(f"  Function: {func.__name__}")
    print(f"  Number of runs: {num_runs}")
    print(f"  Mean execution time: {mean_time:.6f} seconds")
    print(f"  Standard deviation: {stdev_time:.6f} seconds")
    print(f"  Min execution time: {min(times):.6f} seconds")
    print(f"  Max execution time: {max(times):.6f} seconds")


def main():
    num_runs = 100
    df = load_data()

    print(f"Python version: {platform.python_version()}")
    print(f"Python implementation: {platform.python_implementation()}")
    print(f"Python build: {platform.python_build()}")
    print(f"System: {platform.system()}")
    print(f"Machine: {platform.machine()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}", end="\n\n")

    print("Size: 1k")
    bench(
        confusion_and_neighborhood,
        args=(df.sample(n=1_000, random_state=42, replace=False),),
        kwargs={},
        num_runs=num_runs,
    )
    print("Size: 10k")
    bench(
        confusion_and_neighborhood,
        args=(df.sample(n=10_000, random_state=42, replace=False),),
        kwargs={},
        num_runs=num_runs,
    )
    print("Size: 100k")
    bench(
        confusion_and_neighborhood,
        args=(df.sample(n=100_000, random_state=42, replace=False),),
        kwargs={},
        num_runs=num_runs,
    )
    print("Size: 1M")
    bench(
        confusion_and_neighborhood,
        args=(df.sample(n=1_000_000, random_state=42, replace=False),),
        kwargs={},
        num_runs=num_runs,
    )


if __name__ == "__main__":
    main()
