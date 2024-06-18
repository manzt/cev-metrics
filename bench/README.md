Rough benchmark for the `confusion_and_neighborhood` function for different sizes of the input.

```sh
rye run bench
# Python version: 3.12.2
# Python implementation: CPython
# Python build: ('main', 'Feb 25 2024 03:55:42')
# System: Darwin
# Machine: arm64
# Platform: macOS-14.5-arm64-arm-64bit
# Processor: arm
# 
# Size: 1k
#   Function: confusion_and_neighborhood
#   Number of runs: 100
#   Mean execution time: 0.016283 seconds
#   Standard deviation: 0.001135 seconds
#   Min execution time: 0.014792 seconds
#   Max execution time: 0.026159 seconds
# Size: 10k
#   Function: confusion_and_neighborhood
#   Number of runs: 100
#   Mean execution time: 0.060442 seconds
#   Standard deviation: 0.001123 seconds
#   Min execution time: 0.058218 seconds
#   Max execution time: 0.063350 seconds
# Size: 100k
#   Function: confusion_and_neighborhood
#   Number of runs: 100
#   Mean execution time: 0.351976 seconds
#   Standard deviation: 0.004212 seconds
#   Min execution time: 0.340749 seconds
#   Max execution time: 0.372692 seconds
# Size: 1M
#   Function: confusion_and_neighborhood
#   Number of runs: 100
#   Mean execution time: 3.297965 seconds
#   Standard deviation: 0.128846 seconds
#   Min execution time: 3.090444 seconds
#   Max execution time: 3.895661 seconds
```
