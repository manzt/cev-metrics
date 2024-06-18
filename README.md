# cev-metrics

A companion library to [`cev`](https://github.com/ozettetech/comparative-embedding-visualization). Rust implementations of our embedding comparison metrics, exposed via Python bindings.

## development

This project uses [`rye`](https://rye-up.com/) for development. You can create
a virtual environment with `rye sync`.

```sh
rye sync
```

To develop the Rust component, you will need [`maturin`](https://github.com/PyO3/maturin).

```sh
maturin develop --release && rye run python ./x.py
```

Code quality checks are available via `rye`.

```sh
rye lint # lints code
rye fmt  # formats code
rye test # runs tests
```

The [benchmark](./bench/README.md) can be run with `rye run bench`.
