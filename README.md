# cev-metrics

A companion library to [`cev`](https://github.com/ozettetech/comparative-embedding-visualization). Rust implementations of our embedding comparison metrics, exposed via Python bindings.

## development

This project uses [`uv`](https://astral.sh/uv/) for development.

To develop the Rust component, you will need [`maturin`](https://github.com/PyO3/maturin).

```sh
uvx maturin develop --uv --release && uv run bench/run.py
```

Code quality checks are available via `uv`.

```sh
uv run ruff check .
uv run pytest
```

The [benchmark](./bench/README.md) can be run with `uv run bench/run.py`.
