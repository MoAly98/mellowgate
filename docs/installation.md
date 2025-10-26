# Installation

Mellowgate ships as a pure-Python package with optional extras for development.
The library targets Python 3.11–3.13 and relies on JAX for accelerated
computation.

## Quick install

```bash
pip install mellowgate
```

This pulls the latest release from PyPI together with the core runtime
dependencies:

- `jax`
- `flax`
- `matplotlib`
- `rich`

After installation you can confirm the package is available by running:

```bash
python -c "import mellowgate; print(mellowgate.__version__)"
```

## Working from source

The repository bundles a [pixi](https://pixi.sh/) environment that mirrors the
CI configuration. Clone the project and bootstrap the toolchain:

```bash
git clone https://github.com/MoAly98/mellowgate.git
cd mellowgate
pixi install
make setup-dev  # installs pre-commit hooks and the editable package
```

The development environment exposes the commonly used commands as make targets:

::::{grid} 1
:gutter: 3

:::{grid-item-card} 🚀 Format, lint, and test
:class-header: bg-info text-white
`make format`, `make lint`, and `make test` run the same checks as CI. Use
`make pre-commit` to run all pre-commit hooks locally.
:::

:::{grid-item-card} 📊 Coverage reports
:class-header: bg-dark text-white
`make test-cov` executes the test suite with coverage and writes an HTML report
under `htmlcov`. Open `htmlcov/index.html` in a browser to inspect results.
:::

:::{grid-item-card} 🧪 Examples
:class-header: bg-light
The `example.py` script and the documentation walkthroughs (quickstart, concepts,
tutorial) illustrate the high-level utilities. Run them inside the pixi shell to
ensure JAX is configured consistently.
:::

::::

## Optional extras

Enable additional dependency groups through pixi features:

| Feature | Command | Purpose |
|---------|---------|---------|
| `docs`  | `pixi run -e docs sphinx-build -b html docs docs/_build/html` | Installs the documentation toolchain with myst, sphinx-design, and Mermaid. |
| `dev`   | `pixi shell -e dev` | Pulls in docs, linting, formatting, and testing dependencies for contributor workflows. |

When developing locally we recommend staying inside the pixi-managed shell so
all tools use the pinned versions from `pixi.lock`.
