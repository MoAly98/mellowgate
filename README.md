# mellowgate

Tools for estimating gradients in discrete optimisation problems with JAX.
Mellowgate provides three estimators—finite differences, REINFORCE, and
Gumbel-Softmax—alongside helper utilities for defining problems, running
parameter sweeps, and collecting results.

- Documentation: https://mellowgate.readthedocs.io
- Issues: https://github.com/MoAly98/mellowgate/issues

## Installation

```bash
pip install mellowgate
```

For local development inside the pinned pixi environment:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
git clone https://github.com/MoAly98/mellowgate.git
cd mellowgate
make setup-dev
```

## Quick links

- [Quickstart](https://mellowgate.readthedocs.io/en/latest/quickstart.html) —
  define a discrete problem, configure estimators, and inspect results.
- [Concepts](https://mellowgate.readthedocs.io/en/latest/concepts.html) —
  mathematical background for expectations, gradients, and estimator formulas.
- [Tutorial](https://mellowgate.readthedocs.io/en/latest/tutorial.html) —
  end-to-end example analysing a switching controller.

## Development

Common tasks are exposed through the Makefile:

```bash
make lint        # Ruff lint/format check
make format      # Ruff formatter
make test        # Pytest suite
make docs        # Build Sphinx documentation
```

See the [contributing guide](https://mellowgate.readthedocs.io/en/latest/contributing.html)
for coding standards, test guidance, and documentation tips.

## License

Licensed under the MIT License. See [LICENSE](LICENSE).
