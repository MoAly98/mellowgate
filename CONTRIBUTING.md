# Contributing to mellowgate

Thanks for your interest in improving mellowgate! This guide explains how to set
up a development environment, run checks, and propose changes.

---

## 1. Prerequisites

- **Python 3.11–3.13**
- **[pixi](https://pixi.sh/)** for environment management
- **Git** for version control

Installing pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

---

## 2. Getting started

Clone the repository and bootstrap the development environment:

```bash
git clone https://github.com/MoAly98/mellowgate.git
cd mellowgate
make setup-dev
```

`make setup-dev` installs dependencies, registers pre-commit hooks, and installs
the package in editable mode.

---

## 3. Development workflow

1. Create a feature branch off `main`.
2. Make your changes.
3. Run the checks shown below.
4. Commit with a semantic prefix (`feat:`, `fix:`, `docs:`, `refactor:`, …).
5. Push and open a pull request describing **why**, **what changed**, and **how
   you tested it**.

---

## 4. Running checks

### Pre-commit (recommended before every commit)

```bash
pixi run pre-commit run --all-files
```

### Linting and formatting

```bash
make lint
make format  # only if you need to apply fixes
```

### Tests

```bash
make test
make test-cov  # optional coverage report
```

### Documentation

```bash
make docs
open docs/_build/html/index.html
```

Update the docs when you change behaviour or introduce new features.

---

## 5. Submitting pull requests

- Keep changes focused and small when possible.
- Include tests for new behaviour or bug fixes.
- Update docs and docstrings if public behaviour changes.
- Reference related issues in the pull request description.

---

## 6. Reporting issues

When filing a bug report, include:

- Steps to reproduce
- Expected vs actual behaviour
- Environment details (`python --version`, platform)
- Relevant logs or stack traces

For feature requests, describe the problem you are solving and any proposed API.

We appreciate your help keeping mellowgate reliable and easy to use!
