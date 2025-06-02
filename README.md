# Model

## Setup

### Install uv

https://docs.astral.sh/uv/getting-started/installation/


```bash
uv venv                                     # creates venv
uv sync --all-extras --dev                  # installs dependencies on venv
uv run -m segmentation.examples.models      # this is how to run .py files
```

Format code with ruff
```bash
uv run ruff check --select I --fix    # format imports, or run without --fix to check only
uv run ruff format                    # format code, or run with --check
```

Check typing
```bash
uv run mypy . --config-file pyproject.toml  # runs type linter
```