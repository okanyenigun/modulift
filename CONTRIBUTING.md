# Contributing to Modulift

Thank you for your interest in contributing to Modulift! Your contributions are greatly appreciated and help improve the package for the entire community.

## Development

The project uses [uv](https://docs.astral.sh/uv/). CI runs the same checks:

```bash
uv sync                            # create the environment, including dev tools
uv run ruff check .                # lint
uv run ruff format --check .       # formatting (use `ruff format .` to fix)
uv run pytest --cov                # tests (offline, no dataset download)
uv build && uvx twine check dist/* # packaging
```

When you bump the version, change it in both `pyproject.toml` and `modulift/__init__.py`, and add a `CHANGELOG.md` entry. A release is published to PyPI when a GitHub release with a matching `vX.Y.Z` tag is created.
