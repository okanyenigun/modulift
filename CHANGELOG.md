## [0.1.1] | 06.11.2026

### Added

- `pyproject.toml` (hatchling build backend), `uv.lock` and `.python-version` were added. The project is now managed with `uv`.

- `twine` and `pytest` moved to a `dev` dependency group.

- A unit test suite (`tests/`) was added. It runs offline, using a fake dataset and a fake embedding model.

- `__all__` was added to the package.

- GitHub Actions: CI (ruff lint and format, lock-file check, tests on Python 3.9-3.13 with coverage, build and `twine check`), publishing to PyPI through Trusted Publishing when a release is published (with a TestPyPI option), a weekly smoke test against the real dataset, and Dependabot.

- Python 3.12 and 3.13 classifiers were added.

- The dataset and data derived from it (TF-IDF matrix, embeddings, keyword sets, sentence-transformers model) are now cached, so repeated calls are much faster.

- Argument checks: `limit` must be a positive integer (not a float or bool), text arguments must be non-empty strings, keywords must be non-empty strings, and unexpected keyword arguments to `search_by_description` raise a `TypeError`.

### Changed

- Minimum supported Python version raised from 3.7 to 3.9, as required by `scikit-learn` and `datasets`. The 3.7 and 3.8 classifiers were removed.

- Dependency pins (`==`) were relaxed to minimum versions (`>=`).

- License metadata now uses the SPDX `license = "MIT"` field.

- `load_data` raises a `RuntimeError` with a clear message (instead of printing and returning `None`) when the dataset cannot be loaded, has no `train` split, is empty or lacks required columns. Missing values in the dataset are cleaned up on load.

- `search_by_keywords` with `method="exact"` now matches whole words/phrases only (`"data"` no longer matches `"metadata"`) and is case-insensitive on both sides. Jaccard matching is case-insensitive as well.

- `find_similar_packages` now looks up the reference package case-insensitively, like `search_by_package_name`.

- All results now contain plain Python types (`int`, `float`, `str`) instead of NumPy types.

- Type hints and docstrings were corrected.

- The default-embedding-model notice is printed only when that model is first loaded, not on every call.

### Fixed

- `search_by_description` with `method="fts"` treated the query as a regular expression, so queries such as `c++` or `(` raised an error. It is now a plain-text search.

- `find_similar_packages` removed the reference package by dropping the first result, which returned the reference (and dropped a real match) when another package tied with it. The reference is now excluded explicitly and ties are ordered deterministically.

- Packages with missing descriptions, keywords or popularity no longer break searches.

- Results no longer depend on the dataset having a default 0..n index.

### Removed

- `setup.py` and `requirements.txt` were removed in favor of `pyproject.toml`.

## [0.1.0] | 23.05.2025

### Added

- `sentence-transformers` added to the requirements.

### Changed

- Api was refactored.

- Tutorials were updated.

## [0.0.3] | 31.01.2025

### Added

- The `find_similar_package` function has been added to the API. It identifies similar packages for a given package name using TF-IDF.

## [0.0.2] | 07.01.2025

### Added

- The maintenance pipeline code has been implemented.

### Changed

- The popularity ranking has been updated, and the code (including the sorting mechanism) has been modified accordingly.

### Fixed

- The Markdown display issue has been resolved.
