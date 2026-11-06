<p align="center">
  <img src="https://raw.githubusercontent.com/okanyenigun/modulift/master/docs/assets/modulift-wide.png" alt="modulift" width="400">
</p>

<p align="center">
  <b>Find the right Python package among ~485,000 on PyPI: by keywords, plain-language description, or similarity to one you already know.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/modulift/"><img src="https://img.shields.io/pypi/v/modulift" alt="PyPI version"></a>
  <a href="https://pypi.org/project/modulift/"><img src="https://img.shields.io/pypi/pyversions/modulift" alt="Python versions"></a>
  <a href="https://github.com/okanyenigun/modulift/actions/workflows/ci.yml"><img src="https://github.com/okanyenigun/modulift/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/okanyenigun/modulift/blob/master/Licence.md"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
</p>

---

You often know *what* you need ("something that fills in missing values in a DataFrame") but not *which* package does it. Web searches and AI assistants tend to surface the same few popular libraries. **modulift** searches a dataset of PyPI packages, each with a description, keywords, a 1–5 popularity score and a sentence embedding, so the smaller, more specialized packages turn up too. You can search straight from your code or notebook.

## Installation

```bash
pip install modulift
```

modulift needs Python 3.10 or newer. The dataset is hosted on the Hugging Face Hub ([`yeniguno/pypi`](https://huggingface.co/datasets/yeniguno/pypi)) and is downloaded on first use (see [Performance](#performance)).

## Quick start

```python
import modulift as mf

mf.search_by_description("turn my python script into a nice command line tool", method="cosine")
mf.search_by_keywords("web scraping", "async", relation="and")
mf.find_similar_packages("pandera", method="cosine")
mf.search_by_package_name("requests")
```

| Function | What it does | Methods |
|---|---|---|
| `search_by_package_name(name)` | Look up one package (case-insensitive) | |
| `search_by_keywords(*keywords, relation=...)` | Search by keywords, with AND/OR logic | `exact` (default), `jaccard` |
| `search_by_description(text)` | Search descriptions | `fts` (default), `cosine` |
| `find_similar_packages(name)` | Packages related to one you know | `tf-idf` (default), `cosine`, `jaccard` |

Every search takes `limit` (default `5`) and `markdown=True`, which renders the results in Jupyter.

## Usage

The examples below are real results, with descriptions shortened.

### Look up a package

```python
mf.search_by_package_name("requests")
```

```python
{
    "package": "requests",
    "description": "requests is a python library for making http requests, designed for ease of use and robust functionality.",
    "keywords": "http, requests, python, api, authentication, data handling",
    "popularity": 5,
}
```

The lookup is case-insensitive. An exact-case match wins if one exists. A name that isn't in the dataset raises `ValueError`.

### Search by keywords

```python
mf.search_by_keywords("data validation", "pandas", "schema", relation="and")
```

```text
pandera-report        pandera extension for row-based reporting, providing enhanced data validation...
pl-compare            compare and find differences between two polars dataframes...
shmessy               deal with messy pandas dataframes...
data-linter           automatic validation of tabular data within data engineering...
minimonolith-schema   validating the columns of data tables...
```

- `relation="or"` (the default) matches any keyword. `relation="and"` matches all of them.
- `method="exact"` matches **whole words and phrases**, case-insensitively, so `"data"` matches "data science" but not "metadata". Results are sorted by popularity.
- `method="jaccard"` ranks packages by keyword overlap, and adds a `jaccard_score` to each result.

### Search by description

Plain-text search (`fts`, the default) finds descriptions that contain your text:

```python
mf.search_by_description("missing data")
```

```text
fancyimpute   matrix completion and imputation... to handle missing data in datasets
fbprophet     time series forecasting... designed to handle strong seasonal effects and missing data
hyperimpute   handling missing data in machine learning pipelines...
impyute       missing data imputation algorithms...
knnimpute     k-nearest neighbors (knn) imputation...
```

Semantic search (`cosine`) embeds your query with a sentence-transformers model and compares it with every package's embedding. Describe the problem in your own words:

```python
mf.search_by_description("turn my python script into a nice command line tool", method="cosine")
```

```text
uclt      0.780   functions to create a command-line tool
pith      0.775   a simple command-line tool for executing python scripts...
captain   0.767   a python cli framework for creating easy-to-use command-line scripts
hazwaz    0.766   a python library for writing command-line scripts
pyglib    0.765   a framework for creating command-line scripts and applications
```

The default model is `sentence-transformers/all-MiniLM-L6-v2`. Use another with `embedding_model="..."`. Each result gets a `cosine_similarity`.

### Find similar packages

```python
mf.find_similar_packages("pandera", method="cosine")
```

```text
pandera-tool        data manipulation and analysis using the pandera library...
panderaext          extends the pandera type checking framework...
streamlit_pandera   data ingestion, validation, and processing using pandera...
panorma             typed models for pandas dataframes using classes...
pano                creating and maintaining panoramic data models...
```

- `tf-idf` compares description and keyword text.
- `cosine` compares embeddings.
- `jaccard` compares keyword sets.

The reference package itself is never part of the results.

### Results

Every function returns plain Python data, so the output can go straight into `json.dumps` or `pandas.DataFrame`:

```python
{"package": str, "description": str, "keywords": str, "popularity": int}
# plus "jaccard_score" or "cosine_similarity" (float) for those methods
```

Invalid arguments (an empty query, a non-positive `limit`, an unknown method) raise `ValueError`. If the dataset can't be loaded, you get a `RuntimeError` with the reason.

## Performance

- **First use downloads the dataset:** about 1 GB, plus a few GB of disk for the Hugging Face cache. Cosine search also downloads the embedding model (about 90 MB) the first time.
- **Within a process, everything is cached:** the dataset, the TF-IDF matrix, the embeddings, the keyword index and the model. Only the first search is slow.
- **Logging:** modulift doesn't print. It reports events such as loading the embedding model through the standard `logging` module (logger `modulift`). To see them:

  ```python
  import logging
  logging.basicConfig(level=logging.INFO)
  ```

## Things to keep in mind

- **The dataset is a snapshot of PyPI.** It can miss very new packages and include ones that have since been abandoned.
- **PyPI includes clones, forks and test uploads.** Treat results as leads. Check a package on PyPI (latest release, license, supported Python versions) before you depend on it.

## Use with AI coding agents

[`skills/modulift`](https://github.com/okanyenigun/modulift/tree/master/skills/modulift) is an agent skill that teaches a coding agent (such as Claude Code) to use modulift when it has to choose a dependency. The agent searches in several ways, filters out clones and abandoned packages, checks the shortlist against live PyPI data, and only then recommends a package. To use it, copy the folder into `~/.claude/skills/` or a project's `.claude/skills/`.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/okanyenigun/modulift/blob/master/CONTRIBUTING.md) for the development setup (uv, ruff, pytest) and release process, and [CHANGELOG.md](https://github.com/okanyenigun/modulift/blob/master/CHANGELOG.md) for what changed.

## License

MIT. See [Licence.md](https://github.com/okanyenigun/modulift/blob/master/Licence.md).

## Contact

Okan Yenigün: [okanyenigun@gmail.com](mailto:okanyenigun@gmail.com)
