from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from modulift import api, funcs


@pytest.fixture(autouse=True)
def clear_caches():
    """The dataset and derived data are cached; never let that leak between tests."""
    api._clear_caches()
    yield
    api._clear_caches()


@pytest.fixture
def sample_df():
    """A small, deterministic stand-in for the Hugging Face dataset."""
    return pd.DataFrame(
        {
            "package": ["pandas", "numpy", "requests", "scikit-learn", "Flask"],
            "description": [
                "Data analysis library for tabular data",
                "Fundamental package for numerical computing",
                "HTTP library for humans",
                "Machine learning in Python and data analysis",
                "A lightweight web framework",
            ],
            "keywords": [
                "data, analysis, dataframe",
                "array, math, data",
                "http, web, client",
                "machine learning, data, analysis",
                "web, framework, wsgi",
            ],
            "popularity": [10, 9, 8, 7, 3],
            "embedding": [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.9, 0.1, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.8, 0.0, 0.2]),
                np.array([0.0, 0.9, 0.1]),
            ],
        }
    )


@pytest.fixture
def mock_load_data(monkeypatch, sample_df):
    """Patch load_data where api.py uses it, returning a fresh copy per call."""
    monkeypatch.setattr("modulift.api.load_data", lambda: sample_df.copy())
    return sample_df


@pytest.fixture
def mock_markdown(monkeypatch):
    """Capture markdown_print calls instead of rendering with IPython."""
    calls = []
    monkeypatch.setattr("modulift.api.markdown_print", lambda data: calls.append(data))
    return calls


@pytest.fixture
def use_df(monkeypatch):
    """Serve an arbitrary DataFrame as the dataset (bypassing load_data's cleaning)."""

    def _use(df):
        monkeypatch.setattr("modulift.api.load_data", lambda: df.copy())

    return _use


@pytest.fixture
def raw_dataset(monkeypatch):
    """
    Serve a raw DataFrame through the real load_data() (cleaning + caching included)
    by faking the Hugging Face loader. Returns the list of load_dataset calls.
    """

    def _serve(df):
        calls = []
        split = MagicMock()
        split.to_pandas.return_value = df

        def fake_load_dataset(*args, **kwargs):
            calls.append(args)
            return {"train": split}

        monkeypatch.setattr(funcs, "load_dataset", fake_load_dataset)
        return calls

    return _serve


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Install a fake sentence_transformers module; returns the list of created model names."""
    import sys
    import types

    created = []

    class FakeModel:
        def __init__(self, name):
            created.append(name)

        def encode(self, text, convert_to_numpy=True):
            return np.array([1.0, 0.0, 0.0])

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return created
