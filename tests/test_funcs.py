from unittest.mock import MagicMock

import pandas as pd
import pytest
from datasets.exceptions import DataFilesNotFoundError, DatasetNotFoundError

from modulift import funcs
from modulift.funcs import _prepare, jaccard_similarity, load_data, markdown_print


class TestJaccardSimilarity:
    def test_identical_sets(self):
        assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        assert jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(
            0.5
        )

    def test_both_empty_returns_zero(self):
        assert jaccard_similarity(set(), set()) == 0

    def test_one_empty(self):
        assert jaccard_similarity({"a"}, set()) == 0.0

    def test_accepts_list_as_second_argument(self):
        assert jaccard_similarity({"a", "b"}, ["a", "c"]) == pytest.approx(1 / 3)

    def test_accepts_any_iterables_and_ignores_duplicates(self):
        assert jaccard_similarity(["a", "a", "b"], ("b", "c")) == pytest.approx(1 / 3)

    def test_returns_float(self):
        assert isinstance(jaccard_similarity(set(), set()), float)


class TestLoadData:
    def test_returns_prepared_train_split(self, monkeypatch):
        raw = pd.DataFrame(
            {
                "package": ["x"],
                "description": ["d"],
                "keywords": ["k"],
                "popularity": [1],
            }
        )
        split = MagicMock()
        split.to_pandas.return_value = raw
        load_dataset = MagicMock(return_value={"train": split})
        monkeypatch.setattr(funcs, "load_dataset", load_dataset)

        result = load_data()

        load_dataset.assert_called_once_with("yeniguno/pypi")
        pd.testing.assert_frame_equal(result, raw)

    def test_result_is_cached(self, raw_dataset):
        calls = raw_dataset(
            pd.DataFrame(
                {
                    "package": ["x"],
                    "description": ["d"],
                    "keywords": ["k"],
                    "popularity": [1],
                }
            )
        )

        first = load_data()
        second = load_data()

        assert first is second
        assert len(calls) == 1

    def test_failures_are_not_cached(self, monkeypatch):
        attempts = []

        def flaky(*a, **k):
            attempts.append(1)
            raise ConnectionError("offline")

        monkeypatch.setattr(funcs, "load_dataset", flaky)

        for _ in range(2):
            with pytest.raises(RuntimeError):
                load_data()
        assert len(attempts) == 2

    @pytest.mark.parametrize(
        "exc, match",
        [
            (DatasetNotFoundError("missing"), "not found"),
            (DataFilesNotFoundError("no files"), "no matching files"),
            (ConnectionError("offline"), "internet connection"),
        ],
    )
    def test_load_failures_raise_runtime_error(self, monkeypatch, exc, match):
        def boom(*a, **k):
            raise exc

        monkeypatch.setattr(funcs, "load_dataset", boom)

        with pytest.raises(RuntimeError, match=match) as info:
            load_data()
        assert info.value.__cause__ is exc

    def test_missing_train_split_raises(self, monkeypatch):
        monkeypatch.setattr(funcs, "load_dataset", lambda *a, **k: {"test": None})

        with pytest.raises(RuntimeError, match="train"):
            load_data()

    def test_empty_dataset_raises(self, monkeypatch):
        split = MagicMock()
        split.to_pandas.return_value = pd.DataFrame()
        monkeypatch.setattr(funcs, "load_dataset", lambda *a, **k: {"train": split})

        with pytest.raises(RuntimeError, match="empty"):
            load_data()


class TestPrepare:
    def raw(self, **overrides):
        data = {
            "package": ["a", "b"],
            "description": ["d1", "d2"],
            "keywords": ["k1", "k2"],
            "popularity": [1, 2],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_missing_columns_raise(self):
        with pytest.raises(RuntimeError, match="missing required columns"):
            _prepare(self.raw().drop(columns=["keywords"]))

    def test_missing_values_in_text_columns_become_empty_strings(self):
        df = _prepare(
            self.raw(
                package=["a", None],
                description=[None, "d2"],
                keywords=["k1", float("nan")],
            )
        )

        assert df["package"].tolist() == ["a", ""]
        assert df["description"].tolist() == ["", "d2"]
        assert df["keywords"].tolist() == ["k1", ""]

    def test_popularity_is_integer_and_missing_becomes_zero(self):
        df = _prepare(self.raw(popularity=[float("nan"), 2.0]))

        assert df["popularity"].tolist() == [0, 2]
        assert pd.api.types.is_integer_dtype(df["popularity"])

    def test_index_is_reset(self):
        raw = self.raw()
        raw.index = [10, 20]

        assert _prepare(raw).index.tolist() == [0, 1]

    def test_does_not_modify_input(self):
        raw = self.raw(description=[None, "d2"])

        _prepare(raw)

        assert raw["description"].isna().any()


class TestMarkdownPrint:
    @pytest.fixture
    def displayed(self, monkeypatch):
        shown = []
        monkeypatch.setattr("IPython.display.display", lambda obj: shown.append(obj))
        return shown

    def test_formats_each_item(self, displayed):
        markdown_print([{"package": "pandas", "popularity": 10}])

        assert len(displayed) == 1
        text = displayed[0].data
        assert "**Package:** pandas" in text
        assert "**Popularity:** 10" in text
        assert text.endswith("---\n")

    def test_one_separator_per_item(self, displayed):
        markdown_print([{"a": 1}, {"a": 2}])

        assert displayed[0].data.count("---") == 2

    def test_empty_list(self, displayed):
        markdown_print([])

        assert displayed[0].data == ""
