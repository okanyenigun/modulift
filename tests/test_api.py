import sys
import types

import numpy as np
import pandas as pd
import pytest

from modulift import api, funcs
from modulift import (
    find_similar_packages,
    search_by_description,
    search_by_keywords,
    search_by_package_name,
)

RECORD_KEYS = {"package", "description", "keywords", "popularity"}


class TestSearchByKeywords:
    def test_requires_a_keyword(self, mock_load_data):
        with pytest.raises(ValueError, match="At least one keyword"):
            search_by_keywords()

    def test_invalid_relation(self, mock_load_data):
        with pytest.raises(ValueError, match="Relation"):
            search_by_keywords("data", relation="xor")

    def test_invalid_method(self, mock_load_data):
        with pytest.raises(ValueError, match="Method"):
            search_by_keywords("data", method="fuzzy")

    @pytest.mark.parametrize("limit", [0, -1])
    def test_invalid_limit(self, mock_load_data, limit):
        with pytest.raises(ValueError, match="Limit"):
            search_by_keywords("data", limit=limit)

    def test_exact_or_sorted_by_popularity(self, mock_load_data):
        results = search_by_keywords("data", "web", limit=10)

        names = [r["package"] for r in results]
        assert names == ["pandas", "numpy", "requests", "scikit-learn", "Flask"]
        assert all(set(r) == RECORD_KEYS for r in results)

    def test_exact_and_requires_all_keywords(self, mock_load_data):
        results = search_by_keywords("data", "analysis", relation="and", limit=10)

        assert [r["package"] for r in results] == ["pandas", "scikit-learn"]

    def test_keywords_are_case_insensitive(self, mock_load_data):
        results = search_by_keywords("DATA", "Analysis", relation="and", limit=10)

        assert [r["package"] for r in results] == ["pandas", "scikit-learn"]

    def test_limit_is_respected(self, mock_load_data):
        assert len(search_by_keywords("data", limit=2)) == 2

    def test_no_match_returns_empty_list(self, mock_load_data):
        assert search_by_keywords("nonexistent-keyword") == []

    def test_jaccard_returns_scores_in_descending_order(self, mock_load_data):
        results = search_by_keywords("data", "analysis", method="jaccard", limit=3)

        scores = [r["jaccard_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert len(results) == 3
        assert RECORD_KEYS | {"jaccard_score"} == set(results[0])

    def test_jaccard_best_match(self, mock_load_data):
        # {"data", "analysis"} vs pandas' {"data", "analysis", "dataframe"} -> 2/3
        results = search_by_keywords(
            "data", "analysis", "dataframe", method="jaccard", limit=1
        )

        assert results[0]["package"] == "pandas"
        assert results[0]["jaccard_score"] == pytest.approx(1.0)

    def test_markdown_flag_calls_markdown_print(self, mock_load_data, mock_markdown):
        results = search_by_keywords("data", markdown=True)

        assert mock_markdown == [results]

    def test_markdown_not_called_by_default(self, mock_load_data, mock_markdown):
        search_by_keywords("data")

        assert mock_markdown == []


class TestSearchByPackageName:
    def test_requires_name(self, mock_load_data):
        with pytest.raises(ValueError, match="Package name"):
            search_by_package_name("")

    def test_exact_match(self, mock_load_data):
        result = search_by_package_name("pandas")

        assert set(result) == RECORD_KEYS
        assert result["package"] == "pandas"
        assert result["popularity"] == 10

    def test_case_insensitive_fallback(self, mock_load_data):
        assert search_by_package_name("flask")["package"] == "Flask"
        assert search_by_package_name("PANDAS")["package"] == "pandas"

    def test_not_found(self, mock_load_data):
        with pytest.raises(ValueError, match="No package found"):
            search_by_package_name("does-not-exist")

    def test_markdown_wraps_result_in_list(self, mock_load_data, mock_markdown):
        result = search_by_package_name("numpy", markdown=True)

        assert mock_markdown == [[result]]


class TestSearchByDescription:
    def test_requires_description(self, mock_load_data):
        with pytest.raises(ValueError, match="Description"):
            search_by_description("")

    @pytest.mark.parametrize("limit", [0, -3])
    def test_invalid_limit(self, mock_load_data, limit):
        with pytest.raises(ValueError, match="Limit"):
            search_by_description("data", limit=limit)

    def test_invalid_method(self, mock_load_data):
        with pytest.raises(ValueError, match="Method"):
            search_by_description("data", method="bm25")

    def test_fts_matches_case_insensitively_sorted_by_popularity(self, mock_load_data):
        results = search_by_description("DATA ANALYSIS", limit=10)

        assert [r["package"] for r in results] == ["pandas", "scikit-learn"]
        assert all(set(r) == RECORD_KEYS for r in results)

    def test_fts_limit(self, mock_load_data):
        assert len(search_by_description("library", limit=1)) == 1

    def test_fts_no_match(self, mock_load_data):
        assert search_by_description("zzzz") == []

    def test_markdown_flag(self, mock_load_data, mock_markdown):
        results = search_by_description("web", markdown=True)

        assert mock_markdown == [results]

    def test_cosine_uses_model_and_ranks_by_similarity(
        self, mock_load_data, monkeypatch, capsys
    ):
        created = []

        class FakeModel:
            def __init__(self, name):
                created.append(name)

            def encode(self, text, convert_to_numpy=True):
                return np.array([1.0, 0.0, 0.0])

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeModel
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        results = search_by_description("anything", method="cosine", limit=2)

        assert created == ["sentence-transformers/all-MiniLM-L6-v2"]
        assert "default embedding model" in capsys.readouterr().out
        assert [r["package"] for r in results] == ["pandas", "numpy"]
        assert results[0]["cosine_similarity"] == pytest.approx(1.0)
        assert results[0]["cosine_similarity"] >= results[1]["cosine_similarity"]

    def test_cosine_custom_embedding_model(self, mock_load_data, monkeypatch, capsys):
        created = []

        class FakeModel:
            def __init__(self, name):
                created.append(name)

            def encode(self, text, convert_to_numpy=True):
                return np.array([0.0, 1.0, 0.0])

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeModel
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

        results = search_by_description(
            "web", method="cosine", limit=1, embedding_model="my/model"
        )

        assert created == ["my/model"]
        assert "default embedding model" not in capsys.readouterr().out
        assert results[0]["package"] == "requests"


class TestFindSimilarPackages:
    def test_requires_reference(self, mock_load_data):
        with pytest.raises(ValueError, match="Reference package"):
            find_similar_packages("")

    @pytest.mark.parametrize("limit", [0, -1])
    def test_invalid_limit(self, mock_load_data, limit):
        with pytest.raises(ValueError, match="Limit"):
            find_similar_packages("pandas", limit=limit)

    def test_invalid_method(self, mock_load_data):
        with pytest.raises(ValueError, match="Method"):
            find_similar_packages("pandas", method="euclid")

    def test_unknown_package(self, mock_load_data):
        with pytest.raises(ValueError, match="No package found"):
            find_similar_packages("nope")

    @pytest.mark.parametrize("method", ["tf-idf", "cosine", "jaccard"])
    def test_excludes_reference_and_respects_limit(self, mock_load_data, method):
        results = find_similar_packages("pandas", limit=2, method=method)

        names = [r["package"] for r in results]
        assert len(names) == 2
        assert "pandas" not in names
        assert all(set(r) == RECORD_KEYS for r in results)
        assert all(isinstance(r["popularity"], int) for r in results)

    def test_tfidf_most_similar(self, mock_load_data):
        results = find_similar_packages("pandas", limit=1)

        assert results[0]["package"] == "scikit-learn"

    def test_cosine_most_similar(self, mock_load_data):
        results = find_similar_packages("pandas", limit=1, method="cosine")

        assert results[0]["package"] == "numpy"

    def test_jaccard_most_similar(self, mock_load_data):
        results = find_similar_packages("pandas", limit=1, method="jaccard")

        assert results[0]["package"] == "scikit-learn"

    def test_markdown_flag(self, mock_load_data, mock_markdown):
        results = find_similar_packages("pandas", markdown=True)

        assert mock_markdown == [results]


class TestPublicApi:
    def test_all_exports_exist(self):
        import modulift

        for name in modulift.__all__:
            assert hasattr(modulift, name)

    def test_version_is_string(self):
        import modulift

        assert isinstance(modulift.__version__, str)


def make_df(rows, index=None):
    """rows: (package, description, keywords, popularity, embedding)"""
    df = pd.DataFrame(
        rows, columns=["package", "description", "keywords", "popularity", "embedding"]
    )
    if index is not None:
        df.index = index
    return df


class TestFtsIsPlainText:
    @pytest.mark.parametrize("query", ["c++", "(", "[abc", "*", "a|b"])
    def test_regex_metacharacters_do_not_raise(self, mock_load_data, query):
        assert search_by_description(query) == []

    def test_dot_is_literal(self, mock_load_data):
        # As a regex "." would match every description.
        assert search_by_description(".") == []

    def test_literal_special_characters_match(self, use_df):
        use_df(
            make_df(
                [
                    ("cpp-lib", "A C++ (modern) library", "cpp", 1, np.zeros(3)),
                    ("py-lib", "A python library", "py", 2, np.zeros(3)),
                ]
            )
        )

        assert [r["package"] for r in search_by_description("c++ (modern)")] == [
            "cpp-lib"
        ]


class TestReferenceIsNeverReturned:
    """Another package tying with the reference must not displace/replace it."""

    @pytest.fixture
    def tied_df(self):
        same = ("same text", "same, words", 5, np.array([1.0, 0.0, 0.0]))
        return make_df(
            [
                ("alpha", *same),
                ("beta", *same),
                ("gamma", *same),
                ("other", "something else", "different", 1, np.array([0.0, 1.0, 0.0])),
            ]
        )

    @pytest.mark.parametrize("method", ["tf-idf", "cosine", "jaccard"])
    @pytest.mark.parametrize("reference", ["alpha", "beta", "gamma"])
    def test_reference_excluded_and_ties_kept(self, use_df, tied_df, method, reference):
        use_df(tied_df)

        results = find_similar_packages(reference, limit=2, method=method)

        names = [r["package"] for r in results]
        assert reference not in names
        assert sorted(names) == sorted({"alpha", "beta", "gamma"} - {reference})

    @pytest.mark.parametrize("method", ["tf-idf", "cosine", "jaccard"])
    def test_limit_larger_than_dataset(self, mock_load_data, method):
        results = find_similar_packages("pandas", limit=100, method=method)

        assert len(results) == 4
        assert "pandas" not in [r["package"] for r in results]

    def test_ties_are_deterministic(self, use_df, tied_df):
        use_df(tied_df)

        runs = [
            [r["package"] for r in find_similar_packages("alpha", limit=2)]
            for _ in range(3)
        ]

        assert runs[0] == runs[1] == runs[2]


class TestMissingValues:
    @pytest.fixture
    def dirty(self, raw_dataset):
        raw_dataset(
            pd.DataFrame(
                {
                    "package": ["pandas", "numpy", "ghost", None],
                    "description": ["data analysis", None, None, "web stuff"],
                    "keywords": ["data, analysis", "data, math", None, float("nan")],
                    "popularity": [10, float("nan"), 3, 1],
                    "embedding": [np.array([1.0, 0, 0])] * 4,
                }
            )
        )

    def test_keywords_exact(self, dirty):
        assert [r["package"] for r in search_by_keywords("data", limit=10)] == [
            "pandas",
            "numpy",
        ]

    def test_keywords_jaccard(self, dirty):
        results = search_by_keywords("data", method="jaccard", limit=10)

        assert len(results) == 4
        assert all(isinstance(r["jaccard_score"], float) for r in results)

    def test_description_fts(self, dirty):
        assert [r["package"] for r in search_by_description("data")] == ["pandas"]

    @pytest.mark.parametrize("method", ["tf-idf", "cosine", "jaccard"])
    def test_find_similar(self, dirty, method):
        results = find_similar_packages("pandas", limit=3, method=method)

        assert len(results) == 3
        assert all(isinstance(r["description"], str) for r in results)
        assert all(isinstance(r["keywords"], str) for r in results)

    def test_missing_popularity_becomes_zero(self, dirty):
        assert search_by_package_name("numpy")["popularity"] == 0

    def test_missing_text_values_are_empty_strings_in_results(self, dirty):
        result = search_by_package_name("ghost")

        assert result["description"] == ""
        assert result["keywords"] == ""


class TestKeywordMatching:
    @pytest.fixture
    def kw_df(self, use_df):
        use_df(
            make_df(
                [
                    ("meta", "d", "metadata, tooling", 3, np.zeros(3)),
                    ("db", "d", "database, sql", 2, np.zeros(3)),
                    ("ds", "d", "Data Science, Machine Learning", 1, np.zeros(3)),
                    ("cpp", "d", "C++, systems", 0, np.zeros(3)),
                ]
            )
        )

    def test_does_not_match_inside_other_words(self, kw_df):
        assert [r["package"] for r in search_by_keywords("data")] == ["ds"]

    def test_dataset_keywords_are_case_insensitive(self, kw_df):
        assert [r["package"] for r in search_by_keywords("data science")] == ["ds"]
        assert [r["package"] for r in search_by_keywords("MACHINE")] == ["ds"]

    def test_multi_word_phrase(self, kw_df):
        assert [r["package"] for r in search_by_keywords("machine learning")] == ["ds"]
        assert search_by_keywords("learning machine") == []

    def test_special_characters(self, kw_df):
        assert [r["package"] for r in search_by_keywords("c++")] == ["cpp"]

    def test_surrounding_whitespace_ignored(self, kw_df):
        assert [r["package"] for r in search_by_keywords("  sql  ")] == ["db"]

    def test_jaccard_is_case_insensitive_on_dataset(self, kw_df):
        results = search_by_keywords(
            "data science", "machine learning", method="jaccard", limit=1
        )

        assert results[0]["package"] == "ds"
        assert results[0]["jaccard_score"] == pytest.approx(1.0)


class TestCaseInsensitiveReference:
    @pytest.mark.parametrize("name", ["pandas", "PANDAS", "Pandas", "  pandas "])
    @pytest.mark.parametrize("method", ["tf-idf", "cosine", "jaccard"])
    def test_find_similar_any_case(self, mock_load_data, name, method):
        results = find_similar_packages(name, limit=1, method=method)

        assert "pandas" not in [r["package"] for r in results]
        assert len(results) == 1

    def test_find_similar_mixed_case_package(self, mock_load_data):
        assert find_similar_packages("flask", limit=1)[0]["package"] != "Flask"

    def test_exact_case_wins_over_case_insensitive(self, use_df):
        use_df(
            make_df(
                [
                    ("flask", "lower", "k", 1, np.zeros(3)),
                    ("Flask", "upper", "k", 2, np.zeros(3)),
                ]
            )
        )

        assert search_by_package_name("Flask")["description"] == "upper"
        assert search_by_package_name("flask")["description"] == "lower"
        assert search_by_package_name("FLASK")["description"] == "lower"


class TestNativeTypes:
    def assert_native(self, record):
        assert type(record["package"]) is str
        assert type(record["description"]) is str
        assert type(record["keywords"]) is str
        assert type(record["popularity"]) is int
        for key in ("jaccard_score", "cosine_similarity"):
            if key in record:
                assert type(record[key]) is float

    def test_keywords_exact(self, mock_load_data):
        for r in search_by_keywords("data", limit=10):
            self.assert_native(r)

    def test_keywords_jaccard(self, mock_load_data):
        for r in search_by_keywords("data", method="jaccard", limit=10):
            self.assert_native(r)

    def test_package_name(self, mock_load_data):
        self.assert_native(search_by_package_name("pandas"))

    def test_description_fts(self, mock_load_data):
        for r in search_by_description("data", limit=10):
            self.assert_native(r)

    def test_description_cosine(self, mock_load_data, fake_sentence_transformers):
        for r in search_by_description("x", method="cosine", limit=10):
            self.assert_native(r)

    @pytest.mark.parametrize("method", ["tf-idf", "cosine", "jaccard"])
    def test_find_similar(self, mock_load_data, method):
        for r in find_similar_packages("pandas", limit=10, method=method):
            self.assert_native(r)

    def test_results_are_json_serialisable(self, mock_load_data):
        import json

        json.dumps(search_by_keywords("data", method="jaccard"))
        json.dumps(search_by_package_name("pandas"))
        json.dumps(find_similar_packages("pandas", method="jaccard"))


class TestIndexGuardrail:
    @pytest.fixture
    def offset_df(self, sample_df):
        df = sample_df.copy()
        df.index = [100, 7, 42, 3, 55]
        return df

    def test_all_functions_work_with_non_default_index(self, use_df, offset_df):
        use_df(offset_df)

        assert search_by_package_name("requests")["package"] == "requests"
        assert [r["package"] for r in search_by_keywords("web")] == [
            "requests",
            "Flask",
        ]
        assert search_by_keywords("data", method="jaccard", limit=1)[0]["package"] in {
            "pandas",
            "numpy",
            "scikit-learn",
        }
        assert [r["package"] for r in search_by_description("lightweight")] == ["Flask"]
        results = find_similar_packages("pandas", limit=1, method="cosine")
        assert results[0]["package"] == "numpy"
        assert find_similar_packages("pandas", limit=1)[0]["package"] == "scikit-learn"

    def test_load_data_resets_index(self, raw_dataset, sample_df):
        df = sample_df.copy()
        df.index = [100, 7, 42, 3, 55]
        raw_dataset(df)

        assert funcs.load_data().index.tolist() == [0, 1, 2, 3, 4]


class TestArgumentChecks:
    @pytest.mark.parametrize("keyword", [123, None, "", "   ", b"data", ["data"]])
    def test_keywords_must_be_non_empty_strings(self, mock_load_data, keyword):
        with pytest.raises(ValueError, match="Keywords must be non-empty strings"):
            search_by_keywords("data", keyword)

    @pytest.mark.parametrize("limit", [2.5, True, "3", None, 0, -1])
    def test_limit_must_be_a_positive_int(self, mock_load_data, limit):
        for call in (
            lambda: search_by_keywords("data", limit=limit),
            lambda: search_by_description("data", limit=limit),
            lambda: find_similar_packages("pandas", limit=limit),
        ):
            with pytest.raises(ValueError, match="Limit"):
                call()

    @pytest.mark.parametrize("value", [None, 123, "", "   ", ["pandas"]])
    def test_package_name_must_be_non_empty_string(self, mock_load_data, value):
        with pytest.raises(ValueError, match="Package name"):
            search_by_package_name(value)

    @pytest.mark.parametrize("value", [None, 123, "", "   "])
    def test_description_must_be_non_empty_string(self, mock_load_data, value):
        with pytest.raises(ValueError, match="Description"):
            search_by_description(value)

    @pytest.mark.parametrize("value", [None, 123, "", "   "])
    def test_reference_must_be_non_empty_string(self, mock_load_data, value):
        with pytest.raises(ValueError, match="Reference package"):
            find_similar_packages(value)

    def test_unexpected_kwargs_rejected(self, mock_load_data):
        with pytest.raises(TypeError, match="Unexpected keyword arguments.*'bogus'"):
            search_by_description("data", bogus=1)

    def test_typo_in_embedding_model_name_is_not_silently_ignored(self, mock_load_data):
        with pytest.raises(TypeError, match="embeding_model"):
            search_by_description("data", method="cosine", embeding_model="x")

    @pytest.mark.parametrize("value", ["", "  ", None, 5])
    def test_embedding_model_must_be_non_empty_string(self, mock_load_data, value):
        with pytest.raises(ValueError, match="embedding_model"):
            search_by_description("data", method="cosine", embedding_model=value)

    def test_validation_happens_before_loading_data(self, monkeypatch):
        def boom():
            raise AssertionError("dataset should not be loaded for invalid input")

        monkeypatch.setattr("modulift.api.load_data", boom)

        with pytest.raises(ValueError):
            search_by_keywords("data", limit=0)
        with pytest.raises(ValueError):
            search_by_package_name("")
        with pytest.raises(TypeError):
            search_by_description("x", nope=1)


class TestCaching:
    @pytest.fixture
    def counted(self, raw_dataset, sample_df):
        return raw_dataset(sample_df)

    def test_dataset_loaded_once_across_calls(self, counted):
        search_by_keywords("data")
        search_by_keywords("data", method="jaccard")
        search_by_package_name("pandas")
        search_by_description("data")
        find_similar_packages("pandas")

        assert len(counted) == 1

    def test_derived_data_is_cached(self, counted):
        find_similar_packages("pandas", method="tf-idf")
        find_similar_packages("numpy", method="tf-idf")
        find_similar_packages("pandas", method="cosine")
        find_similar_packages("numpy", method="cosine")
        search_by_keywords("data", method="jaccard")
        search_by_keywords("web", method="jaccard")

        assert api._tfidf_matrix.cache_info().misses == 1
        assert api._tfidf_matrix.cache_info().hits >= 1
        assert api._embedding_matrix.cache_info().misses == 1
        assert api._embedding_matrix.cache_info().hits >= 1
        assert api._keyword_sets.cache_info().misses == 1

    def test_model_loaded_once_per_name(
        self, mock_load_data, fake_sentence_transformers, capsys
    ):
        search_by_description("a", method="cosine")
        search_by_description("b", method="cosine")
        search_by_description("c", method="cosine", embedding_model="my/model")
        search_by_description("d", method="cosine", embedding_model="my/model")

        assert fake_sentence_transformers == [
            "sentence-transformers/all-MiniLM-L6-v2",
            "my/model",
        ]
        # The default-model notice appears only when the default model is actually loaded.
        assert capsys.readouterr().out.count("default embedding model") == 1

    def test_calls_do_not_mutate_shared_dataset(self, raw_dataset, sample_df):
        raw_dataset(sample_df)
        before = funcs.load_data().copy()

        search_by_keywords("data", method="jaccard")
        find_similar_packages("pandas", method="jaccard")
        search_by_description("data")

        pd.testing.assert_frame_equal(funcs.load_data(), before)
        assert "keyword_set" not in funcs.load_data().columns

    def test_missing_embedding_column_gives_clear_error(self, use_df, sample_df):
        use_df(sample_df.drop(columns=["embedding"]))

        with pytest.raises(RuntimeError, match="embedding"):
            find_similar_packages("pandas", method="cosine")


class TestVersion:
    def test_dunder_version_matches_installed_metadata(self):
        """__version__ is hardcoded, so guard against forgetting to bump it."""
        from importlib.metadata import version

        import modulift

        assert modulift.__version__ == version("modulift")
