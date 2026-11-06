import logging
import operator
import re
from functools import lru_cache, reduce
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd

from . import funcs
from .funcs import jaccard_similarity, load_data, markdown_print

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Library code logs instead of printing, so it never writes to the caller's stdout.
# Enable with e.g. logging.basicConfig(level=logging.INFO).
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_limit(limit: Any) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Limit must be a positive integer.")


def _validate_text(value: Any, message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    return value.strip()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _get_df() -> pd.DataFrame:
    """
    Return the dataset with a clean 0..n-1 index. Several code paths convert
    index labels into row positions, so this must hold.
    """
    df = load_data()
    if not df.index.equals(pd.RangeIndex(len(df))):
        df = df.reset_index(drop=True)
    return df


def _record(row: Any, **extra: Any) -> Dict[str, Any]:
    """Build a result dictionary made of plain Python types."""
    record = {
        "package": str(row["package"]),
        "description": str(row["description"]),
        "keywords": str(row["keywords"]),
        "popularity": int(row["popularity"]),
    }
    record.update(extra)
    return record


def _scored_results(
    df: pd.DataFrame, scores: pd.Series, score_name: str
) -> List[Dict[str, Any]]:
    return [
        _record(df.iloc[int(i)], **{score_name: float(score)})
        for i, score in scores.items()
    ]


def _top(scores: pd.Series, limit: int) -> pd.Series:
    # A stable sort keeps results deterministic when scores tie.
    return scores.sort_values(ascending=False, kind="stable").head(limit)


def _find_package_index(df: pd.DataFrame, name: str) -> int:
    """Case-insensitive package lookup (an exact-case match wins if there is one)."""
    matches = df.index[df["package"] == name]
    if len(matches) == 0:
        matches = df.index[df["package"].str.lower() == name.lower()]
    if len(matches) == 0:
        raise ValueError(
            f"No package found with the name '{name}'. Ensure the name is correct."
        )
    return int(matches[0])


def _split_keywords(text: str) -> set:
    return {token.strip() for token in text.split(",") if token.strip()}


def _whole_word_pattern(keyword: str) -> str:
    return rf"(?<!\w){re.escape(keyword)}(?!\w)"


# ---------------------------------------------------------------------------
# Cached derived data (computed once per process from the cached dataset)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _keywords_lower() -> pd.Series:
    return _get_df()["keywords"].str.lower()


@lru_cache(maxsize=1)
def _keyword_sets() -> pd.Series:
    return _keywords_lower().apply(_split_keywords)


@lru_cache(maxsize=1)
def _tfidf_matrix():
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = _get_df()
    text = df["description"] + " " + df["keywords"]
    return TfidfVectorizer(stop_words="english").fit_transform(text)


@lru_cache(maxsize=1)
def _embedding_matrix() -> np.ndarray:
    df = _get_df()
    if "embedding" not in df.columns:
        raise RuntimeError(
            "The dataset has no 'embedding' column, so cosine similarity is unavailable."
        )
    return np.stack(df["embedding"].values)


@lru_cache(maxsize=4)
def _get_model(name: str):
    from sentence_transformers import SentenceTransformer

    if name == DEFAULT_EMBEDDING_MODEL:
        logger.info(
            "Using default embedding model: %s. Pass 'embedding_model' to use a different one.",
            name,
        )
    else:
        logger.info("Loading embedding model: %s", name)
    return SentenceTransformer(name)


def _clear_caches() -> None:
    """Drop the cached dataset and everything derived from it (mainly for tests)."""
    for cached in (
        _keywords_lower,
        _keyword_sets,
        _tfidf_matrix,
        _embedding_matrix,
        _get_model,
        funcs.load_data,
    ):
        cached.cache_clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_by_keywords(
    *args: str,
    relation: Literal["and", "or"] = "or",
    limit: int = 5,
    method: Literal["exact", "jaccard"] = "exact",
    markdown: bool = False,
) -> List[Dict[str, Any]]:
    """
    Search for packages by keywords.
    This function allows you to search for packages based on keywords. You can specify the relation between the keywords
    (AND/OR) and the method of searching (exact match or Jaccard similarity). The results can be limited to a certain number
    and can be displayed in Markdown format. Matching is case-insensitive.

    With the "exact" method a keyword matches when it appears as a whole word (or phrase) in a package's keywords, so
    "data" matches "data science" but not "metadata". With the "jaccard" method the top `limit` packages are always
    returned, ranked by the overlap between their keywords and the given keywords.

    Example:
        search_by_keywords("data", "science", relation="and", limit=10, method="exact", markdown=True)

    Args:
        *args (str): The keywords to search for. At least one is required.
        relation (Literal["and", "or"], optional): The relation between the keywords, or: any of the keywords must be present, and: all of the keywords must be present. Only used by the "exact" method. Defaults to "or".
        limit (int, optional): The maximum number of results to return. Defaults to 5.
        method (Literal["exact", "jaccard"], optional): The method of searching, exact: whole-word match, jaccard: Jaccard similarity. Defaults to "exact".
        markdown (bool, optional): If True, the results will also be displayed in Markdown format (requires IPython/Jupyter). Defaults to False.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the package name, description, keywords, and popularity
        (plus a "jaccard_score" for the "jaccard" method). Example:
        [
            {
                "package": "example-package",
                "description": "An example package for demonstration purposes.",
                "keywords": "example, demo, test",
                "popularity": 4,
            },
            ...
        ]
    """

    # Validation
    if not args:
        raise ValueError("At least one keyword must be provided.")
    if relation not in ["and", "or"]:
        raise ValueError("Relation must be 'and' or 'or'.")
    if method not in ["exact", "jaccard"]:
        raise ValueError("Method must be 'exact' or 'jaccard'.")
    _validate_limit(limit)
    if any(not isinstance(arg, str) or not arg.strip() for arg in args):
        raise ValueError("Keywords must be non-empty strings.")

    keywords = [arg.strip().lower() for arg in args]

    df = _get_df()

    if method == "exact":
        lowered = _keywords_lower()
        masks = [
            lowered.str.contains(_whole_word_pattern(keyword), regex=True, na=False)
            for keyword in keywords
        ]
        combine = operator.or_ if relation == "or" else operator.and_
        mask = reduce(combine, masks)

        matched = df[mask].sort_values("popularity", ascending=False, kind="stable")
        results = [_record(row) for row in matched.head(limit).to_dict("records")]
    else:
        query = set(keywords)
        scores = _keyword_sets().apply(lambda tokens: jaccard_similarity(tokens, query))
        results = _scored_results(df, _top(scores, limit), "jaccard_score")

    if markdown:
        markdown_print(results)

    return results


def search_by_package_name(
    package_name: str,
    markdown: bool = False,
) -> Dict[str, Any]:
    """
    Search for a package by its name.
    This function allows you to search for a package based on its name (case-insensitive). If the package is found,
    its details will be returned.

    Example:
        search_by_package_name("example-package", markdown=True)

    Args:
        package_name (str): The name of the package to search for.
        markdown (bool, optional): If True, the result will also be displayed in Markdown format (requires IPython/Jupyter). Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing the package name, description, keywords, and popularity. Example:
        {
            "package": "example-package",
            "description": "An example package for demonstration purposes.",
            "keywords": "example, demo, test",
            "popularity": 4,
        }
    """
    # Validation
    package_name = _validate_text(
        package_name, "Package name must be a non-empty string."
    )

    df = _get_df()
    result = _record(df.iloc[_find_package_index(df, package_name)])

    if markdown:
        markdown_print([result])

    return result


def search_by_description(
    description: str,
    limit: int = 5,
    method: Literal["fts", "cosine"] = "fts",
    markdown: bool = False,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Search for packages by description.
    This function allows you to search for packages based on their description. You can specify the method of searching
    (full-text search or cosine similarity) and the number of results to return. The results can be displayed in Markdown format.

    With "fts" the description is matched as a case-insensitive plain-text substring. With "cosine" the top `limit`
    packages are always returned, ranked by embedding similarity.

    Example:
        search_by_description("data analysis", limit=10, method="fts", markdown=True)

    Args:
        description (str): The description to search for.
        limit (int, optional): The maximum number of results to return. Defaults to 5.
        method (Literal["fts", "cosine"], optional): The method of searching, fts: full-text search, cosine: cosine similarity. Defaults to "fts".
        markdown (bool, optional): If True, the results will also be displayed in Markdown format (requires IPython/Jupyter). Defaults to False.
        **kwargs: Only `embedding_model` (str) is accepted; it names the sentence-transformers model used by the "cosine" method. Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the package name, description, keywords, and popularity
        (plus a "cosine_similarity" for the "cosine" method). Example:
        [
            {
                "package": "example-package",
                "description": "An example package for demonstration purposes.",
                "keywords": "example, demo, test",
                "popularity": 4,
            },
            ...
        ]
    """
    # Validation
    description = _validate_text(description, "Description must be a non-empty string.")
    _validate_limit(limit)
    if method not in ["fts", "cosine"]:
        raise ValueError("Method must be 'fts' or 'cosine'.")
    embedding_model = kwargs.pop("embedding_model", DEFAULT_EMBEDDING_MODEL)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}.")
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        raise ValueError("embedding_model must be a non-empty string.")

    df = _get_df()

    if method == "fts":
        mask = df["description"].str.contains(
            description, case=False, regex=False, na=False
        )
        matched = df[mask].sort_values("popularity", ascending=False, kind="stable")
        results = [_record(row) for row in matched.head(limit).to_dict("records")]
    else:
        from sklearn.metrics.pairwise import cosine_similarity

        model = _get_model(embedding_model.strip())
        query_embedding = model.encode(description, convert_to_numpy=True)
        similarities = cosine_similarity(
            [query_embedding], _embedding_matrix()
        ).flatten()
        results = _scored_results(
            df, _top(pd.Series(similarities), limit), "cosine_similarity"
        )

    if markdown:
        markdown_print(results)

    return results


def find_similar_packages(
    reference_package: str,
    limit: int = 5,
    method: Literal["tf-idf", "cosine", "jaccard"] = "tf-idf",
    markdown: bool = False,
) -> List[Dict[str, Any]]:
    """
    Find similar packages based on a reference package.
    This function allows you to find similar packages based on a reference package (looked up case-insensitively).
    You can specify the method of searching (TF-IDF, cosine similarity, or Jaccard similarity) and the number of
    results to return. The reference package itself is never part of the results. The results can be displayed in
    Markdown format.

    Example:
        find_similar_packages("example-package", limit=10, method="tf-idf", markdown=True)

    Args:
        reference_package (str): The name of the reference package to search for.
        limit (int, optional): The maximum number of results to return. Defaults to 5.
        method (Literal["tf-idf", "cosine", "jaccard"], optional): The method of searching, tf-idf: TF-IDF similarity, cosine: cosine similarity of embeddings, jaccard: Jaccard similarity of keywords. Defaults to "tf-idf".
        markdown (bool, optional): If True, the results will also be displayed in Markdown format (requires IPython/Jupyter). Defaults to False.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the package name, description, keywords, and popularity. Example:
        [
            {
                "package": "example-package",
                "description": "An example package for demonstration purposes.",
                "keywords": "example, demo, test",
                "popularity": 4,
            },
            ...
        ]
    """
    # Validation
    reference_package = _validate_text(
        reference_package, "Reference package name must be a non-empty string."
    )
    _validate_limit(limit)
    if method not in ["tf-idf", "cosine", "jaccard"]:
        raise ValueError("Method must be 'tf-idf', 'cosine', or 'jaccard'.")

    df = _get_df()
    reference_idx = _find_package_index(df, reference_package)

    if method == "tf-idf":
        from sklearn.metrics.pairwise import cosine_similarity

        matrix = _tfidf_matrix()
        similarities = cosine_similarity(matrix[reference_idx], matrix).flatten()
    elif method == "cosine":
        from sklearn.metrics.pairwise import cosine_similarity

        matrix = _embedding_matrix()
        similarities = cosine_similarity([matrix[reference_idx]], matrix).flatten()
    else:
        sets = _keyword_sets()
        query_set = sets.iloc[reference_idx]
        similarities = sets.apply(
            lambda tokens: jaccard_similarity(tokens, query_set)
        ).to_numpy()

    # Remove the reference by position rather than assuming it ranks first:
    # another package can tie with it (e.g. identical descriptions).
    scores = pd.Series(np.asarray(similarities, dtype=float)).drop(reference_idx)
    candidates = _top(scores, limit).index

    results = [_record(df.iloc[int(i)]) for i in candidates]

    if markdown:
        markdown_print(results)

    return results
