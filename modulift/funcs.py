import pandas as pd
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError, DataFilesNotFoundError
from functools import lru_cache
from typing import Iterable, Any, List, Dict

REQUIRED_COLUMNS = ("package", "description", "keywords", "popularity")
TEXT_COLUMNS = ("package", "description", "keywords")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the raw dataset so the rest of the package can rely on it:
    required columns exist, text columns hold strings (never NaN/None),
    popularity is an integer and the index is a clean 0..n-1 range.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise RuntimeError(
            f"Dataset 'yeniguno/pypi' is missing required columns: {missing}."
        )

    df = df.reset_index(drop=True)
    for column in TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype(str)
    df["popularity"] = (
        pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)
    )
    return df


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """
    Load the dataset from the Hugging Face Hub.

    The result is cached for the lifetime of the process, so it is shared between
    calls and must not be modified in place.

    Raises:
        RuntimeError: If the dataset cannot be loaded or is empty/malformed.
    """
    try:
        dataset = load_dataset("yeniguno/pypi")["train"]
    except DatasetNotFoundError as e:
        raise RuntimeError(f"Dataset 'yeniguno/pypi' not found: {e}") from e
    except DataFilesNotFoundError as e:
        raise RuntimeError(
            f"Dataset 'yeniguno/pypi' exists but has no matching files: {e}"
        ) from e
    except KeyError as e:
        raise RuntimeError("Dataset 'yeniguno/pypi' has no 'train' split.") from e
    except OSError as e:
        # Covers connection problems (requests/huggingface_hub errors derive from it).
        raise RuntimeError(
            f"Could not download dataset 'yeniguno/pypi'. Check your internet connection: {e}"
        ) from e

    df = dataset.to_pandas()
    if df is None or df.empty:
        raise RuntimeError("Dataset 'yeniguno/pypi' was loaded but is empty.")
    return _prepare(df)


def jaccard_similarity(set1: Iterable, set2: Iterable) -> float:
    """
    Calculate the Jaccard similarity between two collections, treated as sets.
    """
    a, b = set(set1), set(set2)
    union = len(a | b)
    return len(a & b) / union if union != 0 else 0.0


def markdown_print(data: List[Dict[str, Any]]) -> None:
    """
    Print the data in Markdown format.
    """
    from IPython.display import display, Markdown

    markdown_text = ""

    for item in data:
        for key, value in item.items():
            markdown_text += f"**{key.capitalize()}:** {value}\n\n"
        markdown_text += "\n---\n"

    display(Markdown(markdown_text))
