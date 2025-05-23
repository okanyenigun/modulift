import pandas as pd
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError, DataFilesNotFoundError
from typing import Sequence, Any, List, Dict


def load_data() -> pd.DataFrame:
    """
    Load the dataset from the Hugging Face Hub.
    """
    try:
        dataset = load_dataset("yeniguno/pypi")["train"]
        return dataset.to_pandas()
    except DatasetNotFoundError as e:
        print(f"🚫 Dataset not found: {e}")
    except DataFilesNotFoundError as e:
        print(f"🚫 Dataset repo exists but no matching files: {e}")
        

def jaccard_similarity(set1: Sequence, set2: Sequence) -> float:
    """
    Calculate the Jaccard similarity between two sets.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def markdown_print(data: List[Dict[str, Any]]) -> None:
    """
    Print the data in Markdown format.
    """
    from IPython.display import display, Markdown

    markdown_text= ""

    for item in data:
        for key, value in item.items():
            markdown_text += f"**{key.capitalize()}:** {value}\n\n"
        markdown_text += "\n---\n"

    display(Markdown(markdown_text))
    return