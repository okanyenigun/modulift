from datasets import load_dataset

def load_data():
    dataset = load_dataset("yeniguno/pypi")["train"]
    df = dataset.to_pandas()
    return df