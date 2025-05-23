import pandas as pd
from typing import Literal, List, Dict
from .funcs import load_data, jaccard_similarity, markdown_print


def search_by_keywords(
    *args,
    relation: Literal["and", "or"] = "or",
    limit: int = 5,
    method: Literal["exact", "jaccard"] = "exact",
    markdown: bool = False,
) -> List[Dict]:
    """
    Search for packages by keywords.
    This function allows you to search for packages based on keywords. You can specify the relation between the keywords
    (AND/OR) and the method of searching (exact match or Jaccard similarity). The results can be limited to a certain number
    and can be printed in Markdown format.

    Example:
        search_by_keywords("data", "science", relation="and", limit=10, method="exact", markdown=True)

    Args:
        relation (Literal["and", "or"], optional): The relation between the keywords, or: any of the keywords must be present, and: all of the keywords must be present. Defaults to "or".
        limit (int, optional): The maximum number of results to return. Defaults to 5.
        method (Literal["exact", "jaccard"], optional): The method of searching, exact: exact match, jaccard: Jaccard similarity. Defaults to "exact".
        markdown (bool, optional): If True, the results will be printed in Markdown format. Defaults to False.

    Returns:
        List[Dict]: A list of dictionaries containing the package name, description, keywords, and popularity. Example:
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
    if limit <= 0:
        raise ValueError("Limit must be a positive integer.")
    
    keywords = [arg.lower() for arg in args]
    
    # Load the dataset
    df = load_data()

    # Filter the dataset based on the keywords
    if method == "exact":
        if relation == "or":
            mask = df["keywords"].apply(lambda cell: any(keyword in str(cell) for keyword in keywords))
        elif relation == "and":
            mask = df["keywords"].apply(lambda cell: all(keyword in str(cell) for keyword in keywords))
        
        results = df[mask][["package", "description", "keywords", "popularity"]] \
                            .sort_values("popularity", ascending=False) \
                            .head(limit) \
                            .to_dict(orient="records")
    elif method == "jaccard":
        df["keyword_set"] = df["keywords"].apply(lambda x: set(x.split(", ")))
        jaccard_scores = df["keyword_set"].apply(lambda x: jaccard_similarity(x, keywords))
        jaccard_scores = pd.Series(jaccard_scores).sort_values(ascending=False).head(limit)
        results = [
            {
                "package": df.iloc[i]["package"],
                "description": df.iloc[i]["description"],
                "keywords": df.iloc[i]["keywords"],
                "popularity": df.iloc[i]["popularity"],
                "jaccard_score": jaccard_scores[i],
            }
            for i in jaccard_scores.index
        ]

    if markdown:
        markdown_print(results)
    
    return results
        
    
def search_by_package_name(
    package_name: str,
    markdown: bool = False,
) -> Dict[str, str]:
    """
    Search for a package by its name.
    This function allows you to search for a package based on its name. If the package is found, its details will be returned.

    Example:
        search_by_package_name("example-package", markdown=True)
    
    Args:
        package_name (str): The name of the package to search for.
        markdown (bool, optional): If True, the results will be printed in Markdown format. Defaults to False.

    Returns:
        Dict[str, str]: A dictionary containing the package name, description, keywords, and popularity. Example:
        {
            "package": "example-package",
            "description": "An example package for demonstration purposes.",
            "keywords": "example, demo, test",
            "popularity": 4,
        }
    """
    # Validation
    if not package_name:
        raise ValueError("Package name must be provided.")
    
    df = load_data()

    # Filter the dataset based on the package name
    result_df = df[df["package"] == package_name][["package", "description", "keywords", "popularity"]]
    
    if result_df.empty:
        result_df = df[df["package"].str.lower() == package_name.lower()][["package", "description", "keywords", "popularity"]]

    if result_df.empty:
        raise ValueError(f"No package found with the name '{package_name}'.")
    
    result = result_df.iloc[0].to_dict()

    if markdown:
        markdown_print([result])
    
    return result


def search_by_description(
    description: str,
    limit: int = 5,
    method: Literal["fts", "cosine"]="fts",
    markdown: bool = False,
    **kwargs
) -> List[Dict]:
    """
    Search for packages by description.
    This function allows you to search for packages based on their description. You can specify the method of searching
    (full-text search or cosine similarity) and the number of results to return. The results can be printed in Markdown format.

    Example:
        search_by_description("data analysis", limit=10, method="fts", markdown=True)

    Args:
        description (str): The description to search for.
        limit (int, optional): The maximum number of results to return. Defaults to 5.
        method (Literal["fts", "cosine"], optional): The method of searching, fts: full-text search, cosine: cosine similarity. Defaults to "fts".
        markdown (bool, optional): _description_. Defaults to False.

    Returns:
        List[Dict]: A list of dictionaries containing the package name, description, keywords, and popularity. Example:
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
    if not description:
        raise ValueError("Description must be provided.")
    if limit <= 0:
        raise ValueError("Limit must be a positive integer.")
    if method not in ["fts", "cosine"]:
        raise ValueError("Method must be 'fts' or 'cosine'.")
    
    df = load_data()

    # Filter the dataset based on the description
    if method == "fts":
        mask = df["description"].str.contains(description, case=False, na=False)

        results = df[mask][["package", "description", "keywords", "popularity"]] \
                            .sort_values("popularity", ascending=False) \
                            .head(limit) \
                            .to_dict(orient="records")
    elif method == "cosine":
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from sentence_transformers import SentenceTransformer

        embedding_model = kwargs.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        if embedding_model == "sentence-transformers/all-MiniLM-L6-v2":
            print("Using default embedding model: sentence-transformers/all-MiniLM-L6-v2." 
                  "If you want to use a different model, please specify it under the 'embedding_model' parameter."
                  )
        model = SentenceTransformer(embedding_model)

        query_embedding = model.encode(description, convert_to_numpy=True)
        embeddings = np.stack(df["embedding"].values)
        cosine_similarities = cosine_similarity([query_embedding], embeddings).flatten()
        cosine_similarities = pd.Series(cosine_similarities).sort_values(ascending=False).head(limit)
        results = [
            {
                "package": df.iloc[i]["package"],
                "description": df.iloc[i]["description"],
                "keywords": df.iloc[i]["keywords"],
                "popularity": df.iloc[i]["popularity"],
                "cosine_similarity": cosine_similarities[i],
            }
            for i in cosine_similarities.index
        ]

    if markdown:
        markdown_print(results)
    
    return results


def find_similar_packages(
    reference_package: str,
    limit: int = 5,
    method: Literal["tf-idf", "cosine", "jaccard"] = "tf-idf",
    markdown: bool = False,
) -> List[Dict]:
    """
    Find similar packages based on a reference package.
    This function allows you to find similar packages based on a reference package. You can specify the method of searching
    (TF-IDF, cosine similarity, or Jaccard similarity) and the number of results to return. The results can be printed in Markdown format.

    Example:
        find_similar_packages("example-package", limit=10, method="tf-idf", markdown=True)

    Args:
        reference_package (str): The name of the reference package to search for.
        limit (int, optional): The maximum number of results to return. Defaults to 5.
        method (Literal[&quot;tf, optional): The method of searching, tf-idf: TF-IDF similarity, cosine: cosine similarity, jaccard: Jaccard similarity. Defaults to "tf-idf".
        markdown (bool, optional): If True, the results will be printed in Markdown format. Defaults to False.

    Returns:
        List[Dict]: A list of dictionaries containing the package name, description, keywords, and popularity. Example:
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
    if not reference_package:
        raise ValueError("Reference package name must be provided.")
    if limit <= 0:
        raise ValueError("Limit must be a positive integer.")
    if method not in ["tf-idf", "cosine", "jaccard"]:
        raise ValueError("Method must be 'tf-idf', 'cosine', or 'jaccard'.")
    
    df = load_data()

    package_row = df.query("package == @reference_package")
    if package_row.empty:
        raise ValueError(f"No package found with the name '{reference_package}'. Ensure the name is correct.")

    reference_idx = package_row.index[0]

    if method == "tf-idf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        combined_text = df["description"] + " " + df["keywords"]
        tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)
        cosine_similarities = cosine_similarity(tfidf_matrix[reference_idx], tfidf_matrix).flatten()
        candidates = pd.Series(cosine_similarities).sort_values(ascending=False).head(limit+1).index[1:]

    elif method == "cosine":
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        query_embedding = df.iloc[reference_idx]["embedding"]
        embeddings = np.stack(df["embedding"].values)
        cosine_similarities = cosine_similarity([query_embedding], embeddings).flatten()
        candidates = pd.Series(cosine_similarities).sort_values(ascending=False).head(limit+1).index[1:]
    
    elif method == "jaccard":
        df["keyword_set"] = df["keywords"].apply(lambda x: set(x.split(", ")))
        query_keywords = df.iloc[reference_idx]["keywords"]
        query_set = set(query_keywords.split(", "))
        df_filtered = df["keyword_set"].apply(lambda x: jaccard_similarity(x, query_set))
        candidates = pd.Series(df_filtered).sort_values(ascending=False).head(limit+1).index[1:]

    results = [
        {
            "package": df.iloc[i]["package"],
            "description": df.iloc[i]["description"],
            "keywords": df.iloc[i]["keywords"],
            "popularity": int(df.iloc[i]["popularity"]),

        }
        for i in candidates
    ]

    if markdown:
        markdown_print(results)

    return results