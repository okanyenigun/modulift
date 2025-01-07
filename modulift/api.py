from IPython.display import display, Markdown
from typing import List, Optional
from .data import load_data


def search_by_keywords(*args: str, relation: str = "or", limit: Optional[int] = None, markdown: bool = False) -> List[dict]:
    """
    Search for keywords in the 'keywords' column with a specified relation ('or' or 'and') and optional limit on results.
    
    Parameters:
    *args (str): Keywords to search for in the 'keywords' column.
    relation (str, optional): Defines the logical relationship for keyword matching. 
                              Use "or" for matching any keyword, and "and" for matching all keywords. 
                              Defaults to "or".
    limit (int, optional): The maximum number of results to return. Defaults to None, which returns all matches.
    markdown (bool, optional): If True, prints the results in markdown format. Defaults to False.
    
    Returns:
    List[dict]: A list of dictionaries with 'package', 'description', 'keywords', and 'popularity' fields 
                for each row that matches the search criteria, sorted by 'popularity' in the specified order.
    
    Notes:
    - The 'popularity' column is sorted in the order ['very popular', 'popular', 'not popular'].
    - Markdown output is only printed and does not affect the returned result structure.

    Example:
    >>> search_by_keywords("machine learning", "data science", relation="and", limit=5, markdown=True)
    [
        {
            'package': 'example-package',
            'description': 'An example package for machine learning.',
            'keywords': 'machine learning, data science, AI',
            'popularity': 'very popular'
        },
        ...
    ]
    """
    # Validate and normalize inputs
    if not args:
        raise ValueError("At least one keyword must be provided for search.")
    
    if relation not in {"or", "and"}:
        raise ValueError("The 'relation' parameter must be either 'or' or 'and'.")
    
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("The 'limit' parameter must be a positive integer if specified.")
    
    # Load the DataFrame
    df = load_data()
    
    # Convert the input keywords to a list
    keywords = list(args)
    
    # Define the boolean mask based on 'or' or 'and' logic
    if relation == "and":
        mask = df['keywords'].apply(lambda cell: all(keyword.lower() in str(cell).lower() for keyword in keywords))
    else:  # default to 'or' logic
        mask = df['keywords'].apply(lambda cell: any(keyword.lower() in str(cell).lower() for keyword in keywords))
    
    # Filter the DataFrame with matching rows and select required columns
    result_df = df[mask][['package', 'description', 'keywords', 'popularity']]
    
    # Sort the DataFrame by 'POPULARITY' based on the defined order
    result_df = result_df.sort_values('popularity', ascending=False)
    
    # Apply limit if specified
    if limit is not None:
        result_df = result_df.head(limit)
    
    # Convert to a list of dictionaries
    results = result_df.to_dict(orient='records')
    
    # Print in markdown format if markdown is True
    if markdown:
        for i, row in enumerate(results, 1):

            package = row['package'].strip()
            description = row['description'].strip()
            keywords = row['keywords'].strip()
            popularity = str(row['popularity'])

            markdown_text = f"""
**{package}**

- **Description**: {description}
- **Keywords**: {keywords}
- **Popularity**: {popularity}

---
"""
            display(Markdown(markdown_text))

    return results

def search_by_package_name(package_name: str, markdown: bool = False) -> Optional[dict]:
    """
    Search for an exact package name in the 'package' column of the DataFrame.
    
    Parameters:
    package_name (str): The exact package name to search for.
    markdown (bool, optional): If True, prints the result in markdown format. Defaults to False.
    
    Returns:
    Optional[dict]: A dictionary with 'package', 'description', 'keywords', and 'popularity' fields if a match is found;
                    otherwise, returns None.
    
    Example:
    >>> search_by_package_name("example-package", markdown=True)
    {
        'package': 'example-package',
        'description': 'An example package for data processing.',
        'keywords': 'data processing, analysis, example',
        'popularity': 'very popular'
    }
    """
    if not isinstance(package_name, str) or not package_name.strip():
        raise ValueError("The 'package_name' parameter must be a non-empty string.")

    # Load the DataFrame
    df = load_data()
    
    # Perform exact match search in the 'package' column
    result_df = df[df['package'] == package_name][['package', 'description', 'keywords', 'popularity']]
    
    # Check if there is a match
    if not result_df.empty:
        result = result_df.iloc[0].to_dict()
        
        # Print in markdown format if markdown is True
        if markdown:

            package = result['package'].strip()
            description = result['description'].strip()
            keywords = result['keywords'].strip()
            popularity = str(result['popularity'])

            markdown_text = f"""
**{package}**

- **Description**: {description}
- **Keywords**: {keywords}
- **Popularity**: {popularity}

---
"""

            display(Markdown(markdown_text))
        
        return result
    else:
        if markdown:
            print(f"No match found for package name '{package_name}'.")
        return None
    
    
def search_by_description(text: str, limit: Optional[int] = None, markdown: bool = False) -> List[dict]:
    """
    Perform a full-text search in the 'description' column for a specific text, with an optional limit and markdown output.
    
    Parameters:
    text (str): The text to search for in the 'description' column.
    limit (int, optional): The maximum number of results to return. Defaults to None, which returns all matches.
    markdown (bool, optional): If True, prints the results in markdown format. Defaults to False.
    
    Returns:
    List[dict]: A list of dictionaries with 'package', 'description', 'keywords', and 'popularity' fields
                for rows where the 'description' column contains the search text, limited by 'limit' if specified.
    
    Example:
    >>> search_by_description("data analysis", limit=5, markdown=True)
    [
        {
            'package': 'example-package',
            'description': 'A package for data analysis and visualization.',
            'keywords': 'data, analysis, visualization',
            'popularity': 'very popular'
        },
        ...
    ]
    """
    # Validate 'text' input
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The 'text' parameter must be a non-empty string.")

    # Validate 'limit' parameter
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("The 'limit' parameter must be a positive integer if specified.")
    
    # Load the DataFrame
    df = load_data()
    
    # Perform case-insensitive full-text search in the 'description' column
    mask = df['description'].str.contains(text, case=False, na=False)
    
    # Filter the DataFrame with matching rows and select required columns
    result_df = df[mask][['package', 'description', 'keywords', 'popularity']]

    # Sort the DataFrame by 'POPULARITY' based on the defined order
    result_df = result_df.sort_values('popularity', ascending=False)
    
    # Apply limit if specified
    if limit is not None:
        result_df = result_df.head(limit)
    
    # Convert to a list of dictionaries
    results = result_df.to_dict(orient='records')
    
    # Print in markdown format if markdown is True
    if markdown:
        for i, row in enumerate(results, 1):

            package = row['package'].strip()
            description = row['description'].strip()
            keywords = row['keywords'].strip()
            popularity = str(row['popularity'])
            
            markdown_text = f"""
**{package}**

- **Description**: {description}
- **Keywords**: {keywords}
- **Popularity**: {popularity}

---
"""
            display(Markdown(markdown_text))
    else:
        print(f"No matches found for the text '{text}'.")
    return results