# Modulift

**Modulift** is a powerful and efficient Python package designed to help developers, data scientists, and researchers explore and discover Python packages on PyPI based on specific keywords, package names, or full-text search within package descriptions. With a comprehensive dataset of PyPI packages, Modulift allows you to find relevant libraries for your project needs, making it an invaluable tool for anyone looking to streamline package selection and integration.

## Key Features

- **Exact Package Name Search**: Retrieve detailed information on a specific package by name.
- **Keyword-Based Search**: Search packages by multiple keywords with flexible "or" and "and" logic to refine results.
- **Full-Text Description Search**: Locate packages by searching for specific phrases or content in their descriptions.
- **Markdown Output Option**: Print search results in Markdown format, ideal for documentation or presentations.

## Installation

Install Modulift via pip:

```bash
pip install modulift
```

## Usage

Modulift provides an easy-to-use API for searching and exploring Python packages. Here’s a quick guide on how to use its core features.

### 1. Search by Package Name

Retrieve detailed information for a specific package by name:

```python
import modulift as mf

result = mf.search_by_package_name("numpy")
print(result)
```

**Output**:

```python
{
    'package': 'numpy',
    'description': 'NumPy is the fundamental package for scientific computing with Python. It provides powerful N-dimensional arrays, sophisticated mathematical functions, tools for integrating C/C++ and Fortran code, and useful linear algebra, Fourier transform, and random number capabilities.',
    'keywords': 'scientific computing, array, linear algebra, Fourier transform, random number generation, data processing, numerical analysis',
    'popularity': 5
}
```

For a different package:

```python
result = mf.search_by_package_name("nullsweep")
print(result)
```

**Output**:

```python
{
    'package': 'nullsweep',
    'description': 'NullSweep is a Python library designed for detecting and handling patterns of missing data in pandas DataFrames. It provides a simple API to identify global missing data patterns across the entire dataset, patterns related to specific features within the dataset, and to impute missing values using various strategies.',
    'keywords': 'missing data, data imputation, pandas, pattern detection, feature-specific patterns',
    'popularity': 4
}
```

Markdown Output
To display the result in Markdown format:

```python
result = mf.search_by_package_name("numpy", markdown=True)
```

**Package**: numpy

- **Description**: NumPy is the fundamental package for scientific computing with Python. It provides powerful N-dimensional arrays, sophisticated mathematical functions, tools for integrating C/C++ and Fortran code, and useful linear algebra, Fourier transform, and random number capabilities.
- **Keywords**: scientific computing, array, linear algebra, Fourier transform, random number generation, data processing, numerical analysis
- **Popularity**: 5

### 2. Search by Keywords

Search for packages using one or more keywords. Modulift supports both "or" and "and" relations.

Single Keyword Search

```python
result = mf.search_by_keywords("data science")
print(len(result))
```

Output:

```python
3141  # Number of packages related to "data science"
```

Multi-Keyword Search with "and" Relation

```python
result = mf.search_by_keywords("data science", "machine learning", "deep learning", relation="and")
print(len(result))
```

Output:

```python
115  # Number of packages related to all three keywords
```

Limiting Results with Markdown Output

```python
result = mf.search_by_keywords("data science", "machine learning", "deep learning", relation="and", limit=3, markdown=True)
```

Markdown Output

**1. sagemaker**

- **Description**: The SageMaker Python SDK is an open-source library designed for training and deploying machine learning models on Amazon SageMaker. It supports popular deep learning frameworks like Apache MXNet and TensorFlow, as well as Amazon's own scalable algorithms optimized for SageMaker. Users can also deploy custom algorithms in Docker containers, making it versatile for various machine learning tasks.
- **Keywords**: machine learning, deep learning, model deployment, Amazon SageMaker, data science, AI, Docker
- **Popularity**: 5

---

**2. fastbook**

- **Description**: fastbook is a comprehensive resource for learning deep learning through practical coding examples using the fastai library and PyTorch. It is designed for coders of all backgrounds, providing accessible education on deep learning techniques and applications across various domains, including computer vision and natural language processing.
- **Keywords**: deep learning, fastai, PyTorch, computer vision, natural language processing, machine learning, data science, Jupyter Notebooks
- **Popularity**: 5

---

**3. ProAiLab**

- **Description**: ProAiLab is a Python library designed for building and deploying AI models. It provides tools for data preprocessing, model training, and evaluation, making it suitable for both beginners and experienced machine learning practitioners.
- **Keywords**: machine learning, AI, deep learning, data science, model development
- **Popularity**: 4

### 3. Full-Text Search in Descriptions

Search for packages based on specific phrases within the description.

```python
result = mf.search_by_description("training and deploying machine learning models on Amazon", markdown=True)
```

Markdown Output

**1. sagemaker**

- **Description**: The SageMaker Python SDK is an open-source library designed for training and deploying machine learning models on Amazon SageMaker. It supports popular deep learning frameworks like Apache MXNet and TensorFlow, as well as Amazon's own scalable algorithms optimized for SageMaker. Users can also deploy custom algorithms in Docker containers, making it versatile for various machine learning tasks.
- **Keywords**: machine learning, deep learning, model deployment, Amazon SageMaker, data science, AI, Docker
- **Popularity**: 5

## License

Modulift is licensed under the MIT License. See [LICENSE](Licence.md) for more details.

## Contributing

We welcome contributions to Modulift! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

For any questions or feedback, please contact the author:

- **Okan Yenigün** - [okanyenigun@gmail.com](mailto:okanyenigun@gmail.com)
