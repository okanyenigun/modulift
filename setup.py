from setuptools import setup, find_packages


with open("Readme.md", "r") as f:
    long_description = f.read()


setup(
    name="modulift",
    version="0.0.3",
    description="A powerful search tool for discovering Python packages based on user queries",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/okanyenigun/modulift",
    author="Okan Yenigün",
    author_email="okanyenigun@gmail.com",
    license="MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Education',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Documentation',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent'
    ],
    install_requires=[
        'datasets==3.1.0',
        'scikit-learn==1.6.1',
        'ipywidgets==8.1.5',
        ],
    extras_require={
        'dev': ['twine==5.1.1'],
        'test': ['pytest==8.2.2']
    },
    python_requires='>=3.7',
)