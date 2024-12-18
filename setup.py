from setuptools import setup, find_packages

setup(
    name="",
    version="0.0.1",
    description="A package for feature extraction, hyperparameter optimization, and validation for item sales data.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost"
    ],
    python_requires=">=3.8",
)