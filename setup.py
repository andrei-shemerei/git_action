from setuptools import setup, find_packages

setup(
    name="feature_sales_prediction_engine",
    version="0.0.2",
    description="A package for feature sales prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost"
    ],
    python_requires="=3.10.0",
)