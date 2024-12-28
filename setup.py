from setuptools import setup, find_packages

setup(
    name="feature_sales_prediction_engine",
    version="0.0.6",
    description="A package for feature sales prediction",
    long_description_content_type="text/markdown",
    packages=["feature_sales_prediction_engine"],
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "xgboost>=1.4.0"
    ],
    python_requires=">=3.10,<4",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/yourusername/feature_sales_prediction_engine",
)