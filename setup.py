# setup.py
from setuptools import setup, find_packages

setup(
    name="agentic_maintenance",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "torch>=2.0.1",
        "scikit-learn>=1.3.0",
        "python-dateutil>=2.8.2",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "pyyaml>=6.0.1"
    ]
)