"""
Own explicit multi-root package discovery for the maintained distribution.

The project intentionally keeps the generated-data bridge beside the generation
workflow while the model-training package remains below model_training.
"""

from setuptools import find_packages, setup

packages = [
    *find_packages(where="model_training", include=("src", "src.*")),
    *find_packages(where=".", include=("data_generation", "data_generation.*")),
]

setup(
    packages=packages,
    package_dir={"": "model_training", "data_generation": "data_generation"},
)
