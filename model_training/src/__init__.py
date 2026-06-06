"""
Project source code organization.

Core modules: common, domain, datasets, learning, analysis, experiments.

Import subpackages explicitly when needed:
  from src import common
  from src import datasets
  from src import domain
  from src import learning
  from src import analysis
  from src import experiments

Lazy imports avoid circular dependencies with training module.
"""

__all__ = [
    "analysis",
    "common",
    "datasets",
    "domain",
    "experiments",
    "learning",
]
