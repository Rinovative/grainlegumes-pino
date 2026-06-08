"""
Reusable project packages for data, learning, analysis and experiments.

Provides:
- analysis: artifact generation, EDA, evaluation and UI modules
- common: shared path and utility modules
- datasets: dataset abstractions, simulation datasets and dataset modules
- domain: field contracts, permeability mappings and physics helpers
- experiments: CLI, config loading and tuning modules
- learning: models, losses, metrics, inference and training modules
"""

from . import analysis, common, datasets, domain, experiments, learning

__all__ = [
    "analysis",
    "common",
    "datasets",
    "domain",
    "experiments",
    "learning",
]
