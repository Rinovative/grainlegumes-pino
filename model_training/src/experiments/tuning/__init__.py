"""
===============================================================================
 experiments.tuning
===============================================================================
Public API for hyperparameter tuning infrastructure.

Responsibilities:
  - Export reusable Optuna study orchestration helpers
  - Export YAML search-space parsing and override helpers
  - Keep package-level imports limited to real public APIs

This package does NOT:
  - Define model-specific Python search spaces
  - Import model-specific study scripts
===============================================================================
"""

from .experiments_tuning_optuna import (
    OptunaStudyConfig,
    create_objective,
    describe_optuna_study_config,
    load_optuna_study_config,
    run_optuna_study,
    run_trial,
)
from .experiments_tuning_search_space import (
    SearchSpaceParameter,
    apply_trial_overrides,
    parse_search_space,
    search_space_summary,
    set_config_path,
    suggest_trial_overrides,
)

__all__ = [
    "OptunaStudyConfig",
    "SearchSpaceParameter",
    "apply_trial_overrides",
    "create_objective",
    "describe_optuna_study_config",
    "load_optuna_study_config",
    "parse_search_space",
    "run_optuna_study",
    "run_trial",
    "search_space_summary",
    "set_config_path",
    "suggest_trial_overrides",
]
