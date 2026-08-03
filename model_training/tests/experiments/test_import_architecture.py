# ruff: noqa: S101, S603
"""Protect dependency-light facades, parser construction, and config semantics."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from support import configs


def _source_tree_roots() -> tuple[Path, Path]:
    """Return both maintained source roots used by cold import probes."""
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "model_training", project_root


def _run_probe(tmp_path: Path, code: str) -> None:
    """Run one cold source-tree import probe from an unrelated working directory."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in _source_tree_roots())
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_cold_source_tree_probe_exposes_both_maintained_roots(tmp_path: Path) -> None:
    """Import and cache representative public names from both source roots."""
    project_root = Path(__file__).resolve().parents[3]
    assert set(_source_tree_roots()) == {project_root / "model_training", project_root}
    _run_probe(
        tmp_path,
        f"""
from pathlib import Path
import data_generation
import src

project_root = Path({str(project_root)!r}).resolve()
assert Path(data_generation.__file__).resolve().is_relative_to(project_root / "data_generation")
assert Path(src.__file__).resolve().is_relative_to(project_root / "model_training")
generated_batch = data_generation.generated_batch
assert data_generation.generated_batch is generated_batch
assert data_generation.load_generated_batch is generated_batch.load_generated_batch
analysis = src.analysis
assert src.analysis is analysis
learning = src.learning
assert src.learning is learning
for package in (data_generation, src, analysis, learning):
    try:
        getattr(package, "unsupported")
    except AttributeError as error:
        assert str(error) == f"module {{package.__name__!r}} has no attribute 'unsupported'"
    else:
        raise AssertionError(f"{{package.__name__}} accepted an unknown attribute")
""",
    )


def test_public_facades_are_typed_dependency_light_and_cached(tmp_path: Path) -> None:
    """Expose stable public aliases without importing optional runtime backends."""
    _run_probe(
        tmp_path,
        """
import sys
from src import common, datasets, domain, learning

assert common.__all__ == ["locking", "paths", "serialization"]
assert datasets.__all__ == ["base", "identity", "metadata", "modules", "simulation"]
assert domain.__all__ == ["field_sets", "fields", "permeability", "physics", "tasks"]
assert learning.__all__ == ["device", "inference", "losses", "metrics", "models", "training"]
assert not {"torch", "neuralop", "wandb"}.intersection(sys.modules)

paths = common.paths
assert common.paths is paths
physics_contracts = domain.physics.contracts
assert domain.physics.contracts is physics_contracts
assert physics_contracts.validate_continuity_kind("div_eps_velocity") == "div_eps_velocity"
assert domain.tasks.registry.get_task("steady_flow").id == "steady_flow"
assert not {"torch", "neuralop", "wandb"}.intersection(sys.modules)
""",
    )


def test_nested_training_facade_loads_only_the_requested_service(tmp_path: Path) -> None:
    """Keep checkpoint, loop, and optimizer backends independent and cached."""
    _run_probe(
        tmp_path,
        """
import sys
from src import learning

training = learning.training
assert not {"torch", "neuralop", "wandb"}.intersection(sys.modules)
checkpoint = training.checkpoint
assert training.checkpoint is checkpoint
assert "src.learning.training.learning_training_loop" not in sys.modules
assert "src.learning.training.learning_training_optim" not in sys.modules
assert not {"neuralop", "wandb"}.intersection(sys.modules)
""",
    )
    _run_probe(
        tmp_path,
        """
import sys
from src import learning

optim = learning.training.optim
assert learning.training.optim is optim
assert "src.learning.training.learning_training_checkpoint" not in sys.modules
assert "src.learning.training.learning_training_loop" not in sys.modules
assert not {"torch", "neuralop", "wandb"}.intersection(sys.modules)
""",
    )
    _run_probe(
        tmp_path,
        """
import sys
from src import learning

loop = learning.training.loop
assert learning.training.loop is loop
assert "src.learning.training.learning_training_optim" not in sys.modules
assert not {"neuralop", "wandb"}.intersection(sys.modules)
""",
    )


def test_metadata_branch_does_not_initialize_model_or_tracking_backends(tmp_path: Path) -> None:
    """Allow metadata dependencies while excluding NeuralOp and W&B initialization."""
    _run_probe(
        tmp_path,
        """
import sys
from src import datasets

assert datasets.metadata.METADATA_SCHEMA_VERSION == 1
assert datasets.metadata is datasets.metadata
assert not {"torch", "neuralop", "wandb"}.intersection(sys.modules)
""",
    )


def test_notebook_context_preparation_remains_torch_free(tmp_path: Path) -> None:
    """Prepare the real metadata-only notebook context without tensor backends."""
    config_path = configs.experiment_config_path(model_kind="fno", physics_enabled=False)
    _run_probe(
        tmp_path,
        f"""
import sys
from pathlib import Path
from src import experiments

assert "torch" not in sys.modules
context = experiments.notebook_support.prepare_notebook_context(
    Path({str(config_path)!r}),
)
assert context.task.id == "steady_flow"
assert context.dataset_previews
assert not {{"torch", "neuralop", "wandb"}}.intersection(sys.modules)
""",
    )


def test_all_parser_constructors_remain_backend_free(tmp_path: Path) -> None:
    """Construct train, Optuna, and artifact parsers without runtime SDKs."""
    _run_probe(
        tmp_path,
        """
import sys
from src.experiments.cli import cli_build_artifacts, cli_config_preflight, cli_optuna, cli_train

for builder in (cli_train._build_parser, cli_optuna._build_parser, cli_build_artifacts._build_parser):
    parser = builder()
    assert "--device {auto,cuda,cpu}" in " ".join(parser.format_help().split())
preflight_parser = cli_config_preflight._build_parser()
assert "{train,optuna}" in " ".join(preflight_parser.format_help().split())
assert not {"torch", "neuralop", "wandb", "optuna", "pandas"}.intersection(sys.modules)
""",
    )


def test_config_resolution_does_not_initialize_neuralop_or_wandb(tmp_path: Path) -> None:
    """Resolve full semantics without model-construction or observer backends."""
    config_path = configs.experiment_config_path(model_kind="fno", physics_enabled=False)
    _run_probe(
        tmp_path,
        f"""
import sys
from pathlib import Path
from src.experiments.config import experiments_config_loader

config = experiments_config_loader.load_and_resolve_config(Path({str(config_path)!r}))
assert config["model"]["kind"] == "fno"
assert config["run"]["device"] == "auto"
assert "neuralop" not in sys.modules
assert "wandb" not in sys.modules
""",
    )


def test_inference_service_is_cached_without_observer_backends(tmp_path: Path) -> None:
    """Resolve the final inference service once without observer initialization."""
    _run_probe(
        tmp_path,
        """
import sys
from src import learning

inference = learning.inference
assert learning.inference is inference
assert inference.context.load_inference_context
assert "neuralop" not in sys.modules
assert "wandb" not in sys.modules
assert "optuna" not in sys.modules
""",
    )


def test_mixed_facades_expose_static_cores_and_cache_lazy_services(tmp_path: Path) -> None:
    """Keep lightweight cores static while preserving real runtime boundaries."""
    _run_probe(
        tmp_path,
        """
import sys
from src import analysis, experiments

artifacts = analysis.artifacts
assert artifacts.__all__ == ["contracts", "generation", "service", "timing"]
assert artifacts.contracts.__name__ == "src.analysis.artifacts.analysis_artifact_contracts"
assert "contracts" in artifacts.__dict__
assert "generation" not in artifacts.__dict__
assert "service" not in artifacts.__dict__
assert "timing" not in artifacts.__dict__
assert "torch" not in sys.modules
assert "pandas" not in sys.modules

presentation = analysis.presentation
assert presentation.__all__ == ["curated", "registry"]
assert presentation.registry.__name__ == "src.analysis.presentation.analysis_presentation_registry"
assert "registry" in presentation.__dict__
assert "curated" not in presentation.__dict__
assert "matplotlib" not in sys.modules

assert experiments.tuning.__all__ == ["optuna", "search_space"]
assert experiments.tuning.search_space.__name__ == "src.experiments.tuning.experiments_tuning_search_space"
assert "search_space" in experiments.tuning.__dict__
assert "optuna" not in experiments.tuning.__dict__
assert "torch" not in sys.modules
assert "optuna" not in sys.modules

for package in (artifacts, presentation, experiments.tuning):
    try:
        getattr(package, "unsupported")
    except AttributeError as error:
        assert "has no attribute 'unsupported'" in str(error)
    else:
        raise AssertionError("unknown lazy facade attribute did not fail")
""",
    )


def test_all_lazy_facades_cache_declared_names(tmp_path: Path) -> None:
    """Cache one real public name through every package-level resolver."""
    _run_probe(
        tmp_path,
        """
import data_generation
import src
from src import analysis, datasets, domain, experiments, learning

packages_and_names = (
    (data_generation, "generated_batch"),
    (data_generation, "load_generated_batch"),
    (src, "analysis"),
    (analysis, "eda"),
    (analysis.artifacts, "service"),
    (analysis.eda, "dataframe"),
    (analysis.evaluation, "case"),
    (analysis.presentation, "curated"),
    (datasets, "identity"),
    (domain.physics, "contracts"),
    (experiments, "notebook_support"),
    (experiments.tuning, "optuna"),
    (learning, "device"),
    (learning.losses, "factory"),
    (learning.training, "optim"),
)
for package, name in packages_and_names:
    first = getattr(package, name)
    assert package.__dict__[name] is first
    assert getattr(package, name) is first
""",
    )


def test_all_lazy_facades_reject_unknown_attributes_consistently(tmp_path: Path) -> None:
    """Raise clear errors from every package that implements lazy access."""
    _run_probe(
        tmp_path,
        """
import data_generation
import src
from src import analysis, datasets, domain, experiments, learning

packages = (
    data_generation,
    src,
    analysis,
    analysis.artifacts,
    analysis.eda,
    analysis.evaluation,
    analysis.presentation,
    datasets,
    domain.physics,
    experiments,
    experiments.tuning,
    learning,
    learning.losses,
    learning.training,
)
for package in packages:
    try:
        getattr(package, "unsupported")
    except AttributeError as error:
        assert str(error) == f"module {package.__name__!r} has no attribute 'unsupported'"
    else:
        raise AssertionError(f"{package.__name__} accepted an unknown attribute")
""",
    )


def test_analysis_import_branches_keep_optional_runtime_boundaries(tmp_path: Path) -> None:
    """Probe EDA, readers, and artifact service independently in fresh processes."""
    _run_probe(
        tmp_path,
        """
import sys
from src import analysis

for name in ("src.analysis.artifacts", "src.analysis.evaluation", "torch", "wandb"):
    assert name not in sys.modules, name
assert analysis.eda.dataframe.generate_eda_dataframe
for name in (
    "src.analysis.artifacts",
    "src.analysis.evaluation",
    "src.analysis.eda.eda_panel",
    "src.analysis.eda.plots",
    "matplotlib",
    "ipywidgets",
    "torch",
    "wandb",
):
    assert name not in sys.modules, name
""",
    )
    _run_probe(
        tmp_path,
        """
import sys
from src.analysis.artifacts import contracts
from src.analysis.evaluation import evaluation_dataframe

assert contracts.ARTIFACT_SCHEMA_VERSION == 1
assert evaluation_dataframe.build_eval_df
for name in (
    "src.analysis.artifacts.analysis_artifact_generation",
    "src.analysis.artifacts.analysis_artifact_service",
    "src.analysis.evaluation.evaluation_panel",
    "src.analysis.evaluation.evaluation_plot",
    "matplotlib",
    "ipywidgets",
    "torch",
    "wandb",
):
    assert name not in sys.modules, name
""",
    )
    _run_probe(
        tmp_path,
        """
import sys
from src.analysis.artifacts import service

assert service.build_artifacts
assert "src.learning" not in sys.modules
assert "wandb" not in sys.modules
""",
    )
