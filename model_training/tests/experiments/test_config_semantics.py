# ruff: noqa: S101
"""Verify strict semantic experiment configuration contracts."""

import copy
from pathlib import Path

import pytest
from src import domain, experiments

_CONFIG_ROOT = Path(__file__).parents[2] / "configs"
_EXPERIMENTS = sorted((_CONFIG_ROOT / "experiments").glob("*.yaml"))


def _raw_fno() -> dict[str, object]:
    return experiments.config.loader.load_yaml(_CONFIG_ROOT / "experiments/steady_flow_fno.yaml")


@pytest.mark.parametrize("path", _EXPERIMENTS, ids=lambda path: path.stem)
def test_every_experiment_yaml_resolves(path: Path) -> None:
    """All maintained experiments resolve to one exact task contract."""
    config = experiments.config.loader.load_and_resolve_config(path)
    task = domain.tasks.registry.get_task("steady_flow")
    assert config["task_contract"]["digest"] == task.contract_digest
    assert config["model"]["params"]["in_channels"] == len(task.input_names)
    assert config["model"]["params"]["out_channels"] == len(task.output_names)
    expected_objective = {
        "id": "normalized_relative_h1",
        "kind": "relative_h1",
        "space": "normalized",
        "fields": ["p", "u", "v"],
        "reduction": "sample_mean",
        "direction": "minimize",
    }
    assert config["evaluation"]["objective"] == expected_objective
    selected = next(metric for metric in config["evaluation"]["metrics"] if metric["id"] == expected_objective["id"])
    assert selected == expected_objective


def test_descriptive_pi_yaml_names_use_loss_composition() -> None:
    """PI filenames do not create class-style model identifiers."""
    for filename, expected_kind in (
        ("steady_flow_pifno.yaml", "fno"),
        ("steady_flow_piuno.yaml", "uno"),
    ):
        config = experiments.config.loader.load_and_resolve_config(_CONFIG_ROOT / "experiments" / filename)
        assert config["model"]["kind"] == expected_kind
        assert config["loss"]["physics"]["enabled"] is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda cfg: cfg.update({"task": "unknown_task"}), "Unknown task"),
        (lambda cfg: cfg["model"].update({"kind": "PI-FNO"}), "Unknown model identifier"),
        (lambda cfg: cfg["loss"]["data"].update({"kind": "H1Loss"}), "not allowed by task"),
        (lambda cfg: cfg["evaluation"]["metrics"][0].update({"kind": "RMSEOverall"}), "Unknown metric identifier"),
        (lambda cfg: cfg["evaluation"]["metrics"][0].update({"fields": ["temperature"]}), "unknown task output"),
        (lambda cfg: cfg["evaluation"]["objective"].update({"id": "unknown_metric"}), "is not a declared metric id"),
        (lambda cfg: cfg["evaluation"]["objective"].update({"direction": "maximize"}), "unknown key"),
        (lambda cfg: cfg["model"]["params"].update({"in_channels": 8}), "task-fixed channel"),
        (lambda cfg: cfg.setdefault("data", {}).update({"ood_datasets": ["first", "second"]}), "exactly one"),
        (lambda cfg: cfg.setdefault("data", {}).update({"train_dataset": "../escape"}), "single non-empty path component"),
        (lambda cfg: cfg["run"].update({"prefix": "../escape"}), "single non-empty path component"),
    ],
)
def test_invalid_semantic_identifiers_and_overrides_fail(mutation: object, match: str) -> None:
    """Strict path-rich resolution rejects noncanonical IDs and task overrides."""
    config = copy.deepcopy(_raw_fno())
    mutation(config)  # type: ignore[operator]
    with pytest.raises(ValueError, match=match):
        experiments.config.loader.resolve_config(config)


def test_unsupported_uno_depth_fails_before_fresh_run_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semantic resolution rejects invalid UNO topology before output allocation."""
    config = experiments.config.loader.load_yaml(_CONFIG_ROOT / "experiments/steady_flow_piuno.yaml")
    model = config["model"]
    assert isinstance(model, dict)
    params = model["params"]
    assert isinstance(params, dict)
    params["n_layers"] = 3
    allocation_attempted = False

    def fail_allocation(_run_dir: Path | str) -> Path:
        nonlocal allocation_attempted
        allocation_attempted = True
        pytest.fail("run allocation must not be attempted")

    monkeypatch.setattr(experiments.run.config_loader, "load_yaml", lambda _path: config)
    monkeypatch.setattr(experiments.run, "allocate_run_directory", fail_allocation)
    output_root = tmp_path / "outputs"

    with pytest.raises(ValueError, match="supports exactly 5 or 7 layers"):
        experiments.run.run_experiment("invalid-uno.yaml", output_root=output_root)

    assert allocation_attempted is False
    assert not output_root.exists()


def test_duplicate_metric_id_fails() -> None:
    """Metric ids are unique within an evaluation contract."""
    config = copy.deepcopy(_raw_fno())
    evaluation = config["evaluation"]
    assert isinstance(evaluation, dict)
    metrics = evaluation["metrics"]
    assert isinstance(metrics, list)
    first, second = metrics[:2]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    second["id"] = first["id"]
    with pytest.raises(ValueError, match="Duplicate evaluation metric id"):
        experiments.config.loader.resolve_config(config)
