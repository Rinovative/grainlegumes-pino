# ruff: noqa: S101
"""Validate dynamic executable identities and read-only config discovery."""

from __future__ import annotations

from src import common, experiments
from support import configs

_MIN_TIMESTAMP_DIGITS = 8


def _assert_canonical_name(resolved: dict) -> None:
    """Verify one resolved request against the authoritative semantic grammar."""
    run = resolved["run"]
    name = run["name"]
    model_variant = experiments.config.loader.resolved_model_variant(resolved)
    scientific_variant = experiments.config.loader.resolved_scientific_variant(resolved)
    expected = [model_variant]
    if scientific_variant is not None:
        expected.append(scientific_variant)
    seed_token = f"s{run['seed']}"
    expected.append(seed_token)
    if run.get("suffix") is not None:
        expected.append(run["suffix"])

    parts = name.split("__")
    assert parts == expected
    assert all(parts)
    assert name == "__".join(parts)
    assert parts.count(model_variant) == 1
    assert parts.count(seed_token) == 1
    assert resolved["task"] not in name
    assert "___" not in name
    assert not any(part.isdigit() and len(part) >= _MIN_TIMESTAMP_DIGITS for part in parts)
    if run.get("suffix") is not None:
        assert parts[-1] == run["suffix"]
    local_dir = common.paths.resolve_run_output_dir(
        resolved["task"],
        name,
        output_root=resolved["paths"]["output_root"],
    )
    assert local_dir.name == name


def test_executable_discovery_is_unique_read_only_and_uses_only_the_current_hierarchy() -> None:
    """Prove dynamic identities and resolution without freezing or mutating directory membership."""
    before = {path.relative_to(configs.TASKS_CONFIG_ROOT): path.read_bytes() for path in configs.executable_config_paths()}
    names_by_scope: dict[tuple[str, str], set[str]] = {}

    resolved_requests = [
        experiments.config.loader.load_and_resolve_config(path) for path in (*configs.experiment_config_paths(), *configs.acceptance_config_paths())
    ]
    studies = [experiments.tuning.optuna.load_optuna_study_config(path) for path in configs.optuna_config_paths()]
    resolved_requests.extend(study.base_config for study in studies)

    for resolved in resolved_requests:
        _assert_canonical_name(resolved)
        scope = (resolved["task"], resolved["tracking"]["wandb"]["workflow"])
        scoped_names = names_by_scope.setdefault(scope, set())
        assert resolved["run"]["name"] not in scoped_names
        scoped_names.add(resolved["run"]["name"])

    study_names = [study.study["name"] for study in studies]
    assert len(study_names) == len(set(study_names))
    after = {path.relative_to(configs.TASKS_CONFIG_ROOT): path.read_bytes() for path in configs.executable_config_paths()}
    assert after == before
