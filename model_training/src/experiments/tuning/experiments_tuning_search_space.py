"""
===============================================================================
 experiments_tuning_search_space.py
===============================================================================
Parse and apply YAML-defined Optuna search spaces.

Responsibilities:
  - Validate search_space blocks from Optuna YAML files
  - Convert YAML parameter specs into Optuna trial suggestions
  - Apply sampled values to resolved experiment configs by dotted paths
  - Support dictionary keys and list indices in override paths

Design principles:
  - YAML is the normal interface for hyperparameter search spaces
  - Search-space parsing has no dependency on model-specific scripts
  - Trial overrides are explicit and auditable
  - Invalid paths fail fast instead of silently creating new config keys

This module does NOT:
  - Create or run Optuna studies
  - Build models, losses, datasets, or optimizers
  - Define model-specific Python search spaces
===============================================================================
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

SearchKind = Literal["categorical", "float", "int", "fixed"]


class TrialLike(Protocol):
    """Minimal Optuna trial interface used by the search-space parser."""

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        """Suggest one value from categorical choices."""
        ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float:
        """Suggest a floating-point value."""
        ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
        step: int = 1,
    ) -> int:
        """Suggest an integer value."""
        ...


@dataclass(frozen=True)
class SearchSpaceParameter:
    """
    One YAML-defined Optuna search parameter.

    Parameters
    ----------
    path : str
        Dotted path in the resolved experiment config to override
    name : str
        Optuna parameter name stored in the study
    kind : SearchKind
        Suggestion type: categorical, float, int, or fixed
    values : tuple[Any, ...]
        Categorical choices, used for categorical parameters
    low : int | float | None
        Lower bound for numeric suggestions
    high : int | float | None
        Upper bound for numeric suggestions
    step : int | float | None
        Optional numeric step
    log : bool
        Whether numeric sampling should use log scale
    value : Any
        Fixed value, used for fixed parameters

    """

    path: str
    name: str
    kind: SearchKind
    values: tuple[Any, ...] = ()
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    log: bool = False
    value: Any = None


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Validate and return a mapping value."""
    if not isinstance(value, Mapping):
        msg = f"{label} must be a mapping, got: {type(value).__name__}"
        raise TypeError(msg)
    return cast("Mapping[str, Any]", value)


def _require_sequence(value: Any, *, label: str) -> Sequence[Any]:
    """Validate and return a non-string sequence value."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{label} must be a sequence, got: {value!r}"
        raise TypeError(msg)
    if not value:
        msg = f"{label} must not be empty"
        raise ValueError(msg)
    return value


def _parse_kind(value: Any, *, path: str) -> SearchKind:
    """Parse a YAML search parameter kind."""
    kind = str(value).lower()
    valid_kinds = {"categorical", "float", "int", "fixed"}
    if kind not in valid_kinds:
        msg = f"search_space.{path}.type must be one of {sorted(valid_kinds)}, got: {kind!r}"
        raise ValueError(msg)
    return cast("SearchKind", kind)


def _parse_numeric_bounds(spec: Mapping[str, Any], *, path: str) -> tuple[int | float, int | float]:
    """Parse low/high bounds from a numeric search spec."""
    if "low" not in spec or "high" not in spec:
        msg = f"search_space.{path} numeric specs require low and high"
        raise KeyError(msg)
    low = cast("int | float", spec["low"])
    high = cast("int | float", spec["high"])
    return low, high


def parse_search_space(raw_search_space: Any) -> tuple[SearchSpaceParameter, ...]:
    """
    Parse a YAML search_space block.

    Parameters
    ----------
    raw_search_space : Any
        Raw search_space value loaded from YAML

    Returns
    -------
    tuple[SearchSpaceParameter, ...]
        Validated search-space parameters

    Raises
    ------
    TypeError
        If the search_space block or a spec has the wrong type
    ValueError
        If a spec is incomplete or invalid

    """
    mapping = _require_mapping(raw_search_space, label="search_space")
    if not mapping:
        msg = "search_space must contain at least one parameter"
        raise ValueError(msg)

    parameters: list[SearchSpaceParameter] = []
    for path, raw_spec in mapping.items():
        spec = _require_mapping(raw_spec, label=f"search_space.{path}")
        kind = _parse_kind(spec.get("type", "categorical"), path=path)
        name = str(spec.get("name", path))

        if kind == "categorical":
            values = tuple(copy.deepcopy(value) for value in _require_sequence(spec.get("values"), label=f"search_space.{path}.values"))
            parameters.append(SearchSpaceParameter(path=path, name=name, kind=kind, values=values))
            continue

        if kind == "fixed":
            if "value" not in spec:
                msg = f"search_space.{path}.value is required for fixed parameters"
                raise KeyError(msg)
            parameters.append(SearchSpaceParameter(path=path, name=name, kind=kind, value=copy.deepcopy(spec["value"])))
            continue

        low, high = _parse_numeric_bounds(spec, path=path)
        step = cast("int | float | None", spec.get("step"))
        log = bool(spec.get("log", False))
        if log and step is not None:
            msg = f"search_space.{path} cannot combine log sampling with step"
            raise ValueError(msg)
        parameters.append(
            SearchSpaceParameter(
                path=path,
                name=name,
                kind=kind,
                low=low,
                high=high,
                step=step,
                log=log,
            )
        )

    return tuple(parameters)


def suggest_trial_overrides(trial: TrialLike, search_space: Sequence[SearchSpaceParameter]) -> dict[str, Any]:
    """
    Sample a trial and return config-path overrides.

    Parameters
    ----------
    trial : TrialLike
        Optuna trial or compatible object
    search_space : Sequence[SearchSpaceParameter]
        Parsed search-space parameters

    Returns
    -------
    dict[str, Any]
        Mapping from config dotted paths to sampled override values

    """
    overrides: dict[str, Any] = {}
    for parameter in search_space:
        if parameter.kind == "categorical":
            overrides[parameter.path] = trial.suggest_categorical(parameter.name, list(parameter.values))
        elif parameter.kind == "float":
            if parameter.low is None or parameter.high is None:
                msg = f"Float parameter {parameter.path!r} is missing bounds"
                raise ValueError(msg)
            overrides[parameter.path] = trial.suggest_float(
                parameter.name,
                float(parameter.low),
                float(parameter.high),
                log=parameter.log,
                step=float(parameter.step) if parameter.step is not None else None,
            )
        elif parameter.kind == "int":
            if parameter.low is None or parameter.high is None:
                msg = f"Int parameter {parameter.path!r} is missing bounds"
                raise ValueError(msg)
            overrides[parameter.path] = trial.suggest_int(
                parameter.name,
                int(parameter.low),
                int(parameter.high),
                log=parameter.log,
                step=int(parameter.step) if parameter.step is not None else 1,
            )
        elif parameter.kind == "fixed":
            overrides[parameter.path] = copy.deepcopy(parameter.value)
        else:
            msg = f"Unsupported search parameter kind: {parameter.kind!r}"
            raise ValueError(msg)
    return overrides


def _descend(container: Any, token: str, *, full_path: str) -> Any:
    """Descend one token into a dict or list config container."""
    if isinstance(container, MutableMapping):
        if token not in container:
            msg = f"Override path {full_path!r} does not exist at key {token!r}"
            raise KeyError(msg)
        return container[token]
    if isinstance(container, list):
        if not token.isdigit():
            msg = f"Override path {full_path!r} expected a list index, got {token!r}"
            raise TypeError(msg)
        index = int(token)
        if index >= len(container):
            msg = f"Override path {full_path!r} list index {index} is out of range"
            raise IndexError(msg)
        return container[index]
    msg = f"Override path {full_path!r} cannot descend into {type(container).__name__}"
    raise TypeError(msg)


def _assign(container: Any, token: str, value: Any, *, full_path: str) -> None:
    """Assign one value into a dict or list config container."""
    if isinstance(container, MutableMapping):
        if token not in container:
            msg = f"Override path {full_path!r} does not exist at key {token!r}"
            raise KeyError(msg)
        container[token] = value
        return
    if isinstance(container, list):
        if not token.isdigit():
            msg = f"Override path {full_path!r} expected a list index, got {token!r}"
            raise TypeError(msg)
        index = int(token)
        if index >= len(container):
            msg = f"Override path {full_path!r} list index {index} is out of range"
            raise IndexError(msg)
        container[index] = value
        return
    msg = f"Override path {full_path!r} cannot assign into {type(container).__name__}"
    raise TypeError(msg)


def set_config_path(config: dict[str, Any], path: str, value: Any) -> None:
    """
    Set a value in a resolved config by dotted path.

    Parameters
    ----------
    config : dict[str, Any]
        Resolved experiment config to mutate
    path : str
        Dotted path. Numeric tokens index into lists, e.g. model.params.n_modes.0
    value : Any
        Value to assign

    """
    tokens = path.split(".")
    if not tokens or any(token == "" for token in tokens):
        msg = f"Invalid override path: {path!r}"
        raise ValueError(msg)

    current: Any = config
    for token in tokens[:-1]:
        current = _descend(current, token, full_path=path)
    _assign(current, tokens[-1], copy.deepcopy(value), full_path=path)


def apply_trial_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return a config copy with trial overrides applied.

    Parameters
    ----------
    config : dict[str, Any]
        Resolved base experiment config
    overrides : Mapping[str, Any]
        Mapping from dotted config paths to override values

    Returns
    -------
    dict[str, Any]
        New resolved config copy with overrides applied

    """
    trial_config = copy.deepcopy(config)
    for path, value in overrides.items():
        set_config_path(trial_config, path, value)
    return trial_config


def search_space_summary(search_space: Sequence[SearchSpaceParameter]) -> list[dict[str, Any]]:
    """
    Build a serializable summary of parsed search parameters.

    Parameters
    ----------
    search_space : Sequence[SearchSpaceParameter]
        Parsed search-space parameters

    Returns
    -------
    list[dict[str, Any]]
        Human-readable search-space summary

    """
    summary: list[dict[str, Any]] = []
    for parameter in search_space:
        item: dict[str, Any] = {
            "path": parameter.path,
            "name": parameter.name,
            "type": parameter.kind,
        }
        if parameter.kind == "categorical":
            item["values"] = list(parameter.values)
        elif parameter.kind in ("float", "int"):
            item.update({"low": parameter.low, "high": parameter.high, "log": parameter.log})
            if parameter.step is not None:
                item["step"] = parameter.step
        elif parameter.kind == "fixed":
            item["value"] = parameter.value
        summary.append(item)
    return summary
