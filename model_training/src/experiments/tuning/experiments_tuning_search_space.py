"""
===============================================================================
experiments_tuning_search_space.py
===============================================================================
Parse YAML-defined Optuna search spaces and apply trial overrides.

Responsibilities:
  - Parse exact categorical, float, integer, and fixed parameter schemas
  - Validate approved dotted paths, supported values, and base-value containment
  - Request typed Optuna suggestions and apply them to isolated config copies

Design principles:
  - Search dimensions are explicit, finite where required, and independently auditable
  - Trial overrides cannot mutate task, objective, path, seed, or derived channel identity
  - Parsing and config application remain independent of model implementations

This module does NOT:
  - Create studies or execute trials; ``experiments.tuning.optuna`` owns lifecycle
  - Construct models or optimizers; learning factories own runtime objects
===============================================================================
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

SearchKind = Literal["categorical", "float", "int", "fixed"]

_COMMON_SEARCH_PATH_KINDS: dict[str, frozenset[SearchKind]] = {
    "data.batch_size": frozenset({"categorical", "int"}),
    "optimizer.lr": frozenset({"categorical", "float"}),
    "optimizer.weight_decay": frozenset({"categorical", "float"}),
}
_MODEL_SEARCH_PATH_KINDS: dict[str, dict[str, frozenset[SearchKind]]] = {
    "fno": {
        "model.params.n_modes.0": frozenset({"categorical"}),
        "model.params.n_modes.1": frozenset({"categorical"}),
        "model.params.hidden_channels": frozenset({"categorical"}),
        "model.params.n_layers": frozenset({"categorical"}),
    },
    "uno": {
        "model.params.modes_x": frozenset({"categorical"}),
        "model.params.modes_y": frozenset({"categorical"}),
        "model.params.hidden_channels": frozenset({"categorical"}),
        "model.params.n_layers": frozenset({"categorical"}),
        "model.params.mode_ratio": frozenset({"categorical", "float"}),
    },
}
_PHYSICS_SEARCH_PATH_KINDS: dict[str, frozenset[SearchKind]] = {
    "loss.physics.residual_weight.target": frozenset({"categorical", "float"}),
    "loss.physics.residual_weight.warmup.epochs": frozenset({"categorical", "int"}),
    "loss.physics.boundary_weight.target": frozenset({"categorical", "float"}),
    "loss.physics.boundary_weight.warmup.epochs": frozenset({"categorical", "int"}),
    "loss.physics.continuity": frozenset({"categorical"}),
}


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
        Fixed value, used for fixed parameters.

    Notes
    -----
    Instances are immutable parsed transport records. Path admission against a
    resolved model/task contract occurs separately in
    :func:`validate_search_space_paths`.

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


def _require_nonempty_string(value: Any, *, label: str) -> str:
    """Return a non-empty string without coercing another scalar type."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{label} must be a non-empty string, got: {value!r}"
        raise TypeError(msg)
    return value


def _parse_kind(value: Any, *, path: str) -> SearchKind:
    """Parse one exact YAML search parameter kind."""
    kind = _require_nonempty_string(value, label=f"search_space.{path}.kind")
    valid_kinds = {"categorical", "float", "int", "fixed"}
    if kind not in valid_kinds:
        msg = f"search_space.{path}.kind must be one of {sorted(valid_kinds)}, got: {kind!r}"
        raise ValueError(msg)
    return cast("SearchKind", kind)


def _require_finite_number(value: Any, *, label: str) -> int | float:
    """Return one finite YAML number while rejecting booleans and strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{label} must be a finite number, got: {value!r}"
        raise TypeError(msg)
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"{label} must be finite, got: {value!r}"
        raise ValueError(msg)
    return value


def _require_exact_int(value: Any, *, label: str) -> int:
    """Return one exact integer while rejecting booleans and integral floats."""
    if type(value) is not int:
        msg = f"{label} must be an integer, got: {value!r}"
        raise TypeError(msg)
    return value


def _parse_numeric_spec(
    spec: Mapping[str, Any],
    *,
    path: str,
    kind: Literal["float", "int"],
) -> tuple[int | float, int | float, int | float | None, bool]:
    """
    Parse one exact numeric parameter domain without scalar coercion.

    Required/optional keys depend on integer versus float kind. Bounds must be
    finite and ordered, log sampling requires positive bounds and forbids a
    step, and any step must divide the closed bound span exactly.
    """
    required_keys = {"name", "kind", "low", "high"}
    optional_keys = {"step", "log"}
    missing_keys = sorted(required_keys.difference(spec))
    if missing_keys:
        msg = f"search_space.{path} is missing required key(s): {missing_keys}."
        raise KeyError(msg)
    unknown_keys = sorted(set(spec).difference(required_keys | optional_keys))
    if unknown_keys:
        msg = f"search_space.{path} contains key(s) invalid for {kind}: {unknown_keys}."
        raise ValueError(msg)

    low: int | float
    high: int | float
    if kind == "int":
        low = _require_exact_int(spec["low"], label=f"search_space.{path}.low")
        high = _require_exact_int(spec["high"], label=f"search_space.{path}.high")
    else:
        low = _require_finite_number(spec["low"], label=f"search_space.{path}.low")
        high = _require_finite_number(spec["high"], label=f"search_space.{path}.high")

    if low >= high:
        msg = f"search_space.{path} requires low < high, got low={low!r}, high={high!r}"
        raise ValueError(msg)

    log = spec.get("log", False)
    if type(log) is not bool:
        msg = f"search_space.{path}.log must be a boolean, got: {log!r}"
        raise TypeError(msg)
    if log and (low <= 0 or high <= 0):
        msg = f"search_space.{path} log sampling requires positive bounds"
        raise ValueError(msg)

    step: int | float | None = None
    if "step" in spec:
        if kind == "int":
            step = _require_exact_int(spec["step"], label=f"search_space.{path}.step")
        else:
            step = _require_finite_number(spec["step"], label=f"search_space.{path}.step")
        if step <= 0:
            msg = f"search_space.{path}.step must be positive, got: {step!r}"
            raise ValueError(msg)
    if log and step is not None:
        msg = f"search_space.{path} cannot combine log sampling with step"
        raise ValueError(msg)
    if step is not None:
        if kind == "int":
            divisible = (int(high) - int(low)) % int(step) == 0
        else:
            step_count = (float(high) - float(low)) / float(step)
            divisible = math.isclose(step_count, round(step_count), rel_tol=1e-9, abs_tol=1e-9)
        if not divisible:
            msg = f"search_space.{path}.step must divide the bound span exactly"
            raise ValueError(msg)
    return low, high, step, log


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
        If a domain, choice, bound, step, or parameter name is invalid.
    KeyError
        If a kind-specific required key is absent.

    """
    mapping = _require_mapping(raw_search_space, label="search_space")
    if not mapping:
        msg = "search_space must contain at least one parameter"
        raise ValueError(msg)

    parameters: list[SearchSpaceParameter] = []
    for raw_path, raw_spec in mapping.items():
        path = _require_nonempty_string(raw_path, label="search_space parameter path")
        spec = _require_mapping(raw_spec, label=f"search_space.{path}")
        if "kind" not in spec:
            msg = f"search_space.{path}.kind is required"
            raise KeyError(msg)
        if "name" not in spec:
            msg = f"search_space.{path}.name is required"
            raise KeyError(msg)
        kind = _parse_kind(spec["kind"], path=path)
        name = _require_nonempty_string(spec["name"], label=f"search_space.{path}.name")

        if kind == "categorical":
            expected_keys = {"name", "kind", "values"}
            unknown_keys = sorted(set(spec).difference(expected_keys))
            missing_keys = sorted(expected_keys.difference(spec))
            if unknown_keys:
                msg = f"search_space.{path} contains key(s) invalid for categorical: {unknown_keys}."
                raise ValueError(msg)
            if missing_keys:
                msg = f"search_space.{path} is missing required key(s): {missing_keys}."
                raise KeyError(msg)
            values = tuple(
                copy.deepcopy(value)
                for value in _require_sequence(
                    spec["values"],
                    label=f"search_space.{path}.values",
                )
            )
            for value in values:
                if value is not None and type(value) not in {bool, int, float, str}:
                    msg = f"search_space.{path}.values must contain only supported scalar choices"
                    raise TypeError(msg)
                if isinstance(value, float) and not math.isfinite(value):
                    msg = f"search_space.{path}.values must contain only finite choices"
                    raise ValueError(msg)
            if len(set(values)) != len(values):
                msg = f"search_space.{path}.values must be unique"
                raise ValueError(msg)
            parameters.append(SearchSpaceParameter(path=path, name=name, kind=kind, values=values))
            continue

        if kind == "fixed":
            expected_keys = {"name", "kind", "value"}
            unknown_keys = sorted(set(spec).difference(expected_keys))
            missing_keys = sorted(expected_keys.difference(spec))
            if unknown_keys:
                msg = f"search_space.{path} contains key(s) invalid for fixed: {unknown_keys}."
                raise ValueError(msg)
            if missing_keys:
                msg = f"search_space.{path} is missing required key(s): {missing_keys}."
                raise KeyError(msg)
            parameters.append(
                SearchSpaceParameter(
                    path=path,
                    name=name,
                    kind=kind,
                    value=copy.deepcopy(spec["value"]),
                )
            )
            continue

        numeric_kind = cast('Literal["float", "int"]', kind)
        low, high, step, log = _parse_numeric_spec(spec, path=path, kind=numeric_kind)
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

    names = [parameter.name for parameter in parameters]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        msg = f"search_space contains duplicate parameter name(s): {duplicate_names}."
        raise ValueError(msg)
    return tuple(parameters)


def _resolved_path_value(config: dict[str, Any], path: str) -> Any:
    """Return the current value at one validated dotted config path."""
    current: Any = config
    for token in path.split("."):
        current = _descend(current, token, full_path=path)
    return current


def _parameter_contains_value(parameter: SearchSpaceParameter, value: Any) -> bool:
    """
    Test whether a parsed domain contains the resolved base value exactly.

    Categorical and fixed domains use equality; numeric domains require type,
    inclusive bounds, and step-grid alignment. This admission rule guarantees
    the embedded base experiment is itself a member of the declared study.
    """
    if parameter.kind == "categorical":
        return value in parameter.values
    if parameter.kind == "fixed":
        return value == parameter.value
    if parameter.low is None or parameter.high is None:
        return False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric_value = float(value)
    if not float(parameter.low) <= numeric_value <= float(parameter.high):
        return False
    if parameter.step is None:
        return True
    offset = (numeric_value - float(parameter.low)) / float(parameter.step)
    return math.isclose(offset, round(offset), rel_tol=1e-9, abs_tol=1e-9)


def validate_search_space_paths(
    config: dict[str, Any],
    search_space: Sequence[SearchSpaceParameter],
) -> None:
    """
    Validate approved search paths, parameter kinds, and base-value containment.

    Parameters
    ----------
    config : dict[str, Any]
        Fully resolved experiment base config.
    search_space : Sequence[SearchSpaceParameter]
        Parsed semantic override definitions.

    Raises
    ------
    ValueError
        If a path is unapproved for the model/physics mode, its parameter kind is
        unsupported, structural UNO depth is invalid, or the resolved base value
        lies outside the declared domain.

    Notes
    -----
    Requiring the base point makes every study configuration internally coherent
    and prevents a search recipe from describing a different model than it embeds.

    """
    model = _require_mapping(config.get("model"), label="config.model")
    model_kind = _require_nonempty_string(model.get("kind"), label="config.model.kind")
    model_paths = _MODEL_SEARCH_PATH_KINDS.get(model_kind)
    if model_paths is None:
        msg = f"No maintained Optuna search policy exists for model kind {model_kind!r}."
        raise ValueError(msg)

    allowed = {**_COMMON_SEARCH_PATH_KINDS, **model_paths}
    loss = _require_mapping(config.get("loss"), label="config.loss")
    physics = _require_mapping(loss.get("physics"), label="config.loss.physics")
    if physics.get("enabled") is True:
        allowed.update(_PHYSICS_SEARCH_PATH_KINDS)

    for parameter in search_space:
        tokens = parameter.path.split(".")
        if not tokens or any(token == "" for token in tokens):
            msg = f"Invalid override path: {parameter.path!r}"
            raise ValueError(msg)
        allowed_kinds = allowed.get(parameter.path)
        if allowed_kinds is None:
            msg = f"Search-space path {parameter.path!r} is not approved for model={model_kind!r}, physics_enabled={physics.get('enabled') is True}."
            raise ValueError(msg)
        if parameter.kind not in allowed_kinds:
            msg = f"Search-space path {parameter.path!r} does not support kind {parameter.kind!r}; allowed kinds are {sorted(allowed_kinds)}."
            raise ValueError(msg)
        if model_kind == "uno" and parameter.path == "model.params.n_layers" and parameter.kind == "categorical":
            unsupported_depths = sorted(set(parameter.values).difference({5, 7}))
            if unsupported_depths:
                msg = f"UNO search-space depths must be structurally supported values 5 or 7, got {unsupported_depths}."
                raise ValueError(msg)
        base_value = _resolved_path_value(config, parameter.path)
        if not _parameter_contains_value(parameter, base_value):
            msg = f"Search-space parameter {parameter.path!r} must contain its resolved base value {base_value!r}."
            raise ValueError(msg)

    continuity = next(
        (parameter for parameter in search_space if parameter.path == "loss.physics.continuity"),
        None,
    )
    if continuity is not None:
        task_contract = _require_mapping(config.get("task_contract"), label="config.task_contract")
        physics_contract = _require_mapping(task_contract.get("physics"), label="config.task_contract.physics")
        allowed_continuities = tuple(physics_contract.get("allowed_continuities", ()))
        unsupported = sorted(set(continuity.values).difference(allowed_continuities))
        if unsupported:
            msg = f"Search-space continuity choices are unsupported by the task contract: {unsupported}."
            raise ValueError(msg)


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
    """
    Descend one validated dotted-path token without creating structure.

    Mappings require an existing key and lists require an in-range decimal
    index. Error messages retain the complete original path for YAML diagnostics.
    """
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
    """
    Assign one terminal path token without creating keys or extending lists.

    The caller owns value copying. Missing keys, non-numeric list tokens,
    out-of-range indices, and scalar descent fail with full-path context.
    """
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
        Value deep-copied into the existing terminal node.

    Raises
    ------
    ValueError
        If ``path`` is empty or contains an empty token.
    KeyError
        If a mapping token does not already exist.
    TypeError
        If a token cannot descend into or index the current container.
    IndexError
        If a list token is outside the existing list.

    Notes
    -----
    This function mutates ``config`` but never creates schema structure; callers
    must revalidate the resulting resolved config.

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
            "kind": parameter.kind,
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
