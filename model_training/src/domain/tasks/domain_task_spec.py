"""
===============================================================================
domain_task_spec.py
===============================================================================
Define immutable semantic contracts for learned operator tasks.

Responsibilities:
  - Describe fields, datasets, preprocessing, metrics, and physics selection
  - Validate task structure, exact field membership, and tensor axes
  - Produce deterministic resolved contracts and contract digests

Design principles:
  - Task-fixed semantics have one immutable source of truth
  - Physical units remain distinct from stored/model representations
  - Channel counts and operator dimensionality are derived values

Boundaries:
  - Task registration belongs to domain.tasks.registry
  - Concrete task declarations belong to task-specific modules
  - Physics equations and numerical implementations belong outside this module
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

FieldRole = Literal[
    "coordinate",
    "permeability",
    "porosity",
    "boundary",
    "state",
]
MetricSpace = Literal["normalized", "physical"]
MetricReduction = Literal["sample_mean", "element_mean"]
OptimizationDirection = Literal["minimize", "maximize"]
_SUPPORTED_TENSOR_LAYOUT = ("batch", "channel", "y", "x")
_SUPPORTED_OPERATOR_AXES = (2, 3)
_SUPPORTED_NORMALIZATION_AXES = (0, 2, 3)
_FIELD_ROLES = frozenset({"coordinate", "permeability", "porosity", "boundary", "state"})
_METRIC_SPACES = frozenset({"normalized", "physical"})
_METRIC_REDUCTIONS = frozenset({"sample_mean", "element_mean"})
_OPTIMIZATION_DIRECTIONS = frozenset({"minimize", "maximize"})


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """
    Describe one named tensor field.

    Attributes
    ----------
    name : str
        Canonical machine-readable field name.
    role : FieldRole
        Semantic role within the task.
    unit : str
        Physical unit before train-fitted normalization.
    representation : str
        Stored/model representation distinct from the physical unit.
    source_name : str | None
        Exact producer column name when it differs from the canonical field.

    """

    name: str
    role: FieldRole
    unit: str
    representation: str
    source_name: str | None = None

    def __post_init__(self) -> None:
        """Reject empty names/units/representations and unsupported runtime roles."""
        for label, value in (("name", self.name), ("unit", self.unit), ("representation", self.representation)):
            if not isinstance(value, str) or not value:
                msg = f"Field {label} must be a non-empty string."
                raise ValueError(msg)
        if self.role not in _FIELD_ROLES:
            msg = f"Field {self.name!r} has unsupported role {self.role!r}."
            raise ValueError(msg)
        if self.source_name is not None and (not isinstance(self.source_name, str) or not self.source_name):
            msg = f"Field {self.name!r} source_name must be None or a non-empty string."
            raise ValueError(msg)

    def as_dict(self) -> dict[str, str | None]:
        """
        Return a JSON-serializable field contract.

        Returns
        -------
        dict[str, str | None]
            Field name, role, unit, representation, and optional source name.

        """
        return {
            "name": self.name,
            "role": self.role,
            "unit": self.unit,
            "representation": self.representation,
            "source_name": self.source_name,
        }


@dataclass(frozen=True, slots=True)
class DatasetDefaults:
    """
    Describe logical dataset identifiers selected by a task.

    Attributes
    ----------
    train : str
        Default training dataset identifier.
    ood : tuple[str, ...]
        Ordered default out-of-distribution dataset identifiers.

    """

    train: str
    ood: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject ambiguous or duplicate logical dataset defaults."""
        if not isinstance(self.train, str) or not self.train:
            msg = "Default training dataset id must be a non-empty string."
            raise ValueError(msg)
        if not isinstance(self.ood, tuple) or not self.ood or any(not isinstance(name, str) or not name for name in self.ood):
            msg = "Default OOD datasets must be a non-empty tuple of non-empty strings."
            raise ValueError(msg)
        if len(set(self.ood)) != len(self.ood) or self.train in self.ood:
            msg = "Default dataset ids must be unique across train and OOD roles."
            raise ValueError(msg)

    def as_dict(self) -> dict[str, object]:
        """
        Return a JSON-serializable dataset contract.

        Returns
        -------
        dict[str, object]
            Logical train and ordered OOD dataset identifiers.

        """
        return {"train": self.train, "ood": list(self.ood)}


@dataclass(frozen=True, slots=True)
class PreprocessingSpec:
    """
    Describe task-owned preprocessing assumptions.

    Attributes
    ----------
    input_normalization : str
        Semantic input normalization strategy.
    output_normalization : str
        Semantic output normalization strategy.
    fit_split : str
        Dataset split used to fit preprocessing statistics.

    """

    input_normalization: str
    output_normalization: str
    fit_split: str

    def __post_init__(self) -> None:
        """Require explicit normalization strategies fitted only on training data."""
        if not isinstance(self.input_normalization, str) or not self.input_normalization:
            msg = "Input normalization strategy must be a non-empty string."
            raise ValueError(msg)
        if not isinstance(self.output_normalization, str) or not self.output_normalization:
            msg = "Output normalization strategy must be a non-empty string."
            raise ValueError(msg)
        if self.fit_split != "train":
            msg = f"Normalizer fit_split must be 'train', got {self.fit_split!r}."
            raise ValueError(msg)

    def as_dict(self) -> dict[str, str]:
        """
        Return a JSON-serializable preprocessing contract.

        Returns
        -------
        dict[str, str]
            Input/output normalization assumptions and fit split.

        """
        return {
            "input_normalization": self.input_normalization,
            "output_normalization": self.output_normalization,
            "fit_split": self.fit_split,
        }


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """
    Describe one semantic evaluation metric selected by a task.

    Attributes
    ----------
    id : str
        Unique objective/reporting identifier within the task config.
    kind : str
        Semantic metric implementation identifier.
    space : MetricSpace
        Normalized or physical tensor space.
    fields : tuple[str, ...]
        Ordered task output fields evaluated by the metric.
    reduction : MetricReduction
        Semantic reduction across samples or elements.
    direction : OptimizationDirection
        Objective optimization direction.

    """

    id: str
    kind: str
    space: MetricSpace
    fields: tuple[str, ...]
    reduction: MetricReduction
    direction: OptimizationDirection

    def __post_init__(self) -> None:
        """Reject invalid runtime literals and ambiguous metric declarations."""
        if not isinstance(self.id, str) or not self.id or not isinstance(self.kind, str) or not self.kind:
            msg = "Metric id and kind must be non-empty strings."
            raise ValueError(msg)
        if self.space not in _METRIC_SPACES:
            msg = f"Metric {self.id!r} has unsupported space {self.space!r}."
            raise ValueError(msg)
        if not isinstance(self.fields, tuple) or not self.fields or any(not isinstance(field, str) or not field for field in self.fields):
            msg = f"Metric {self.id!r} fields must be a non-empty tuple of strings."
            raise ValueError(msg)
        if len(set(self.fields)) != len(self.fields):
            msg = f"Metric {self.id!r} contains duplicate fields."
            raise ValueError(msg)
        if self.reduction not in _METRIC_REDUCTIONS:
            msg = f"Metric {self.id!r} has unsupported reduction {self.reduction!r}."
            raise ValueError(msg)
        if self.direction not in _OPTIMIZATION_DIRECTIONS:
            msg = f"Metric {self.id!r} has unsupported direction {self.direction!r}."
            raise ValueError(msg)

    def as_dict(self, *, all_fields: tuple[str, ...]) -> dict[str, object]:
        """
        Return a JSON-serializable semantic metric declaration.

        Parameters
        ----------
        all_fields : tuple[str, ...]
            Complete ordered task outputs used to encode an all-fields selection.

        Returns
        -------
        dict[str, object]
            Semantic metric identifier, kind, space, fields, reduction, and direction.

        """
        fields: object = "all" if self.fields == all_fields else list(self.fields)
        return {
            "id": self.id,
            "kind": self.kind,
            "space": self.space,
            "fields": fields,
            "reduction": self.reduction,
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class PhysicsSpec:
    """
    Describe the semantic selector for a reusable task equation set.

    Attributes
    ----------
    kind : str
        Physics registry identifier.
    equation_set : str
        Descriptive equation-set identifier.
    continuity : str
        Descriptive continuity-formulation identifier.
    boundary : str
        Descriptive boundary-formulation identifier.

    """

    kind: str
    equation_set: str
    continuity: str
    boundary: str

    def __post_init__(self) -> None:
        """Require explicit non-empty semantic physics selectors."""
        values = (self.kind, self.equation_set, self.continuity, self.boundary)
        if any(not isinstance(value, str) or not value for value in values):
            msg = "Physics kind, equation_set, continuity, and boundary must be non-empty strings."
            raise ValueError(msg)

    def as_dict(self) -> dict[str, str]:
        """
        Return a JSON-serializable physics selection.

        Returns
        -------
        dict[str, str]
            Semantic physics selector and descriptive formulation identifiers.

        """
        return {
            "kind": self.kind,
            "equation_set": self.equation_set,
            "continuity": self.continuity,
            "boundary": self.boundary,
        }


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """
    Describe the complete immutable contract for one learned operator task.

    Attributes
    ----------
    id : str
        Canonical task identifier.
    schema_version : int
        Task-contract schema version.
    inputs : tuple[FieldSpec, ...]
        Exact ordered learned input fields.
    outputs : tuple[FieldSpec, ...]
        Exact ordered learned output fields.
    tensor_layout : tuple[str, ...]
        Named model tensor axes.
    operator_axes : tuple[int, ...]
        Spatial axes operated on by the neural operator.
    normalization_axes : tuple[int, ...]
        Axes used to fit per-channel normalization.
    default_datasets : DatasetDefaults
        Logical default train and OOD datasets.
    preprocessing : PreprocessingSpec
        Task-owned preprocessing assumptions.
    data_losses : tuple[str, ...]
        Allowed semantic data-loss kinds.
    default_metrics : tuple[MetricSpec, ...]
        Default semantic evaluation metrics and objective ordering.
    physics : PhysicsSpec
        Task-owned physics selector.

    """

    id: str
    schema_version: int
    inputs: tuple[FieldSpec, ...]
    outputs: tuple[FieldSpec, ...]
    tensor_layout: tuple[str, ...]
    operator_axes: tuple[int, ...]
    normalization_axes: tuple[int, ...]
    default_datasets: DatasetDefaults
    preprocessing: PreprocessingSpec
    data_losses: tuple[str, ...]
    default_metrics: tuple[MetricSpec, ...]
    physics: PhysicsSpec

    def __post_init__(self) -> None:
        """Reject structurally ambiguous task declarations."""
        if not isinstance(self.id, str) or not self.id:
            msg = "Task id must be a non-empty string."
            raise ValueError(msg)
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version <= 0:
            msg = f"Task {self.id!r} schema_version must be a positive integer."
            raise ValueError(msg)
        if self.tensor_layout != _SUPPORTED_TENSOR_LAYOUT:
            msg = f"Task {self.id!r} tensor_layout must be the current 2D layout {_SUPPORTED_TENSOR_LAYOUT!r}."
            raise ValueError(msg)
        if self.operator_axes != _SUPPORTED_OPERATOR_AXES or self.normalization_axes != _SUPPORTED_NORMALIZATION_AXES:
            msg = (
                f"Task {self.id!r} axes must match current 2D operator/normalizer support: "
                f"operator={_SUPPORTED_OPERATOR_AXES!r}, normalization={_SUPPORTED_NORMALIZATION_AXES!r}."
            )
            raise ValueError(msg)
        if (
            not isinstance(self.inputs, tuple)
            or not self.inputs
            or any(not isinstance(field, FieldSpec) for field in self.inputs)
            or not isinstance(self.outputs, tuple)
            or not self.outputs
            or any(not isinstance(field, FieldSpec) for field in self.outputs)
        ):
            msg = f"Task {self.id!r} must declare non-empty FieldSpec input/output tuples."
            raise ValueError(msg)
        if not isinstance(self.default_datasets, DatasetDefaults):
            msg = f"Task {self.id!r} default_datasets must be a DatasetDefaults declaration."
            raise TypeError(msg)
        if not isinstance(self.preprocessing, PreprocessingSpec):
            msg = f"Task {self.id!r} preprocessing must be a PreprocessingSpec declaration."
            raise TypeError(msg)
        if not isinstance(self.physics, PhysicsSpec):
            msg = f"Task {self.id!r} physics must be a PhysicsSpec declaration."
            raise TypeError(msg)
        if not isinstance(self.data_losses, tuple) or not self.data_losses:
            msg = f"Task {self.id!r} must declare at least one semantic data loss."
            raise ValueError(msg)
        if any(not isinstance(loss, str) or not loss for loss in self.data_losses) or len(set(self.data_losses)) != len(self.data_losses):
            msg = f"Task {self.id!r} data_losses must be unique non-empty strings."
            raise ValueError(msg)
        if (
            not isinstance(self.default_metrics, tuple)
            or not self.default_metrics
            or any(not isinstance(metric, MetricSpec) for metric in self.default_metrics)
        ):
            msg = f"Task {self.id!r} must declare a non-empty MetricSpec tuple."
            raise ValueError(msg)

        names = (*self.input_names, *self.output_names)
        if len(names) != len(set(names)):
            msg = f"Task {self.id!r} contains duplicate field names: {names!r}."
            raise ValueError(msg)

        metric_ids = tuple(metric.id for metric in self.default_metrics)
        if len(metric_ids) != len(set(metric_ids)):
            msg = f"Task {self.id!r} contains duplicate default metric ids: {metric_ids!r}."
            raise ValueError(msg)
        unknown_metric_fields = sorted({field for metric in self.default_metrics for field in metric.fields if field not in self.output_names})
        if unknown_metric_fields:
            msg = f"Task {self.id!r} metrics reference unknown output fields: {unknown_metric_fields!r}."
            raise ValueError(msg)

    @property
    def input_names(self) -> tuple[str, ...]:
        """
        Return exact ordered input field names.

        Returns
        -------
        tuple[str, ...]
            Canonical task input order.

        """
        return tuple(field.name for field in self.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        """
        Return exact ordered output field names.

        Returns
        -------
        tuple[str, ...]
            Canonical task output order.

        """
        return tuple(field.name for field in self.outputs)

    @property
    def in_channels(self) -> int:
        """
        Return the input-channel count derived from ordered fields.

        Returns
        -------
        int
            Number of declared task inputs.

        """
        return len(self.inputs)

    @property
    def out_channels(self) -> int:
        """
        Return the output-channel count derived from ordered fields.

        Returns
        -------
        int
            Number of declared task outputs.

        """
        return len(self.outputs)

    @property
    def operator_dimensionality(self) -> int:
        """
        Return the number of task-owned operator axes.

        Returns
        -------
        int
            Spatial operator dimensionality.

        """
        return len(self.operator_axes)

    @property
    def default_objective(self) -> MetricSpec:
        """
        Return the task default evaluation objective.

        Returns
        -------
        MetricSpec
            First declared default metric.

        """
        return self.default_metrics[0]

    def field(self, name: str) -> FieldSpec:
        """
        Resolve one field by its exact canonical name.

        Parameters
        ----------
        name : str
            Canonical task input or output field name.

        Returns
        -------
        FieldSpec
            Immutable matching field descriptor.

        Raises
        ------
        ValueError
            If `name` is not declared by the task.

        """
        for field in (*self.inputs, *self.outputs):
            if field.name == name:
                return field
        available = ", ".join((*self.input_names, *self.output_names))
        msg = f"Unknown field {name!r} for task {self.id!r}. Available fields: {available}."
        raise ValueError(msg)

    def contract_payload(self) -> dict[str, object]:
        """
        Return the canonical payload used for persistence and hashing.

        Returns
        -------
        dict[str, object]
            Complete JSON-serializable task contract without derived digest metadata.

        """
        return {
            "task": self.id,
            "schema_version": self.schema_version,
            "inputs": [field.as_dict() for field in self.inputs],
            "outputs": [field.as_dict() for field in self.outputs],
            "tensor_layout": list(self.tensor_layout),
            "operator_axes": list(self.operator_axes),
            "normalization_axes": list(self.normalization_axes),
            "operator_dimensionality": self.operator_dimensionality,
            "default_datasets": self.default_datasets.as_dict(),
            "preprocessing": self.preprocessing.as_dict(),
            "data_losses": list(self.data_losses),
            "default_metrics": [metric.as_dict(all_fields=self.output_names) for metric in self.default_metrics],
            "physics": self.physics.as_dict(),
        }

    @property
    def contract_digest(self) -> str:
        """
        Return a stable SHA-256 digest of the complete task contract.

        Returns
        -------
        str
            Lowercase hexadecimal SHA-256 digest.

        """
        canonical = json.dumps(
            self.contract_payload(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def resolved_contract(self) -> dict[str, object]:
        """
        Return the persisted resolved task contract with derived metadata.

        Returns
        -------
        dict[str, object]
            Canonical contract plus digest and derived channel counts.

        """
        payload = self.contract_payload()
        payload["digest"] = self.contract_digest
        payload["in_channels"] = self.in_channels
        payload["out_channels"] = self.out_channels
        return payload
