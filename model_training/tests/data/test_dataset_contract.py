# ruff: noqa: S101
"""Verify strict case, merged-dataset, builder, and merger schemas."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import pytest
import torch
from src import datasets, domain

from data_generation.matlab.build_batch_dataset import build_batch_dataset
from data_generation.matlab.merge_batch_cases import merge_batch_cases

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_valid_seven_input_three_output_case_is_canonical(
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    """Mapping insertion order cannot alter exact task tensor order."""
    reverse_inputs = OrderedDict(
        (
            name,
            torch.full((2, 3), float(index)),
        )
        for index, name in reversed(tuple(enumerate(steady_task.input_names)))
    )
    reverse_outputs = OrderedDict(
        (
            name,
            torch.full((2, 3), float(index + 20)),
        )
        for index, name in reversed(tuple(enumerate(steady_task.output_names)))
    )
    payload = datasets.identity.build_case_payload(
        task=steady_task,
        case_id="case_0000",
        input_fields=reverse_inputs,
        output_fields=reverse_outputs,
        source_identity={"token": "case_0000"},
        source_metadata={},
    )
    validated = datasets.identity.validate_case_payload(payload, task=steady_task)

    assert payload["fields"]["inputs"] == list(steady_task.input_names)
    assert payload["fields"]["outputs"] == list(steady_task.output_names)
    assert validated.inputs.shape == (7, 2, 3)
    assert validated.outputs.shape == (3, 2, 3)
    assert validated.inputs[:, 0, 0].tolist() == list(range(7))


@pytest.mark.parametrize(
    ("missing", "noncanonical"),
    [
        ("eps", None),
        ("p_bc", None),
        ("eps", "phi"),
        ("p_bc", "pbc"),
    ],
    ids=("missing-eps", "missing-p-bc", "noncanonical-phi", "noncanonical-pbc"),
)
def test_missing_and_noncanonical_input_fields_fail(
    steady_task: domain.tasks.spec.TaskSpec,
    missing: str,
    noncanonical: str | None,
) -> None:
    """Missing required fields and noncanonical spellings fail closed."""
    inputs = {name: torch.zeros((2, 3)) for name in steady_task.input_names if name != missing}
    if noncanonical is not None:
        inputs[noncanonical] = torch.zeros((2, 3))
    outputs = {name: torch.zeros((2, 3)) for name in steady_task.output_names}

    with pytest.raises(ValueError, match=r"Missing|unexpected"):
        datasets.identity.build_case_payload(
            task=steady_task,
            case_id="case_0000",
            input_fields=inputs,
            output_fields=outputs,
            source_identity={},
            source_metadata={},
        )


def test_duplicate_and_wrong_order_declarations_fail(
    steady_task: domain.tasks.spec.TaskSpec,
    case_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Persisted declarations cannot duplicate or reorder task fields."""
    duplicate = case_payload_factory()
    duplicate["fields"]["inputs"][4] = "kxy"
    with pytest.raises(ValueError, match="duplicate"):
        datasets.identity.validate_case_payload(duplicate, task=steady_task)

    wrong_order = case_payload_factory()
    fields = wrong_order["fields"]["inputs"]
    fields[3], fields[4] = fields[4], fields[3]
    with pytest.raises(ValueError, match="wrong channel order"):
        datasets.identity.validate_case_payload(wrong_order, task=steady_task)


def test_inconsistent_case_field_shapes_fail(
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    """A case cannot stack fields with inconsistent spatial shapes."""
    inputs = {name: torch.zeros((2, 3)) for name in steady_task.input_names}
    inputs["eps"] = torch.zeros((3, 3))
    outputs = {name: torch.zeros((2, 3)) for name in steady_task.output_names}
    with pytest.raises(ValueError, match="inconsistent spatial shapes"):
        datasets.identity.build_case_payload(
            task=steady_task,
            case_id="case_0000",
            input_fields=inputs,
            output_fields=outputs,
            source_identity={},
            source_metadata={},
        )


def test_extra_output_and_wrong_tensor_counts_fail(
    steady_task: domain.tasks.spec.TaskSpec,
    case_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Unexpected learned outputs and channel/label disagreement fail."""
    case = case_payload_factory()
    inputs = dict(case["input_fields"])
    outputs = dict(case["output_fields"])
    outputs["U"] = torch.zeros((2, 3))
    with pytest.raises(ValueError, match="unexpected"):
        datasets.identity.build_case_payload(
            task=steady_task,
            case_id="case_0000",
            input_fields=inputs,
            output_fields=outputs,
            source_identity={},
            source_metadata={},
        )

    validated = datasets.identity.validate_case_payload(case, task=steady_task)
    with pytest.raises(ValueError, match="channel counts"):
        datasets.identity.build_merged_dataset_payload(
            task=steady_task,
            dataset_id="wrong_channels",
            sample_ids=(validated.case_id,),
            source_identities=(validated.source_identity,),
            source_metadata=(validated.source_metadata,),
            case_fingerprints=(validated.fingerprint,),
            inputs=validated.inputs[:-1].unsqueeze(0),
            outputs=validated.outputs.unsqueeze(0),
        )


_COMSOL_HEADER = (
    "% x (m);y (m);br.kappaxx (m^2);br.kappayx (m^2);br.kappaxy (m^2);br.kappayy (m^2);int4(x,y) (1);int5(x,y) (Pa);p (Pa);u (m/s);v (m/s);br.U (m/s)"
)
_MANIFEST_FIELD_SCHEMA = {
    "input_columns": ["x", "y", "Kxx", "Kxy", "Kyy", "eps", "p_bc"],
    "solution_columns": ["x", "y", "kappaxx", "kappayx", "kappaxy", "kappayy", "eps", "p_bc", "p", "u", "v", "U"],
}


def _sha256(path: Path) -> str:
    """Hash one small synthetic producer artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_raw_export(raw_dir: Path, case_id: str, rows: list[list[Any]]) -> None:
    """Write the seven-column authoritative input export for one synthetic case."""
    raw_rows = [[row[index] for index in (0, 1, 2, 4, 5, 6, 7)] for row in rows]
    content = "\n".join(";".join(map(str, row)) for row in raw_rows) + "\n"
    (raw_dir / f"{case_id}.csv").write_text(content, encoding="utf-8")


def _write_batch_manifest(
    raw_dir: Path,
    processed_dir: Path,
    batch_name: str,
    *,
    status: str = "complete",
    save_model: bool = False,
) -> None:
    """Publish the exact terminal synthetic producer contract and file hashes."""
    case_ids = sorted(path.stem for path in raw_dir.glob("case_*.json"))
    records = []
    for case_id in case_ids:
        raw_csv = raw_dir / f"{case_id}.csv"
        raw_json = raw_dir / f"{case_id}.json"
        solution_csv = processed_dir / f"{case_id}_sol.csv"
        solution_model = processed_dir / f"{case_id}_sol.mph"
        records.append(
            {
                "case_id": case_id,
                "status": "complete",
                "stage": "simulation",
                "message": "",
                "files": {
                    "raw_csv_sha256": _sha256(raw_csv),
                    "raw_json_sha256": _sha256(raw_json),
                    "solution_csv_sha256": _sha256(solution_csv),
                    "solution_model_sha256": _sha256(solution_model) if save_model else "",
                },
            },
        )
    manifest = {
        "schema_kind": "comsol_batch_manifest",
        "schema_version": 1,
        "batch_name": batch_name,
        "status": status,
        "configuration": {
            "method": "lhs",
            "variation": 0.2,
            "N": max(len(case_ids), 1),
            "seed": 17,
            "Lx": 1.0,
            "Ly": 1.0,
            "res": 0.5,
            "save_model": save_model,
            "sample_sha256": "0" * 64,
            "template_name": "template_brinkman.mph",
            "template_sha256": "1" * 64,
        },
        "field_schema": _MANIFEST_FIELD_SCHEMA,
        "intended_case_ids": case_ids,
        "cases": records,
    }
    (raw_dir / "batch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_synthetic_comsol_case(
    raw_dir: Path,
    processed_dir: Path,
    case_index: int,
    *,
    include_pressure: bool = True,
) -> None:
    """Write one small unit-bearing COMSOL export and matching metadata."""
    case_id = f"case_{case_index:04d}"
    (raw_dir / f"{case_id}.json").write_text(
        json.dumps({"geometry": {"nx": 2, "ny": 2}}),
        encoding="utf-8",
    )
    offset = 10 * case_index
    rows = [
        ["0", "0", "1e-10", "0", "0", "2e-10", "0.4", "100", str(10 + offset), "1", "2", "3"],
        ["1", "0", "1e-10", "0", "0", "2e-10", "0.5", "101", str(11 + offset), "2", "3", "4"],
        ["0", "1", "1e-10", "0", "0", "2e-10", "0.6", "102", str(12 + offset), "3", "4", "5"],
        ["1", "1", "1e-10", "0", "0", "2e-10", "0.7", "103", str(13 + offset), "4", "5", "6"],
    ]
    _write_raw_export(raw_dir, case_id, rows)
    header = _COMSOL_HEADER
    if not include_pressure:
        header = header.replace(";p (Pa)", "")
        for row in rows:
            row.pop(8)
    content = f"{header}\n" + "\n".join(";".join(row) for row in rows) + "\n"
    (processed_dir / f"{case_id}_sol.csv").write_text(content, encoding="utf-8")
    _write_batch_manifest(raw_dir, processed_dir, raw_dir.name)


def test_synthetic_comsol_case_builder_emits_strict_schema(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    """The real CSV/JSON builder and merger emit the exact task payloads."""
    generated_root = tmp_path / "generated"
    raw_dir = generated_root / "raw" / "synthetic"
    processed_dir = generated_root / "processed" / "synthetic"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    for case_index in range(2):
        _write_synthetic_comsol_case(raw_dir, processed_dir, case_index)

    result = build_batch_dataset(
        "synthetic",
        task_id=steady_task.id,
        generated_data_root=generated_root,
        dataset_root=tmp_path / "datasets",
    )
    payload = torch.load(
        result["cases_dir"] / "case_0000.pt",
        map_location="cpu",
        weights_only=False,
    )
    validated = datasets.identity.validate_case_payload(payload, task=steady_task)
    merged_result = merge_batch_cases(
        "synthetic",
        task_id=steady_task.id,
        dataset_root=tmp_path / "datasets",
    )
    merged = torch.load(
        merged_result["dataset_path"],
        map_location="cpu",
        weights_only=False,
    )
    merged_identity = datasets.identity.validate_merged_dataset_payload(
        merged,
        task=steady_task,
    )
    loaded = datasets.simulation.create_task_dataset(
        merged_result["dataset_path"],
        task=steady_task,
    )

    assert payload["fields"]["inputs"] == list(steady_task.input_names)
    assert payload["fields"]["outputs"] == list(steady_task.output_names)
    assert validated.inputs.shape == (7, 2, 2)
    assert validated.outputs.shape == (3, 2, 2)
    assert torch.count_nonzero(validated.inputs[3]).item() == 0
    assert validated.inputs[5, 0, 0].item() == pytest.approx(0.4)
    assert validated.inputs[6, 0, 0].item() == pytest.approx(100.0)
    assert merged_identity.sample_ids == ("case_0000", "case_0001")
    assert merged["source_metadata"] == [payload["source_metadata"], payload["source_metadata"]]
    assert loaded[0]["meta"] == payload["source_metadata"]
    assert merged["inputs"].shape == (2, 7, 2, 2)
    assert merged["outputs"].shape == (2, 3, 2, 2)
    assert torch.equal(merged["inputs"][0], validated.inputs)
    assert merged["outputs"][1, 0, 0, 0].item() == pytest.approx(20.0)


@pytest.mark.parametrize("missing_side", ["solution", "metadata"])
def test_builder_rejects_incomplete_generated_batches(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    missing_side: str,
) -> None:
    """Metadata and solution membership must match before publication."""
    generated_root = tmp_path / "generated"
    raw_dir = generated_root / "raw" / "incomplete"
    processed_dir = generated_root / "processed" / "incomplete"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    _write_synthetic_comsol_case(raw_dir, processed_dir, 0)
    if missing_side == "solution":
        (processed_dir / "case_0000_sol.csv").unlink()
    else:
        (raw_dir / "case_0000.json").unlink()

    dataset_root = tmp_path / "datasets"
    with pytest.raises(RuntimeError, match="file integrity failure: missing"):
        build_batch_dataset(
            "incomplete",
            task_id=steady_task.id,
            generated_data_root=generated_root,
            dataset_root=dataset_root,
        )
    assert not (dataset_root / "incomplete" / "cases").exists()


def test_builder_failure_does_not_publish_partial_cases(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    """A later invalid source case leaves no authoritative case directory."""
    generated_root = tmp_path / "generated"
    raw_dir = generated_root / "raw" / "invalid"
    processed_dir = generated_root / "processed" / "invalid"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    _write_synthetic_comsol_case(raw_dir, processed_dir, 0)
    _write_synthetic_comsol_case(
        raw_dir,
        processed_dir,
        1,
        include_pressure=False,
    )

    dataset_root = tmp_path / "datasets"
    with pytest.raises(KeyError, match="output source column 'p'"):
        build_batch_dataset(
            "invalid",
            task_id=steady_task.id,
            generated_data_root=generated_root,
            dataset_root=dataset_root,
        )
    batch_dir = dataset_root / "invalid"
    assert not (batch_dir / "cases").exists()
    assert not list(batch_dir.glob(".cases.*"))


def test_builder_sorts_a_shuffled_non_square_cartesian_grid(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    """CSV row order cannot change canonical y/x tensor orientation."""
    generated_root = tmp_path / "generated"
    raw_dir = generated_root / "raw" / "shuffled"
    processed_dir = generated_root / "processed" / "shuffled"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    case_id = "case_0000"
    (raw_dir / f"{case_id}.json").write_text(json.dumps({"geometry": {"nx": 3, "ny": 2}}))
    rows = [
        [x_value, y_value, 1e-10, 0.0, 0.0, 2e-10, 0.5, 100.0, x_value + 10 * y_value, 1.0, 2.0, 3.0]
        for y_value in (0.0, 2.0)
        for x_value in (0.0, 1.0, 2.0)
    ]
    rows = [rows[index] for index in (4, 0, 5, 2, 1, 3)]
    _write_raw_export(raw_dir, case_id, rows)
    content = _COMSOL_HEADER + "\n" + "\n".join(";".join(map(str, row)) for row in rows) + "\n"
    (processed_dir / f"{case_id}_sol.csv").write_text(content, encoding="utf-8")
    _write_batch_manifest(raw_dir, processed_dir, "shuffled")

    result = build_batch_dataset(
        "shuffled",
        task_id=steady_task.id,
        generated_data_root=generated_root,
        dataset_root=tmp_path / "datasets",
    )
    payload = torch.load(result["cases_dir"] / f"{case_id}.pt", map_location="cpu", weights_only=False)
    validated = datasets.identity.validate_case_payload(payload, task=steady_task)

    assert validated.inputs.shape == (7, 2, 3)
    assert torch.equal(validated.inputs[0], torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]))
    assert torch.equal(validated.inputs[1], torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]))
    assert torch.equal(validated.outputs[0], torch.tensor([[0.0, 1.0, 2.0], [20.0, 21.0, 22.0]]))


def test_builder_rejects_nonuniform_grid_and_invalid_physical_fields(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    """Uniform-grid, SPD permeability, porosity, and finite-state domains fail closed."""
    nonuniform_root = tmp_path / "nonuniform" / "generated"
    nonuniform_raw = nonuniform_root / "raw" / "nonuniform"
    nonuniform_processed = nonuniform_root / "processed" / "nonuniform"
    nonuniform_raw.mkdir(parents=True)
    nonuniform_processed.mkdir(parents=True)
    (nonuniform_raw / "case_0000.json").write_text(json.dumps({"geometry": {"nx": 3, "ny": 2}}))
    rows = [[x_value, y_value, 1e-10, 0.0, 0.0, 2e-10, 0.5, 100.0, 1.0, 1.0, 2.0, 3.0] for y_value in (0.0, 1.0) for x_value in (0.0, 1.0, 3.0)]
    _write_raw_export(nonuniform_raw, "case_0000", rows)
    (nonuniform_processed / "case_0000_sol.csv").write_text(
        _COMSOL_HEADER + "\n" + "\n".join(";".join(map(str, row)) for row in rows) + "\n",
        encoding="utf-8",
    )
    _write_batch_manifest(nonuniform_raw, nonuniform_processed, "nonuniform")
    with pytest.raises(ValueError, match="uniform"):
        build_batch_dataset(
            "nonuniform",
            task_id=steady_task.id,
            generated_data_root=nonuniform_root,
            dataset_root=tmp_path / "nonuniform" / "datasets",
        )

    mutations = {
        "non_spd": (3, "2e-10", "positive definite"),
        "porosity": (6, "0", "Porosity"),
        "nonfinite": (8, "nan", "non-finite"),
    }
    for batch_name, (column, replacement, message) in mutations.items():
        generated_root = tmp_path / batch_name / "generated"
        raw_dir = generated_root / "raw" / batch_name
        processed_dir = generated_root / "processed" / batch_name
        raw_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)
        _write_synthetic_comsol_case(raw_dir, processed_dir, 0)
        csv_path = processed_dir / "case_0000_sol.csv"
        lines = csv_path.read_text().splitlines()
        values = lines[1].split(";")
        values[column] = replacement
        if batch_name == "non_spd":
            values[4] = replacement
        lines[1] = ";".join(values)
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _write_batch_manifest(raw_dir, processed_dir, batch_name)

        with pytest.raises(ValueError, match=message):
            build_batch_dataset(
                batch_name,
                task_id=steady_task.id,
                generated_data_root=generated_root,
                dataset_root=tmp_path / batch_name / "datasets",
            )


def test_builder_requires_a_complete_terminal_manifest(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    """Missing or failed producer manifests cannot authorize dataset publication."""
    generated_root = tmp_path / "generated"
    raw_dir = generated_root / "raw" / "manifest"
    processed_dir = generated_root / "processed" / "manifest"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    _write_synthetic_comsol_case(raw_dir, processed_dir, 0)
    (raw_dir / "batch_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="terminal completion manifest"):
        build_batch_dataset(
            "manifest",
            task_id=steady_task.id,
            generated_data_root=generated_root,
            dataset_root=tmp_path / "datasets_missing",
        )

    _write_batch_manifest(raw_dir, processed_dir, "manifest", status="failed")
    with pytest.raises(RuntimeError, match="not complete"):
        build_batch_dataset(
            "manifest",
            task_id=steady_task.id,
            generated_data_root=generated_root,
            dataset_root=tmp_path / "datasets_failed",
        )


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("field-schema-extra", "field_schema must exactly match"),
        ("record-stage", "complete case records must exactly match"),
        ("record-extra", "keys do not match"),
    ],
)
def test_builder_rejects_malformed_complete_manifest_contract(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    variant: str,
    message: str,
) -> None:
    """A complete status never bypasses exact field-schema or case-record checks."""
    generated_root = tmp_path / "generated"
    raw_dir = generated_root / "raw" / "strict_manifest"
    processed_dir = generated_root / "processed" / "strict_manifest"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    _write_synthetic_comsol_case(raw_dir, processed_dir, 0)
    manifest_path = raw_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if variant == "field-schema-extra":
        manifest["field_schema"]["unexpected"] = []
    elif variant == "record-stage":
        manifest["cases"][0]["stage"] = "synthetic"
    elif variant == "record-extra":
        manifest["cases"][0]["unexpected"] = True
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(variant)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises((TypeError, ValueError, RuntimeError), match=message):
        build_batch_dataset(
            "strict_manifest",
            task_id=steady_task.id,
            generated_data_root=generated_root,
            dataset_root=tmp_path / "datasets",
        )


@pytest.mark.parametrize(
    ("operation", "key", "value", "message"),
    [
        ("delete", "template_sha256", None, "keys do not match"),
        ("extra", "unexpected", True, "keys do not match"),
        ("set", "N", 0, r"configuration\.N must be in"),
        ("set", "seed", -1, r"configuration\.seed must be in"),
        ("set", "save_model", 1, "save_model must be boolean"),
        ("set", "sample_sha256", "A" * 64, "64-character lowercase"),
        ("set", "res", 2.0, "cannot exceed the shorter domain"),
    ],
)
def test_builder_enforces_production_manifest_configuration(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    operation: str,
    key: str,
    value: Any,
    message: str,
) -> None:
    """Manifest configuration has required keys, exact JSON types, ranges, and SHA syntax."""
    generated_root = tmp_path / "generated"
    raw_dir = generated_root / "raw" / "bad_configuration"
    processed_dir = generated_root / "processed" / "bad_configuration"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    _write_synthetic_comsol_case(raw_dir, processed_dir, 0)
    manifest_path = raw_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configuration = manifest["configuration"]
    if operation == "delete":
        del configuration[key]
    else:
        configuration[key] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        build_batch_dataset(
            "bad_configuration",
            task_id=steady_task.id,
            generated_data_root=generated_root,
            dataset_root=tmp_path / "datasets",
        )


@pytest.mark.parametrize(
    ("target", "save_model"),
    [
        ("raw_csv", False),
        ("raw_json", False),
        ("solution_csv", False),
        ("solution_model", True),
    ],
)
def test_builder_rejects_authoritative_file_tampering(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    target: str,
    save_model: bool,
) -> None:
    """Every authoritative raw or solution artifact is bound by its SHA-256 digest."""
    generated_root = tmp_path / "generated"
    raw_dir = generated_root / "raw" / "tamper"
    processed_dir = generated_root / "processed" / "tamper"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    _write_synthetic_comsol_case(raw_dir, processed_dir, 0)
    solution_model = processed_dir / "case_0000_sol.mph"
    if save_model:
        solution_model.write_bytes(b"synthetic solved model")
        _write_batch_manifest(raw_dir, processed_dir, "tamper", save_model=True)
    targets = {
        "raw_csv": raw_dir / "case_0000.csv",
        "raw_json": raw_dir / "case_0000.json",
        "solution_csv": processed_dir / "case_0000_sol.csv",
        "solution_model": solution_model,
    }
    target_path = targets[target]
    target_path.write_bytes(target_path.read_bytes() + b"\ntampered")

    with pytest.raises(RuntimeError, match="file integrity failure: SHA-256 mismatch"):
        build_batch_dataset(
            "tamper",
            task_id=steady_task.id,
            generated_data_root=generated_root,
            dataset_root=tmp_path / "datasets",
        )


def test_unversioned_merged_payload_is_rejected(
    steady_task: domain.tasks.spec.TaskSpec,
) -> None:
    """A payload without the required schema declaration is unsupported."""
    unversioned = {
        "inputs": torch.zeros((1, 7, 2, 3)),
        "outputs": torch.zeros((1, 3, 2, 3)),
        "fields": {
            "inputs": list(steady_task.input_names),
            "outputs": list(steady_task.output_names),
        },
    }
    with pytest.raises(ValueError, match="Unsupported dataset schema"):
        datasets.modules.flow.FlowModule(unversioned, task=steady_task)


def test_merger_strictly_rejects_modified_case_content(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    case_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """Dataset construction rejects replacement content with a stale case hash."""
    cases_dir = tmp_path / "modified_case" / "cases"
    cases_dir.mkdir(parents=True)
    payload = case_payload_factory("case_0000")
    payload["input_fields"][steady_task.input_names[0]][0, 0] += 1.0
    torch.save(payload, cases_dir / "case_0000.pt")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        merge_batch_cases(
            "modified_case",
            task_id=steady_task.id,
            dataset_root=tmp_path,
        )


def test_merger_rejects_inconsistent_case_shapes(
    tmp_path: Path,
    steady_task: domain.tasks.spec.TaskSpec,
    case_payload_factory: Callable[..., dict[str, Any]],
) -> None:
    """A batch with individually valid but differently shaped cases fails."""
    cases_dir = tmp_path / "bad_shapes" / "cases"
    cases_dir.mkdir(parents=True)
    torch.save(case_payload_factory("case_0000", shape=(2, 3)), cases_dir / "case_0000.pt")
    torch.save(case_payload_factory("case_0001", shape=(3, 3)), cases_dir / "case_0001.pt")

    with pytest.raises(ValueError, match="inconsistent tensor shapes"):
        merge_batch_cases(
            "bad_shapes",
            task_id=steady_task.id,
            dataset_root=tmp_path,
        )
