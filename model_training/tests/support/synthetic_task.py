"""
Build the unregistered synthetic TaskSpec used by generic contract tests.

The test-only three-input/two-output contract uses arbitrary neutral field names,
units, metrics, and dataset IDs to prove current TaskSpec consumers do not hardcode
the steady-flow channel contract. It is not a production task implementation and
is never registered, trained, or published.
"""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from src import domain


def _invert_kc_response(target: float) -> float:
    """Invert the monotone dimensionless KC response for a test fixture."""
    lower = 0.0
    upper = 1.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        response = midpoint**3 / (1.0 - midpoint) ** 2
        if response > target:
            upper = midpoint
        else:
            lower = midpoint
    return 0.5 * (lower + upper)


def build_synthetic_generation_contract(variation: float) -> dict[str, Any]:
    """Return one internally consistent version-1 packing contract fixture."""
    kappa_nominal = 1e-10
    eps_nominal = 0.5
    reference = kappa_nominal / (eps_nominal**3 / (1.0 - eps_nominal) ** 2)
    reference_variation = 0.8
    natural_kappa = [
        kappa_nominal / (1.0 + reference_variation),
        kappa_nominal * (1.0 + reference_variation),
    ]
    batch_kappa = [
        kappa_nominal / (1.0 + variation),
        kappa_nominal * (1.0 + variation),
    ]
    return {
        "generation_contract_version": 1,
        "kappa_nominal": kappa_nominal,
        "eps_nominal": eps_nominal,
        "A_KC_reference": reference,
        "reference_id_variation": reference_variation,
        "natural_kappa_support": natural_kappa,
        "natural_eps_reference_support": [_invert_kc_response(value / reference) for value in natural_kappa],
        "batch_kappa_support": batch_kappa,
        "batch_eps_reference_support": [_invert_kc_response(value / reference) for value in batch_kappa],
        "packing_scatter_truncation_lower": -3,
        "packing_scatter_truncation_upper": 3,
        "eps_min_global": 0.3,
        "eps_max_global": 0.8,
    }


def build_synthetic_generated_batch_identity(
    *,
    batch_name: str,
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Return one complete deterministic version-1 generated-batch identity."""
    case_ids = list(sample_ids)
    configuration = {
        "method": "lhs",
        "variation": 0.8,
        "N": len(case_ids),
        "seed": 17,
        "Lx": 2.0,
        "Ly": 1.0,
        "res": 0.5,
        "save_model": False,
        "template_name": "synthetic_template.mph",
        "template_sha256": hashlib.sha256(b"synthetic-template").hexdigest(),
    }
    configuration["generation_contract"] = build_synthetic_generation_contract(configuration["variation"])
    sources = [
        {
            "case_id": case_id,
            "raw_csv_sha256": hashlib.sha256(f"{batch_name}:{case_id}:raw".encode()).hexdigest(),
            "solution_csv_sha256": hashlib.sha256(f"{batch_name}:{case_id}:solution".encode()).hexdigest(),
            "solution_model_sha256": "",
        }
        for case_id in case_ids
    ]
    content: dict[str, Any] = {
        "schema_version": 1,
        "batch_name": batch_name,
        "configuration": configuration,
        "field_schema": {
            "input_columns": ["x", "y", "Kxx", "Kxy", "Kyy", "eps", "p_bc"],
            "solution_columns": [
                "x",
                "y",
                "kappaxx",
                "kappayx",
                "kappaxy",
                "kappayy",
                "eps",
                "p_bc",
                "p",
                "u",
                "v",
                "U",
            ],
        },
        "intended_case_ids": case_ids,
        "scientific_case_sources": sources,
        "sampling": {
            "method": configuration["method"],
            "variation": configuration["variation"],
            "N": configuration["N"],
            "seed": configuration["seed"],
            "base": {"synthetic_parameter": 1.0},
            "param_names": ["synthetic_parameter"],
            "generation_contract": configuration["generation_contract"],
        },
    }
    encoded = json.dumps(
        content,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    content["batch_manifest_identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return content


def build_synthetic_task() -> domain.tasks.spec.TaskSpec:
    """
    Return a coherent task with distinct fields, per-field metrics, and no physics.

    Returns
    -------
    domain.tasks.spec.TaskSpec
        Unregistered two-output contract used to prove consumers derive channel
        counts, names, units, and metric definitions from TaskSpec.

    Notes
    -----
    The fixture intentionally simplifies production assumptions: fields use
    arbitrary units, preprocessing uses generic standardization, dataset IDs are
    non-executable labels, and physics kind/equations/boundary are ``none``.

    """
    spec = domain.tasks.spec
    return spec.TaskSpec(
        id="synthetic_field_task",
        schema_version=spec.TASK_SCHEMA_VERSION,
        inputs=(
            spec.FieldSpec("feature_a", "state", "unit_in_a", "identity"),
            spec.FieldSpec("feature_b", "state", "unit_in_b", "identity"),
            spec.FieldSpec("feature_c", "state", "unit_in_c", "identity"),
        ),
        outputs=(
            spec.FieldSpec("response_a", "state", "unit_out_a", "identity"),
            spec.FieldSpec("response_b", "state", "unit_out_b", "identity"),
        ),
        output_groups=(
            spec.OutputGroupSpec("quantity_a", ("response_a",)),
            spec.OutputGroupSpec("quantity_b", ("response_b",)),
        ),
        tensor_layout=("batch", "channel", "y", "x"),
        operator_axes=(2, 3),
        normalization_axes=(0, 2, 3),
        default_datasets=spec.DatasetDefaults(train="synthetic_train", ood=("synthetic_ood",)),
        preprocessing=spec.PreprocessingSpec(
            input_normalization="standard",
            output_normalization="standard",
            fit_split="train",
        ),
        data_losses=("relative_l2",),
        default_metrics=(
            spec.MetricSpec(
                id="normalized_group_macro_rmse",
                kind="group_macro_rmse",
                space="physical",
                fields=("response_a", "response_b"),
                reduction="group_macro_element_mean",
                direction="minimize",
            ),
            spec.MetricSpec(
                id="normalized_relative_l2",
                kind="relative_l2",
                space="normalized",
                fields=("response_a", "response_b"),
                reduction="sample_mean",
                direction="minimize",
            ),
            spec.MetricSpec(
                id="physical_rmse_response_a",
                kind="rmse",
                space="physical",
                fields=("response_a",),
                reduction="element_mean",
                direction="minimize",
            ),
            spec.MetricSpec(
                id="physical_rmse_response_b",
                kind="rmse",
                space="physical",
                fields=("response_b",),
                reduction="element_mean",
                direction="minimize",
            ),
        ),
        physics=spec.PhysicsSpec(
            kind="none",
            equation_set="none",
            continuity="none",
            allowed_continuities=("none",),
            boundary="none",
        ),
    )
