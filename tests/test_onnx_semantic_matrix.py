# /**
#   ******************************************************************************
#   * @file        test_onnx_semantic_matrix.py
#   * @author      Egor Izmaylov
#   * @brief       Covers the official ONNX semantic matrix gate.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from tools import onnx_semantic_matrix
from tools.audit_ops import OperatorInfo


def _info(class_name, *, numerical=True, cuda=True, c_runtime=True):
    c_functions = (f"{class_name.lower()}_forward",) if c_runtime else ()
    return OperatorInfo(
        class_name=class_name,
        line=1,
        bases=("Ops",),
        has_forward=True,
        has_forward_shape=True,
        c_functions=c_functions,
        c_runtime_functions=c_functions,
        c_runtime_kind="C-backed" if c_runtime else "Python orchestration",
        runtime_uses_numpy=False,
        import_supported=True,
        cuda_verified=cuda,
        numerical_planned=numerical,
        status="已数值验证" if numerical and cuda else "已 pytest 语义验证",
        notes=(),
    )


def test_deprecated_official_aliases_are_verified_by_canonical_operator_classes(monkeypatch):
    infos = [_info("ScatterElements"), _info("Resize")]
    metadata = {"operator_class_count": len(infos)}

    monkeypatch.setattr(onnx_semantic_matrix, "audit", lambda: (infos, metadata))
    monkeypatch.setattr(
        onnx_semantic_matrix,
        "parse_latest_official_schema_details",
        lambda: (
            {
                "SCATTER": {"op_type": "Scatter", "since_version": 11, "deprecated": True},
                "UPSAMPLE": {"op_type": "Upsample", "since_version": 10, "deprecated": True},
            },
            None,
        ),
    )
    monkeypatch.setattr(onnx_semantic_matrix, "parse_import_supported_raw_ops", lambda: {"Scatter", "Upsample"})
    monkeypatch.setattr(onnx_semantic_matrix, "parse_import_supported_ops", lambda: set())

    payload, failures = onnx_semantic_matrix.build_matrix()

    assert failures == []
    assert payload["verified_count"] == 2
    assert payload["missing_or_weak_count"] == 0
    assert payload["deprecated_alias_count"] == 2

    rows = {row["op_type"]: row for row in payload["rows"]}
    assert rows["Scatter"]["direct_operator_classes"] == []
    assert rows["Scatter"]["semantic_alias"] == {
        "kind": "deprecated_alias",
        "canonical_operator_classes": ["ScatterElements"],
    }
    assert rows["Scatter"]["status"] == "verified"
    assert rows["Upsample"]["semantic_alias"]["canonical_operator_classes"] == ["Resize"]


def test_row_status_distinguishes_runtime_only_from_strong_semantic_evidence():
    assert onnx_semantic_matrix.row_status(True, [_info("Resize")], {"c_runtime"}) == "runtime_only"
    assert (
        onnx_semantic_matrix.row_status(True, [_info("Resize")], {"c_runtime", "numerical_plan"})
        == "verified"
    )
    assert onnx_semantic_matrix.row_status(False, [_info("Resize")], {"numerical_plan"}) == "missing_import"
