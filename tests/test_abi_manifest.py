# /**
#   ******************************************************************************
#   * @file        test_abi_manifest.py
#   * @author      Egor Izmaylov
#   * @brief       覆盖 C ABI manifest 解析与差异检测。
#   * @details     2026.06.27  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import json

from tools.abi_manifest import build_manifest, compare_manifests, write_manifest


def test_abi_manifest_parses_public_enum_structs_and_functions(tmp_path):
    header = tmp_path / "tensor_ops.h"
    header.write_text(
        """
        typedef enum {
            DTYPE_FLOAT32,
            DTYPE_INT64,
        } DataType;

        typedef struct {
            void* data;
            int* shape;
            int ndim;
            DataType dtype;
        } Tensor;

        void add_forward(const Tensor* A, const Tensor* B, Tensor* O);
        int unique_forward(const Tensor* input, Tensor* values);
        """,
        encoding="utf-8",
    )

    manifest = build_manifest(header)

    assert manifest["data_type"] == ["DTYPE_FLOAT32", "DTYPE_INT64"]
    assert manifest["structs"]["Tensor"] == ["void* data", "int* shape", "int ndim", "DataType dtype"]
    assert manifest["functions"]["add_forward"]["return"] == "void"
    assert manifest["functions"]["unique_forward"]["args"] == ["const Tensor* input", "Tensor* values"]


def test_abi_manifest_compare_reports_signature_changes(tmp_path):
    baseline = {
        "data_type": ["DTYPE_FLOAT32"],
        "structs": {"Tensor": ["void* data"]},
        "functions": {"add_forward": {"return": "void", "args": ["const Tensor* A"]}},
    }
    changed = {
        "data_type": ["DTYPE_FLOAT32", "DTYPE_INT64"],
        "structs": {"Tensor": ["void* data"]},
        "functions": {"add_forward": {"return": "int", "args": ["const Tensor* A"]}},
    }
    output = tmp_path / "abi.json"

    write_manifest(output, baseline)
    assert json.loads(output.read_text(encoding="utf-8"))["functions"]["add_forward"]["return"] == "void"
    assert compare_manifests(baseline, changed) == ["ABI data_type changed", "ABI functions changed"]
