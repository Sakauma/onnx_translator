# /**
#   ******************************************************************************
#   * @file        test_wheelhouse_smoke.py
#   * @author      Egor Izmaylov
#   * @brief       Covers manylinux wheelhouse inspection for runtime wheels.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import zipfile

import pytest

from tools.wheelhouse_smoke import _wheel_tags, inspect_wheelhouse


def _write_runtime_wheel(path, *, include_runtime=True, root_is_pure=False):
    with zipfile.ZipFile(path, "w") as archive:
        if include_runtime:
            archive.writestr("onnx_translator-0.1.0.data/platlib/nn/tensor_ops.so", b"not a real shared object")
        archive.writestr(
            "onnx_translator-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: test\n"
            f"Root-Is-Purelib: {'true' if root_is_pure else 'false'}\n"
            "Tag: cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64\n",
        )


def test_wheel_tags_parse_python_abi_and_platform_tags():
    wheel = "onnx_translator-0.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

    assert _wheel_tags(type("Wheel", (), {"name": wheel})()) == (
        "cp312",
        "cp312",
        "manylinux_2_17_x86_64.manylinux2014_x86_64",
    )


def test_inspect_wheelhouse_accepts_manylinux_runtime_wheel(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _write_runtime_wheel(
        wheelhouse / "onnx_translator-0.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )

    inspected = inspect_wheelhouse(wheelhouse, required_python_tags=["cp312"], required_platform="manylinux")

    assert inspected[0]["python_tag"] == "cp312"


def test_inspect_wheelhouse_accepts_full_python_tag_matrix(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    for tag in ("cp310", "cp311", "cp312"):
        _write_runtime_wheel(
            wheelhouse
            / f"onnx_translator-0.1.0-{tag}-{tag}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        )

    inspected = inspect_wheelhouse(
        wheelhouse,
        required_python_tags=["cp310", "cp311", "cp312"],
        required_platform="manylinux",
    )

    assert sorted(item["python_tag"] for item in inspected) == ["cp310", "cp311", "cp312"]


def test_inspect_wheelhouse_rejects_missing_python_tag(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _write_runtime_wheel(
        wheelhouse / "onnx_translator-0.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )

    with pytest.raises(RuntimeError, match="missing required Python tags"):
        inspect_wheelhouse(wheelhouse, required_python_tags=["cp311"], required_platform="manylinux")


def test_inspect_wheelhouse_rejects_non_manylinux_platform(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _write_runtime_wheel(wheelhouse / "onnx_translator-0.1.0-cp312-cp312-linux_x86_64.whl")

    with pytest.raises(RuntimeError, match="platform tag"):
        inspect_wheelhouse(wheelhouse, required_python_tags=["cp312"], required_platform="manylinux")
