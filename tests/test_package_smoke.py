# /**
#   ******************************************************************************
#   * @file        test_package_smoke.py
#   * @author      Egor Izmaylov
#   * @brief       覆盖 wheel 安装 smoke test 的关键断言脚本。
#   * @details     2026.06.27  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import zipfile

import pytest

from tools.package_smoke import _assert_runtime_wheel, _smoke_script


def _write_wheel(path, *, purelib: bool = False, include_runtime: bool = True):
    with zipfile.ZipFile(path, "w") as archive:
        if include_runtime:
            archive.writestr("nn/tensor_ops.so", b"not a real shared object")
        archive.writestr(
            "onnx_translator-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: test\n"
            f"Root-Is-Purelib: {'true' if purelib else 'false'}\n"
            "Tag: py3-none-linux_x86_64\n",
        )


def test_package_smoke_script_checks_packaged_runtime_and_c_backend():
    script = _smoke_script()

    assert "ONNX_TRANSLATOR_PACKAGE_INSTALL_DIR" in script
    assert "runtime library was not loaded from installed package" in script
    assert "import unexpectedly resolved to repository source" in script
    assert "ADD(" in script
    assert "np.testing.assert_array_equal" in script


def test_assert_runtime_wheel_accepts_platform_specific_runtime_wheel(tmp_path):
    wheel = tmp_path / "onnx_translator-0.1.0-py3-none-linux_x86_64.whl"
    _write_wheel(wheel)

    _assert_runtime_wheel(wheel)


def test_assert_runtime_wheel_accepts_data_platlib_runtime_layout(tmp_path):
    wheel = tmp_path / "onnx_translator-0.1.0-cp312-cp312-linux_x86_64.whl"
    _write_wheel(wheel, include_runtime=False)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("onnx_translator-0.1.0.data/platlib/nn/tensor_ops.so", b"not a real shared object")

    _assert_runtime_wheel(wheel)


def test_assert_runtime_wheel_rejects_data_purelib_runtime_layout(tmp_path):
    wheel = tmp_path / "onnx_translator-0.1.0-cp312-cp312-linux_x86_64.whl"
    _write_wheel(wheel, include_runtime=False)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("onnx_translator-0.1.0.data/purelib/nn/tensor_ops.so", b"not a real shared object")

    with pytest.raises(RuntimeError, match="platlib, not purelib"):
        _assert_runtime_wheel(wheel)


def test_assert_runtime_wheel_rejects_pure_python_tag_for_runtime_wheel(tmp_path):
    wheel = tmp_path / "onnx_translator-0.1.0-py3-none-any.whl"
    _write_wheel(wheel)

    with pytest.raises(RuntimeError, match="platform-specific"):
        _assert_runtime_wheel(wheel)


def test_assert_runtime_wheel_rejects_missing_packaged_backend(tmp_path):
    wheel = tmp_path / "onnx_translator-0.1.0-py3-none-linux_x86_64.whl"
    _write_wheel(wheel, include_runtime=False)

    with pytest.raises(RuntimeError, match="missing packaged C backend"):
        _assert_runtime_wheel(wheel)


def test_assert_runtime_wheel_rejects_purelib_metadata(tmp_path):
    wheel = tmp_path / "onnx_translator-0.1.0-py3-none-linux_x86_64.whl"
    _write_wheel(wheel, purelib=True)

    with pytest.raises(RuntimeError, match="Root-Is-Purelib"):
        _assert_runtime_wheel(wheel)
