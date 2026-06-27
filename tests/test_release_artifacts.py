# /**
#   ******************************************************************************
#   * @file        test_release_artifacts.py
#   * @author      Egor Izmaylov
#   * @brief       Covers release artifact smoke helper checks.
#   * @details     2026.06.27  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

import io
import tarfile

import pytest

from tools.release_artifacts import REQUIRED_SDIST_PATHS, _strip_sdist_root, inspect_sdist


def _write_sdist(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=f"onnx_translator-0.1.0/{name}")
            info.size = len(data)
            archive.addfile(info, fileobj=io.BytesIO(data))


def test_strip_sdist_root_removes_generated_top_level_directory():
    assert _strip_sdist_root("onnx_translator-0.1.0/tensor_ops/tensor_ops.h") == "tensor_ops/tensor_ops.h"
    assert _strip_sdist_root("onnx_translator-0.1.0") == ""


def test_inspect_sdist_accepts_required_release_sources(tmp_path):
    sdist = tmp_path / "package.tar.gz"
    members = {name: "x" for name in REQUIRED_SDIST_PATHS}
    members["tensor_ops/tensor_ops_core.c"] = "int x;"
    members["cuda/verify_mul.cu"] = "int main(){}"
    _write_sdist(sdist, members)

    names = inspect_sdist(sdist)

    assert "tensor_ops/tensor_ops.h" in names
    assert "cuda/verify_add.cu" in names


def test_inspect_sdist_rejects_missing_c_sources(tmp_path):
    sdist = tmp_path / "package.tar.gz"
    members = {name: "x" for name in REQUIRED_SDIST_PATHS if not name.endswith(".c")}
    _write_sdist(sdist, members)

    with pytest.raises(RuntimeError, match="tensor_ops C sources"):
        inspect_sdist(sdist)
