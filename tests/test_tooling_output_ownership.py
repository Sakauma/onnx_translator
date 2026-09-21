# /**
#   ******************************************************************************
#   * @file        test_tooling_output_ownership.py
#   * @author      Egor Izmaylov
#   * @brief       Covers CLI output containment and artifact ownership.
#   * @details     2026.09.21  V1.0.0  Created
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import onnx
import pytest
from onnx import TensorProto, helper

from tools import model_suite
from tools.commands import graph_logic, verify_graph
from nn import GraphVisualization


@pytest.mark.parametrize(
    "task_name",
    ["", ".", "..", "../victim", "nested/task", "nested\\task", "/absolute"],
)
def test_verify_graph_rejects_non_basename_task_names_before_output_changes(
    tmp_path, monkeypatch, task_name
):
    result_root = tmp_path / "result"
    result_root.mkdir()
    sentinel = tmp_path / "sentinel.keep"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(verify_graph, "RESULT_ROOT", result_root)
    monkeypatch.setattr(
        verify_graph.shutil,
        "rmtree",
        lambda *_args, **_kwargs: pytest.fail("invalid task name reached cleanup"),
    )

    with pytest.raises(ValueError, match="task name"):
        verify_graph.run_verification("missing.onnx", task_name)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert list(result_root.iterdir()) == []


def test_verify_graph_rejects_existing_symlink_that_escapes_result_root(tmp_path, monkeypatch):
    result_root = tmp_path / "result"
    outside = tmp_path / "outside"
    result_root.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.keep"
    sentinel.write_text("preserve", encoding="utf-8")
    (result_root / "escaped").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(verify_graph, "RESULT_ROOT", result_root)
    monkeypatch.setattr(
        verify_graph.shutil,
        "rmtree",
        lambda *_args, **_kwargs: pytest.fail("escaping symlink reached cleanup"),
    )

    with pytest.raises(ValueError, match="direct child"):
        verify_graph.run_verification("missing.onnx", "escaped")

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_verify_graph_rejects_in_root_symlink_alias_before_cleanup(tmp_path, monkeypatch):
    result_root = tmp_path / "result"
    victim = result_root / "victim"
    victim.mkdir(parents=True)
    sentinel = victim / "sentinel.keep"
    sentinel.write_text("preserve", encoding="utf-8")
    (result_root / "alias").symlink_to(victim, target_is_directory=True)
    monkeypatch.setattr(verify_graph, "RESULT_ROOT", result_root)
    monkeypatch.setattr(
        verify_graph.shutil,
        "rmtree",
        lambda *_args, **_kwargs: pytest.fail("in-root alias reached cleanup"),
    )

    with pytest.raises(ValueError, match="symbolic link or filesystem alias"):
        verify_graph.run_verification("missing.onnx", "alias")

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_verify_graph_cli_rejects_invalid_task_name_before_checking_model(tmp_path, monkeypatch, capsys):
    result_root = tmp_path / "result"
    result_root.mkdir()
    monkeypatch.setattr(verify_graph, "RESULT_ROOT", result_root)

    with pytest.raises(SystemExit) as caught:
        verify_graph.main(["--model", "missing.onnx", "--task-name", "../victim"])

    assert caught.value.code == 2
    assert "task name must be a non-empty directory basename" in capsys.readouterr().err
    assert list(result_root.iterdir()) == []


def test_verify_graph_passes_validated_result_dir_to_visualization_outside_repo(
    tmp_path, monkeypatch
):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "external_cwd",
        [x],
        [y],
    )
    model_path = tmp_path / "identity.onnx"
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        model_path,
    )
    result_root = tmp_path / "repository" / "result"
    external_cwd = tmp_path / "external"
    external_cwd.mkdir()
    captured = {}

    def capture(_graph, task_name, output_dir=None, raise_on_error=False):
        captured["task_name"] = task_name
        captured["output_dir"] = output_dir
        captured["raise_on_error"] = raise_on_error

    monkeypatch.setattr(verify_graph, "RESULT_ROOT", result_root)
    monkeypatch.setattr(verify_graph, "GraphGenerate", capture)
    monkeypatch.chdir(external_cwd)

    assert verify_graph.run_verification(str(model_path), "validated-task", clean=False) == 0
    assert captured == {
        "task_name": "validated-task",
        "output_dir": result_root / "validated-task",
        "raise_on_error": True,
    }
    assert not (external_cwd / "result").exists()


def test_graph_visualization_honors_explicit_output_dir_outside_repo(tmp_path, monkeypatch):
    output_dir = tmp_path / "repository" / "result" / "validated-task"
    external_cwd = tmp_path / "external"
    external_cwd.mkdir()
    captured = {}

    class FakeDigraph:
        def __init__(self, comment):
            captured["comment"] = comment

        def attr(self, *_args, **_kwargs):
            return None

        def node(self, *_args, **_kwargs):
            return None

        def edge(self, *_args, **_kwargs):
            return None

        def render(self, file_path, format, cleanup):
            captured["render"] = (file_path, format, cleanup)
            return f"{file_path}.{format}"

    monkeypatch.setattr(GraphVisualization, "Digraph", FakeDigraph)
    monkeypatch.chdir(external_cwd)

    GraphVisualization.GraphGenerate(
        SimpleNamespace(ops={}, input_name=[]),
        "validated-task",
        output_dir=output_dir,
    )

    assert output_dir.is_dir()
    assert captured["render"] == (
        str(output_dir / "validated-task_ops_graph"),
        "svg",
        True,
    )
    assert not (external_cwd / "result").exists()


def test_verify_graph_returns_nonzero_when_graphviz_render_fails(tmp_path, monkeypatch):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    model_path = tmp_path / "identity.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model_path)
    monkeypatch.setattr(verify_graph, "RESULT_ROOT", tmp_path / "result")

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("synthetic dot failure")

    monkeypatch.setattr(verify_graph, "GraphGenerate", fail_render)
    assert verify_graph.run_verification(str(model_path), "render-failure", clean=False) == 1


def test_graph_logic_rejects_traversal_and_accepts_legal_task_name(tmp_path, monkeypatch):
    result_root = tmp_path / "result"
    monkeypatch.setattr(verify_graph, "RESULT_ROOT", result_root)
    assert graph_logic.main("missing.onnx", "../escaped") == 2
    assert not (tmp_path / "escaped").exists()

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    model_path = tmp_path / "identity.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), model_path)
    captured = {}

    def capture(_graph, task_name, output_dir=None, raise_on_error=False):
        captured.update(
            task_name=task_name,
            output_dir=output_dir,
            raise_on_error=raise_on_error,
        )

    monkeypatch.setattr(graph_logic, "GraphGenerate", capture)
    assert graph_logic.main(str(model_path), "legal-task") == 0
    assert captured == {
        "task_name": "legal-task",
        "output_dir": result_root / "legal-task",
        "raise_on_error": True,
    }


def test_create_graph_model_cli_accepts_basename_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "tools" / "cli.py"), "create-graph-model", "--output", "model.onnx"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "model.onnx").is_file()


def _enable_model_suite(monkeypatch, tmp_path):
    backend = tmp_path / "tensor_ops.so"
    backend.touch()
    monkeypatch.setattr(model_suite.nn, "TENSOR_OPS_LIB_PATH", str(backend))


def _write_reserved_models(output_dir: Path) -> None:
    for spec in model_suite.MODEL_SPECS:
        path = output_dir / f"{spec.name}.onnx"
        assert path.exists()
        path.write_text(f"generated {spec.name}", encoding="utf-8")


def test_model_smoke_rejects_nonempty_output_without_touching_existing_files(
    tmp_path, monkeypatch, capsys
):
    _enable_model_suite(monkeypatch, tmp_path)
    output_dir = tmp_path / "models"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.keep"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        model_suite,
        "verify_model_suite",
        lambda *_args, **_kwargs: pytest.fail("nonempty output reached model generation"),
    )

    assert model_suite.main(["--output-dir", str(output_dir)]) == 2

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert "must be new or empty" in capsys.readouterr().err


def test_model_smoke_reservation_collision_rolls_back_only_new_reservations(tmp_path):
    output_dir = tmp_path / "models"
    output_dir.mkdir()
    collision = output_dir / f"{model_suite.MODEL_SPECS[1].name}.onnx"
    collision.write_text("pre-existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        model_suite._reserve_output_files(output_dir)

    assert not (output_dir / f"{model_suite.MODEL_SPECS[0].name}.onnx").exists()
    assert collision.read_text(encoding="utf-8") == "pre-existing"


def test_model_smoke_cleans_only_owned_files_and_preserves_existing_empty_directory(
    tmp_path, monkeypatch
):
    _enable_model_suite(monkeypatch, tmp_path)
    output_dir = tmp_path / "models"
    output_dir.mkdir()

    def verify(current_output_dir, check_numeric=True):
        assert check_numeric
        _write_reserved_models(current_output_dir)
        (current_output_dir / "created-externally.keep").write_text("preserve", encoding="utf-8")
        return []

    monkeypatch.setattr(model_suite, "verify_model_suite", verify)

    assert model_suite.main(["--output-dir", str(output_dir)]) == 0

    assert output_dir.is_dir()
    assert (output_dir / "created-externally.keep").read_text(encoding="utf-8") == "preserve"
    assert not any((output_dir / f"{spec.name}.onnx").exists() for spec in model_suite.MODEL_SPECS)


def test_model_smoke_removes_new_empty_output_directory_after_success(tmp_path, monkeypatch):
    _enable_model_suite(monkeypatch, tmp_path)
    output_dir = tmp_path / "models"

    def verify(current_output_dir, check_numeric=True):
        _write_reserved_models(current_output_dir)
        return []

    monkeypatch.setattr(model_suite, "verify_model_suite", verify)

    assert model_suite.main(["--output-dir", str(output_dir)]) == 0
    assert not output_dir.exists()


def test_model_smoke_failure_cleans_owned_files_but_preserves_external_file(
    tmp_path, monkeypatch
):
    _enable_model_suite(monkeypatch, tmp_path)
    output_dir = tmp_path / "models"

    def fail(current_output_dir, check_numeric=True):
        _write_reserved_models(current_output_dir)
        (current_output_dir / "created-externally.keep").write_text("preserve", encoding="utf-8")
        raise RuntimeError("synthetic verification failure")

    monkeypatch.setattr(model_suite, "verify_model_suite", fail)

    with pytest.raises(RuntimeError, match="synthetic verification failure"):
        model_suite.main(["--output-dir", str(output_dir)])

    assert (output_dir / "created-externally.keep").read_text(encoding="utf-8") == "preserve"
    assert not any((output_dir / f"{spec.name}.onnx").exists() for spec in model_suite.MODEL_SPECS)


def test_model_smoke_failure_removes_new_directory_when_no_external_files(
    tmp_path, monkeypatch
):
    _enable_model_suite(monkeypatch, tmp_path)
    output_dir = tmp_path / "models"

    def fail(current_output_dir, check_numeric=True):
        (current_output_dir / f"{model_suite.MODEL_SPECS[0].name}.onnx").write_text(
            "partial generated model", encoding="utf-8"
        )
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(model_suite, "verify_model_suite", fail)

    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        model_suite.main(["--output-dir", str(output_dir)])

    assert not output_dir.exists()


def test_model_smoke_keep_artifacts_retains_owned_models(tmp_path, monkeypatch):
    _enable_model_suite(monkeypatch, tmp_path)
    output_dir = tmp_path / "models"

    def verify(current_output_dir, check_numeric=True):
        _write_reserved_models(current_output_dir)
        return []

    monkeypatch.setattr(model_suite, "verify_model_suite", verify)

    assert model_suite.main(["--output-dir", str(output_dir), "--keep-artifacts"]) == 0
    assert all((output_dir / f"{spec.name}.onnx").is_file() for spec in model_suite.MODEL_SPECS)


def test_model_smoke_keep_artifacts_drops_unwritten_reservations_after_failure(
    tmp_path, monkeypatch
):
    _enable_model_suite(monkeypatch, tmp_path)
    output_dir = tmp_path / "models"
    generated_path = output_dir / f"{model_suite.MODEL_SPECS[0].name}.onnx"

    def fail(current_output_dir, check_numeric=True):
        generated_path.write_text("partial generated model", encoding="utf-8")
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(model_suite, "verify_model_suite", fail)

    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        model_suite.main(["--output-dir", str(output_dir), "--keep-artifacts"])

    assert generated_path.read_text(encoding="utf-8") == "partial generated model"
    assert not any(
        (output_dir / f"{spec.name}.onnx").exists() for spec in model_suite.MODEL_SPECS[1:]
    )
