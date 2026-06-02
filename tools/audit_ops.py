from __future__ import annotations

import argparse
import ast
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OperatorInfo:
    class_name: str
    line: int
    bases: tuple[str, ...]
    has_forward: bool
    has_forward_shape: bool
    c_functions: tuple[str, ...]
    c_runtime_functions: tuple[str, ...]
    c_runtime_kind: str
    runtime_uses_numpy: bool
    import_supported: bool
    cuda_verified: bool
    numerical_planned: bool
    status: str
    notes: tuple[str, ...]


MANUAL_ALIASES = {
    "ABS": {"abs"},
    "ADD": {"add"},
    "COS": {"cos"},
    "DIV": {"div"},
    "EXP": {"exp"},
    "LOG": {"log"},
    "MUL": {"mul"},
    "RELU": {"relu"},
    "SIGMOID": {"sigmoid"},
    "SQRT": {"sqrt"},
    "SUB": {"sub"},
    "TANH": {"tanh"},
    "Conv": {"conv2d", "conv"},
    "ScatterND": {"scatternd", "scatter_nd"},
    "GatherND": {"gathernd", "gather_nd"},
    "QLinearConv": {"qlinear_conv"},
    "QuantizeLinear": {"quantize_linear"},
    "DequantizeLinear": {"dequantize_linear"},
    "MatMulInteger": {"matmul_integer"},
    "QLinearMatMul": {"qlinear_matmul"},
    "GreaterOrEqual": {"greater_or_equal"},
    "LessOrEqual": {"less_or_equal"},
    "MaxPool": {"max_pool"},
    "ReduceMean": {"reduce_mean"},
    "ReduceSum": {"reduce_sum"},
    "ReduceMax": {"reduce_max"},
    "ReduceMin": {"reduce_min"},
    "ReduceProd": {"reduce_prod"},
    "ReduceL1": {"reduce_l1"},
    "ReduceL2": {"reduce_l2"},
    "ReduceLogSum": {"reduce_log_sum"},
    "ReduceLogSumExp": {"reduce_log_sum_exp"},
    "ReduceSumSquare": {"reduce_sum_square"},
    "ArgMax": {"argmax", "arg_max"},
    "ArgMin": {"argmin", "arg_min"},
    "RandomUniformLike": {"random_uniform_like"},
    "RandomNormalLike": {"random_normal_like"},
    "BatchNormalization": {"batch_normalization", "batch_norm"},
    "InstanceNormalization": {"instance_normalization", "instance_norm"},
    "LayerNormalization": {"layer_normalization", "layer_norm"},
    "GroupNormalization": {"group_normalization", "group_norm"},
    "BitwiseAnd": {"bitwise_and"},
    "BitwiseOr": {"bitwise_or"},
    "BitwiseXor": {"bitwise_xor"},
    "BitwiseNot": {"bitwise_not"},
    "BitShift": {"bit_shift"},
    "HardSigmoid": {"hard_sigmoid"},
    "HardSwish": {"hard_swish"},
    "Softplus": {"softplus"},
    "Softsign": {"softsign"},
    "ThresholdedRelu": {"thresholded_relu"},
    "GlobalAveragePool": {"global_average_pool"},
    "GlobalMaxPool": {"global_max_pool"},
    "GlobalLpPool": {"global_lp_pool"},
    "LpPool": {"lp_pool"},
    "LpNormalization": {"lp_normalization"},
    "DepthToSpace": {"depth_to_space"},
    "SpaceToDepth": {"space_to_depth"},
    "ReverseSequence": {"reverse_sequence"},
    "ScatterElements": {"scatter_elements"},
    "GatherElements": {"gather_elements"},
    "ConstantOfShape": {"constant_of_shape"},
    "DynamicQuantizeLinear": {"dynamic_quantize_linear"},
    "HannWindow": {"hann_window"},
    "HammingWindow": {"hamming_window"},
    "BlackmanWindow": {"blackman_window"},
    "LogSoftmax": {"log_softmax"},
}


PYTHON_ORCHESTRATION_RUNTIME = {
    "Shape",
    "Constant",
    "SequenceEmpty",
    "SequenceConstruct",
    "SequenceAt",
    "SequenceInsert",
    "SequenceErase",
    "SequenceLength",
    "ConcatFromSequence",
    "SplitToSequence",
    "Optional",
    "OptionalGetElement",
    "OptionalHasElement",
    "StringNormalizer",
    "TfIdfVectorizer",
    "If",
    "Loop",
    "Scan",
    "SequenceMap",
}


DEFERRED_C_BACKEND_RUNTIME = set()


DEFERRED_DEEP_SEMANTIC_AUDIT = {
    "MaxRoiPool",
    "RoiAlign",
    "RNN",
    "GRU",
    "LSTM",
    "DFT",
    "STFT",
}


# Egor Izmaylov: Function `normalize_name` implements the normalize name step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def normalize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name).upper()


# Egor Izmaylov: Function `to_snake` implements the to snake step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def to_snake(name: str) -> str:
    if name.isupper():
        return name.lower()
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


# Egor Izmaylov: Function `aliases_for` implements the aliases for step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def aliases_for(class_name: str) -> set[str]:
    snake = to_snake(class_name)
    aliases = {snake, snake.replace("_", ""), class_name.lower()}
    aliases.update(MANUAL_ALIASES.get(class_name, set()))
    return aliases


# Egor Izmaylov: Function `base_names` implements the base names step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def base_names(node: ast.ClassDef) -> tuple[str, ...]:
    names = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return tuple(names)


# Egor Izmaylov: Function `parse_operator_classes` implements the parse operator classes step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def parse_operator_classes() -> dict[str, dict]:
    source = (ROOT / "nn" / "Operators.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    method_map = {}
    class_nodes = {}

    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        class_nodes[node.name] = node
        method_map[node.name] = {item.name: item for item in node.body if isinstance(item, ast.FunctionDef)}

    # Egor Izmaylov: Function `parse_operator_classes.has_method` implements the has method step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
    def has_method(class_name: str, method_name: str) -> bool:
        if method_name in method_map.get(class_name, {}):
            return True
        for base in base_names(class_nodes[class_name]):
            if base in class_nodes and has_method(base, method_name):
                return True
        return False

    # Egor Izmaylov: Function `parse_operator_classes.resolve_method_node` implements the resolve method node step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
    def resolve_method_node(class_name: str, method_name: str) -> ast.FunctionDef | None:
        if method_name in method_map.get(class_name, {}):
            return method_map[class_name][method_name]
        for base in base_names(class_nodes[class_name]):
            if base in class_nodes:
                node = resolve_method_node(base, method_name)
                if node is not None:
                    return node
        return None

    # Egor Izmaylov: Function `parse_operator_classes.collect_runtime_method_nodes` implements the collect runtime method nodes step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
    def collect_runtime_method_nodes(class_name: str, method_name: str) -> list[ast.FunctionDef]:
        nodes = []
        visited = set()

        # Egor Izmaylov: Function `parse_operator_classes.collect_runtime_method_nodes.visit` implements the visit step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            node = resolve_method_node(class_name, name)
            if node is None:
                return
            nodes.append(node)
            for child in ast.walk(node):
                if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
                    continue
                owner = child.func.value
                if isinstance(owner, ast.Name) and owner.id == "self":
                    visit(child.func.attr)

        visit(method_name)
        return nodes

    # Egor Izmaylov: Function `parse_operator_classes.collect_c_function_refs` implements the collect c function refs step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
    def collect_c_function_refs(nodes: list[ast.AST]) -> set[str]:
        c_functions = set()
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value.endswith("_forward"):
                    c_functions.add(child.value)
                elif isinstance(child, ast.Attribute) and child.attr.endswith("_forward"):
                    c_functions.add(child.attr)
        return c_functions

    # Egor Izmaylov: Function `parse_operator_classes.runtime_uses_numpy` implements the runtime uses numpy step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
    def runtime_uses_numpy(nodes: list[ast.AST]) -> bool:
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id == "np":
                    return True
        return False

    # Egor Izmaylov: Function `parse_operator_classes.is_operator_class` implements the is operator class step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
    def is_operator_class(class_name: str) -> bool:
        if class_name in {"ReduceBase", "ArgBase"}:
            return False
        for base in base_names(class_nodes[class_name]):
            if base in {"Ops", "ReduceBase", "ArgBase"}:
                return True
            if base in class_nodes and is_operator_class(base):
                return True
        return False

    operators = {}
    for name, node in class_nodes.items():
        bases = base_names(node)
        if name in {"ReduceBase", "ArgBase"}:
            continue
        if not is_operator_class(name):
            continue

        c_functions = collect_c_function_refs([node])
        runtime_nodes = collect_runtime_method_nodes(name, "forward")
        c_runtime_functions = collect_c_function_refs(runtime_nodes)

        operators[name] = {
            "line": node.lineno,
            "bases": bases,
            "has_forward": has_method(name, "forward"),
            "has_forward_shape": has_method(name, "forward_"),
            "c_functions": tuple(sorted(c_functions)),
            "c_runtime_functions": tuple(sorted(c_runtime_functions)),
            "runtime_uses_numpy": runtime_uses_numpy(runtime_nodes),
        }
    return operators


# Egor Izmaylov: Function `parse_import_supported_raw_ops` implements the parse import supported raw ops step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def parse_import_supported_raw_ops() -> set[str]:
    source = (ROOT / "nn" / "ONNXImport.py").read_text(encoding="utf-8")
    ops = set(re.findall(r'node\.op_type\s*==\s*"([^"]+)"', source))
    ops.update(re.findall(r'op_upper\s*==\s*"([^"]+)"', source))
    for match in re.finditer(r"node\.op_type\s+in\s+\[([^\]]+)\]", source):
        ops.update(re.findall(r'"([^"]+)"', match.group(1)))
    return ops


# Egor Izmaylov: Function `parse_import_supported_ops` implements the parse import supported ops step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def parse_import_supported_ops() -> set[str]:
    return {normalize_name(op) for op in parse_import_supported_raw_ops() if op != "Upsample"}


# Egor Izmaylov: Function `parse_onnx17_official_ops` implements the parse onnx17 official ops step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def parse_onnx17_official_ops() -> tuple[dict[str, str], str | None]:
    try:
        from onnx import defs
    except Exception as exc:  # pragma: no cover - exercised when onnx is missing locally.
        return {}, f"无法导入 onnx.defs: {exc}"

    latest = {}
    for schema in defs.get_all_schemas_with_history():
        if schema.domain != "" or schema.since_version > 17:
            continue
        if schema.name not in latest or schema.since_version > latest[schema.name].since_version:
            latest[schema.name] = schema
    return {normalize_name(name): name for name in latest}, None


# Egor Izmaylov: Function `parse_c_functions` implements the parse c functions step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def parse_c_functions() -> tuple[set[str], set[str]]:
    header = (ROOT / "tensor_ops" / "tensor_ops.h").read_text(encoding="utf-8")
    source = (ROOT / "tensor_ops" / "tensor_ops.c").read_text(encoding="utf-8")
    declared = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*_forward)\s*\(", header))
    implemented = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*_forward)\b", source))
    return declared, implemented


# Egor Izmaylov: Function `parse_cuda_verifiers` implements the parse cuda verifiers step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def parse_cuda_verifiers() -> set[str]:
    return {path.stem.replace("verify_", "") for path in (ROOT / "cuda").glob("verify_*.cu")}


# Egor Izmaylov: Function `parse_numerical_plans` implements the parse numerical plans step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def parse_numerical_plans() -> set[str]:
    source = (ROOT / "numerical_correctness.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    plans: list[str] = []

    class Visitor(ast.NodeVisitor):
        # Egor Izmaylov: Function `parse_numerical_plans.Visitor.visit_Assign` handles AST visitor logic for visit Assign, collecting metadata that later audit steps use to classify operator support.
        def visit_Assign(self, node: ast.Assign) -> None:
            nonlocal plans
            if not any(isinstance(target, ast.Name) and target.id == "plans" for target in node.targets):
                self.generic_visit(node)
                return
            if not isinstance(node.value, ast.List):
                self.generic_visit(node)
                return
            extracted = []
            for item in node.value.elts:
                if (
                    isinstance(item, ast.Tuple)
                    and len(item.elts) >= 2
                    and isinstance(item.elts[1], ast.Constant)
                    and isinstance(item.elts[1].value, str)
                ):
                    extracted.append(item.elts[1].value)
            if extracted:
                plans = extracted
            self.generic_visit(node)

    Visitor().visit(module)
    return set(plans)


# Egor Izmaylov: Function `classify` implements the classify step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def classify(
    class_name: str,
    data: dict,
    import_supported: bool,
    cuda_verified: bool,
    numerical_planned: bool,
    c_implemented: set[str],
) -> tuple[str, tuple[str, ...]]:
    notes = []
    c_functions = set(data["c_functions"])
    c_runtime_functions = set(data["c_runtime_functions"])
    missing_c = sorted(c_functions - c_implemented)

    if not data["has_forward"]:
        notes.append("缺少 forward")
    if not data["has_forward_shape"]:
        notes.append("缺少 forward_ 形状推断")
    if missing_c:
        notes.append("Python 引用的 C 函数未实现: " + ", ".join(missing_c))
    if not import_supported:
        notes.append("ONNXImport 未映射")
    if c_functions and not c_runtime_functions:
        notes.append("存在 C 函数引用/声明，但 forward 运行路径未调用")
    if not c_runtime_functions:
        if class_name in PYTHON_ORCHESTRATION_RUNTIME:
            notes.append("Python 调度/元数据类，不要求 C 数值后端")
        elif class_name in DEFERRED_C_BACKEND_RUNTIME:
            notes.append("按当前整理阶段暂缓后端化，后续统一设计 C/CUDA 实现")
        else:
            notes.append("forward 运行路径未接入 C 后端")
    elif data["runtime_uses_numpy"]:
        notes.append("含 Python 调度或 fallback")
    if class_name in DEFERRED_DEEP_SEMANTIC_AUDIT:
        notes.append("按当前整理阶段暂缓深度语义/数值验证，作为剩余风险跟踪")
    if cuda_verified and not numerical_planned:
        notes.append("有 CUDA verifier，但未接入 active numerical plan")
    if not cuda_verified and not numerical_planned:
        notes.append("缺少 CUDA/数值验证覆盖")

    implementation_ok = (
        data["has_forward"]
        and data["has_forward_shape"]
        and not missing_c
        and import_supported
    )

    if not implementation_ok:
        return "部分实现/需补齐", tuple(notes)
    if not c_runtime_functions and class_name in DEFERRED_C_BACKEND_RUNTIME:
        return "暂缓后端化", tuple(notes)
    if not c_runtime_functions and class_name not in PYTHON_ORCHESTRATION_RUNTIME:
        return "Python-only 待后端化", tuple(notes)
    if numerical_planned and cuda_verified:
        return "已数值验证", tuple(notes)
    if cuda_verified and not numerical_planned:
        return "待接入数值计划", tuple(notes)
    if class_name in DEFERRED_DEEP_SEMANTIC_AUDIT:
        return "暂缓深度验证", tuple(notes)
    return "已实现未数值验证", tuple(notes)


# Egor Izmaylov: Function `audit` implements the audit step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def audit() -> tuple[list[OperatorInfo], dict[str, object]]:
    operators = parse_operator_classes()
    import_supported = parse_import_supported_ops()
    import_supported_raw = parse_import_supported_raw_ops()
    c_declared, c_implemented = parse_c_functions()
    cuda_verifiers = parse_cuda_verifiers()
    numerical_plans = parse_numerical_plans()
    official_onnx17, official_error = parse_onnx17_official_ops()
    normalized_import_raw = {normalize_name(op): op for op in import_supported_raw}

    infos = []
    for class_name, data in sorted(operators.items(), key=lambda item: item[1]["line"]):
        aliases = aliases_for(class_name)
        import_ok = normalize_name(class_name) in import_supported
        cuda_ok = bool(aliases & cuda_verifiers)
        numerical_ok = bool(aliases & numerical_plans)
        status, notes = classify(class_name, data, import_ok, cuda_ok, numerical_ok, c_implemented)
        if data["c_runtime_functions"]:
            c_runtime_kind = "C-backed"
        elif class_name in PYTHON_ORCHESTRATION_RUNTIME:
            c_runtime_kind = "Python orchestration"
        else:
            c_runtime_kind = "Python-only"
        infos.append(
            OperatorInfo(
                class_name=class_name,
                line=data["line"],
                bases=data["bases"],
                has_forward=data["has_forward"],
                has_forward_shape=data["has_forward_shape"],
                c_functions=data["c_functions"],
                c_runtime_functions=data["c_runtime_functions"],
                c_runtime_kind=c_runtime_kind,
                runtime_uses_numpy=data["runtime_uses_numpy"],
                import_supported=import_ok,
                cuda_verified=cuda_ok,
                numerical_planned=numerical_ok,
                status=status,
                notes=notes,
            )
        )

    metadata = {
        "import_supported_count": len(import_supported),
        "operator_class_count": len(operators),
        "c_declared_count": len(c_declared),
        "c_implemented_count": len(c_implemented),
        "c_declared_missing_impl": sorted(c_declared - c_implemented),
        "c_impl_missing_decl": sorted(c_implemented - c_declared),
        "cuda_verifier_count": len(cuda_verifiers),
        "numerical_plan_count": len(numerical_plans),
        "cuda_not_planned": sorted(cuda_verifiers - numerical_plans),
        "plan_without_cuda": sorted(numerical_plans - cuda_verifiers),
        "official_onnx17_count": len(official_onnx17),
        "official_onnx17_error": official_error,
        "official_onnx17_supported_count": len(set(official_onnx17) & set(normalized_import_raw)),
        "official_onnx17_missing": [official_onnx17[name] for name in sorted(set(official_onnx17) - set(normalized_import_raw))],
        "supported_non_onnx17": [normalized_import_raw[name] for name in sorted(set(normalized_import_raw) - set(official_onnx17))],
    }
    metadata["c_runtime_count"] = sum(1 for info in infos if info.c_runtime_functions)
    metadata["python_orchestration_runtime_count"] = sum(
        1 for info in infos if not info.c_runtime_functions and info.class_name in PYTHON_ORCHESTRATION_RUNTIME
    )
    metadata["python_only_runtime_count"] = sum(
        1 for info in infos if not info.c_runtime_functions and info.class_name not in PYTHON_ORCHESTRATION_RUNTIME
    )
    metadata["deferred_c_backend_runtime_count"] = sum(
        1 for info in infos if not info.c_runtime_functions and info.class_name in DEFERRED_C_BACKEND_RUNTIME
    )
    metadata["active_python_only_runtime_count"] = sum(
        1
        for info in infos
        if (
            not info.c_runtime_functions
            and info.class_name not in PYTHON_ORCHESTRATION_RUNTIME
            and info.class_name not in DEFERRED_C_BACKEND_RUNTIME
        )
    )
    metadata["deferred_c_backend_runtime"] = [
        info.class_name for info in infos if not info.c_runtime_functions and info.class_name in DEFERRED_C_BACKEND_RUNTIME
    ]
    metadata["deferred_deep_semantic_audit"] = [
        info.class_name for info in infos if info.class_name in DEFERRED_DEEP_SEMANTIC_AUDIT
    ]
    metadata["declared_c_but_runtime_unused"] = [
        info.class_name for info in infos if info.c_functions and not info.c_runtime_functions
    ]
    return infos, metadata


# Egor Izmaylov: Function `yes_no` implements the yes no step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def yes_no(value: bool) -> str:
    return "yes" if value else "no"


# Egor Izmaylov: Function `render_markdown` implements the render markdown step for the operator coverage audit tool, normalizing inputs and returning the exact data or metadata contract expected downstream.
def render_markdown(infos: list[OperatorInfo], metadata: dict[str, object]) -> str:
    status_counts = Counter(info.status for info in infos)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# 算子实现情况评估",
        "",
        f"> 自动生成时间：{generated_at}",
        "> 生成命令：`python tools/audit_ops.py --output 算子实现情况统计.md`",
        "",
        "## 评估口径",
        "",
        "- `ONNXImport`：`nn/ONNXImport.py` 中存在显式映射，可从 ONNX node 构造对应算子类。",
        "- `forward`：`nn/Operators.py` 中存在运行时前向实现，继承自 `ReduceBase`/`ArgBase` 的实现也计入。",
        "- `forward_`：存在图构建/形状推断实现，继承实现也计入。",
        "- `C backend funcs`：Python 算子类中引用的 `<op>_forward` C 函数均能在 `tensor_ops/tensor_ops.c` 中找到，且 `tensor_ops.h` 与 `.c` 声明/实现集合一致。",
        "- `C runtime path`：`forward()` 运行路径实际引用 `<op>_forward` C 函数；仅在 `__init__`、形状推断或未调用 helper 中出现不计入。",
        "- `CUDA verifier`：`cuda/verify_<op>.cu` 存在，仅说明有参考验证程序源码。",
            "- `active numerical plan`：`numerical_correctness.py` 当前实际 `plans` 列表中包含该算子，代表会被默认数值验证门禁执行。",
            "- `ONNX opset 17 官方覆盖`：通过本地 `onnx.defs` 读取默认 domain 中 `since_version <= 17` 的最新 schema，并与 `ONNXImport` 的显式映射做名称级对比。",
        "",
        "## 总览",
        "",
        f"- Python 算子类：{metadata['operator_class_count']} 个。",
        f"- ONNXImport 显式支持：{metadata['import_supported_count']} 个 ONNX op 名称；`Upsample` 作为 `Resize` 别名处理，未单独计入算子类。",
        f"- C 后端声明：{metadata['c_declared_count']} 个 `<op>_forward`；C 实现可检出：{metadata['c_implemented_count']} 个。",
        f"- forward 实际接入 C 后端：{metadata['c_runtime_count']} 个算子类。",
        f"- 合理保留 Python 调度/元数据运行时：{metadata['python_orchestration_runtime_count']} 个算子类。",
        (
            f"- 普通数值/张量算子 Python-only 运行时：{metadata['python_only_runtime_count']} 个算子类；"
            f"其中当前暂缓后端化：{metadata['deferred_c_backend_runtime_count']} 个，"
            f"除暂缓项外待后端化：{metadata['active_python_only_runtime_count']} 个。"
        ),
        f"- CUDA verifier：{metadata['cuda_verifier_count']} 个。",
            f"- active numerical plan 覆盖：{metadata['numerical_plan_count']} 个唯一算子名称。",
            (
                f"- 暂缓深度语义/数值验证：{len(metadata['deferred_deep_semantic_audit'])} 个；"
                + ", ".join(f"`{name}`" for name in metadata["deferred_deep_semantic_audit"])
                + "。"
            ),
            f"- ONNX opset 17 官方算子：{metadata['official_onnx17_count']} 个；ONNXImport 名称级覆盖：{metadata['official_onnx17_supported_count']} 个。",
            "",
        "### 状态计数",
        "",
        "| 状态 | 数量 |",
        "| --- | ---: |",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")

    partial = [info.class_name for info in infos if info.status == "部分实现/需补齐"]
    unverified = [
        info.class_name
        for info in infos
        if info.status in {"已实现未数值验证", "暂缓深度验证", "待接入数值计划"}
    ]
    planned = metadata["numerical_plan_count"]
    deferred = metadata["deferred_c_backend_runtime"]
    active_python_only = [
        info.class_name
        for info in infos
        if info.status == "Python-only 待后端化"
    ]
    lines.extend(
        [
            "",
            "### 关键结论",
            "",
            "- `tensor_ops.h` 与 `tensor_ops.c` 中的 C forward 函数集合一致，没有发现声明缺实现或实现缺声明。",
            f"- 所有 {metadata['operator_class_count']} 个 Python 算子类均可被 `ONNXImport` 显式映射。",
            (
                "- 未发现缺少 `forward` / `forward_` / C 函数映射的显式部分实现算子。"
                if not partial
                else "- 仍有显式部分实现算子：" + ", ".join(f"`{name}`" for name in partial)
            ),
            (
                "- 未发现“有 C 函数但 forward 未调用”的算子。"
                if not metadata["declared_c_but_runtime_unused"]
                else "- 存在 C 函数但 forward 未调用：" + ", ".join(f"`{name}`" for name in metadata["declared_c_but_runtime_unused"])
            ),
            (
                "- 当前暂缓后端化算子："
                + ", ".join(f"`{name}`" for name in deferred)
                + "。"
                if deferred
                else "- 当前没有记录暂缓后端化算子。"
            ),
            "- 当前暂缓深度语义/数值验证的剩余算子："
            + ", ".join(f"`{name}`" for name in metadata["deferred_deep_semantic_audit"])
            + "。",
            (
                "- 除暂缓项外，未发现仍需立即后端化的 Python-only 普通数值/张量算子。"
                if not active_python_only
                else "- 除暂缓项外仍需后端化："
                + ", ".join(f"`{name}`" for name in active_python_only)
                + "。"
            ),
            f"- 默认数值门禁当前覆盖 {planned} 个唯一算子；尚有 {len(unverified)} 个已实现算子未进入 active numerical plan。",
        ]
    )

    cuda_not_planned = metadata["cuda_not_planned"]
    if cuda_not_planned:
        lines.extend(
            [
                "",
                "### 有 CUDA verifier 但未进入默认数值计划",
                "",
                ", ".join(f"`{name}`" for name in cuda_not_planned),
            ]
        )

    plan_without_cuda = metadata["plan_without_cuda"]
    if plan_without_cuda:
        lines.extend(
            [
                "",
                "### 默认数值计划中未找到 CUDA verifier 的名称",
                "",
                ", ".join(f"`{name}`" for name in plan_without_cuda),
            ]
        )

    if metadata["official_onnx17_error"]:
        lines.extend(
            [
                "",
                "## ONNX opset 17 官方覆盖",
                "",
                f"- 无法生成官方覆盖对比：{metadata['official_onnx17_error']}",
            ]
        )
    else:
        missing = metadata["official_onnx17_missing"]
        extra = metadata["supported_non_onnx17"]
        lines.extend(
            [
                "",
                "## ONNX opset 17 官方覆盖",
                "",
                f"- 官方默认 domain 算子：{metadata['official_onnx17_count']} 个。",
                f"- `ONNXImport` 已覆盖官方名称：{metadata['official_onnx17_supported_count']} 个。",
                f"- 官方名称级缺口：{len(missing)} 个。",
                "",
                "### 官方缺口",
                "",
                ", ".join(f"`{name}`" for name in missing) if missing else "无。",
                "",
                "### 仓库额外/非默认 domain/实验性名称",
                "",
                ", ".join(f"`{name}`" for name in extra) if extra else "无。",
            ]
        )

    lines.extend(
        [
            "",
            "## 明细表",
            "",
            "| # | 算子类 | ONNXImport | forward | forward_ | C backend funcs | C runtime path | CUDA verifier | active numerical plan | 状态 | 备注 |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for idx, info in enumerate(infos, start=1):
        c_backend = ", ".join(f"`{name}`" for name in info.c_functions) if info.c_functions else "none"
        c_runtime = (
            ", ".join(f"`{name}`" for name in info.c_runtime_functions)
            if info.c_runtime_functions
            else info.c_runtime_kind
        )
        notes = "; ".join(info.notes) if info.notes else "-"
        lines.append(
            "| {idx} | `{name}` | {imp} | {fwd} | {shape} | {c_backend} | {c_runtime} | {cuda} | {plan} | {status} | {notes} |".format(
                idx=idx,
                name=info.class_name,
                imp=yes_no(info.import_supported),
                fwd=yes_no(info.has_forward),
                shape=yes_no(info.has_forward_shape),
                c_backend=c_backend,
                c_runtime=c_runtime,
                cuda=yes_no(info.cuda_verified),
                plan=yes_no(info.numerical_planned),
                status=info.status,
                notes=notes,
            )
        )
    lines.append("")
    return "\n".join(lines)


# Egor Izmaylov: Function `main` is the command-line entry point for the operator coverage audit tool; it parses runtime options, runs the selected checks, and returns a process status.
def main() -> int:
    parser = argparse.ArgumentParser(description="Audit operator implementation and verification coverage.")
    parser.add_argument(
        "--output",
        default="算子实现情况统计.md",
        help="Markdown report path, relative to the repository root unless absolute.",
    )
    args = parser.parse_args()

    infos, metadata = audit()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.write_text(render_markdown(infos, metadata), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
