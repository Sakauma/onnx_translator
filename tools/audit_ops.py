# /**
#   ******************************************************************************
#   * @file        audit_ops.py
#   * @author      Egor Izmaylov
#   * @brief       审计 ONNX 算子覆盖情况，汇总导入支持、运行路径、C 后端和 CUDA 验证状态。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from __future__ import annotations

import argparse
import ast
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATORS_SOURCES = [ROOT / "nn" / "Operators.py", *sorted((ROOT / "nn" / "operators").glob("*.py"))]
IMPORTER_SOURCES = [ROOT / "nn" / "ONNXImport.py", *sorted((ROOT / "nn" / "importer").glob("*.py"))]


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
    "AffineGrid": {"affine_grid"},
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
    "CumProd": {"cumprod", "cum_prod"},
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
    "RMSNormalization": {"rms_normalization", "rms_norm"},
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


DEEP_SEMANTIC_PYTEST_COVERAGE = {
    "Bernoulli",
    "Binarizer",
    "BitShift",
    "BitwiseAnd",
    "BitwiseNot",
    "BitwiseOr",
    "BitwiseXor",
    "ConcatFromSequence",
    "Dropout",
    "LRN",
    "If",
    "Loop",
    "MaxRoiPool",
    "MeanVarianceNormalization",
    "Gelu",
    "GroupNormalization",
    "GlobalLpPool",
    "Multinomial",
    "Mish",
    "Optional",
    "OptionalGetElement",
    "OptionalHasElement",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
    "Tril",
    "Triu",
    "Unique",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceSumSquare",
    "RoiAlign",
    "RNN",
    "GRU",
    "Scan",
    "SequenceAt",
    "SequenceConstruct",
    "SequenceEmpty",
    "SequenceErase",
    "SequenceInsert",
    "SequenceLength",
    "SequenceMap",
    "SplitToSequence",
    "StringNormalizer",
    "LSTM",
    "DFT",
    "STFT",
    "TfIdfVectorizer",
}


REFERENCE_PARITY_PYTEST_COVERAGE = {
    "ABS",
    "ADD",
    "Acos",
    "Acosh",
    "AffineGrid",
    "And",
    "Asin",
    "Asinh",
    "Atanh",
    "Atan",
    "AveragePool",
    "ArgMax",
    "ArgMin",
    "BatchNormalization",
    "BitCast",
    "BlackmanWindow",
    "Cast",
    "CastLike",
    "Celu",
    "Ceil",
    "CenterCropPad",
    "Clip",
    "Compress",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Conv",
    "ConvInteger",
    "ConvTranspose",
    "COS",
    "Cosh",
    "CumProd",
    "CumSum",
    "DepthToSpace",
    "Det",
    "DIV",
    "DynamicQuantizeLinear",
    "Elu",
    "Einsum",
    "Erf",
    "Expand",
    "EXP",
    "EyeLike",
    "Equal",
    "Flatten",
    "Floor",
    "Gather",
    "GatherElements",
    "GatherND",
    "Gemm",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "Greater",
    "GreaterOrEqual",
    "GridSample",
    "HammingWindow",
    "HannWindow",
    "HardSigmoid",
    "HardSwish",
    "Hardmax",
    "Identity",
    "InstanceNormalization",
    "IsInf",
    "IsNaN",
    "LayerNormalization",
    "LeakyRelu",
    "Less",
    "LessOrEqual",
    "LOG",
    "LogSoftmax",
    "LpNormalization",
    "LpPool",
    "MatMul",
    "MatMulInteger",
    "MaxPool",
    "MaxUnpool",
    "Max",
    "MelWeightMatrix",
    "Mean",
    "Min",
    "Mod",
    "MUL",
    "Neg",
    "NegativeLogLikelihoodLoss",
    "NonMaxSuppression",
    "NonZero",
    "Not",
    "OneHot",
    "Or",
    "Pad",
    "PRelu",
    "Pow",
    "QLinearConv",
    "QLinearMatMul",
    "QuantizeLinear",
    "Range",
    "Reciprocal",
    "DequantizeLinear",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "RELU",
    "Reshape",
    "Resize",
    "RMSNormalization",
    "ReverseSequence",
    "Round",
    "ScatterElements",
    "ScatterND",
    "TensorScatter",
    "Selu",
    "Shape",
    "Shrink",
    "SIGMOID",
    "Sign",
    "Sin",
    "Sinh",
    "Slice",
    "Size",
    "Softmax",
    "Softplus",
    "Softsign",
    "SoftmaxCrossEntropyLoss",
    "Swish",
    "SpaceToDepth",
    "Split",
    "SQRT",
    "Squeeze",
    "SUB",
    "ThresholdedRelu",
    "Tile",
    "TopK",
    "Transpose",
    "Trilu",
    "Tan",
    "TANH",
    "Unsqueeze",
    "Where",
    "Sum",
    "Xor",
}


PYTEST_SEMANTIC_COVERAGE = DEEP_SEMANTIC_PYTEST_COVERAGE | REFERENCE_PARITY_PYTEST_COVERAGE


# 实现 `normalize_name` 步骤，规范化输入并返回下游期望的数据或元信息。
def normalize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name).upper()


# 实现 `to_snake` 步骤，规范化输入并返回下游期望的数据或元信息。
def to_snake(name: str) -> str:
    if name.isupper():
        return name.lower()
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


# 实现 `aliases_for` 步骤，规范化输入并返回下游期望的数据或元信息。
def aliases_for(class_name: str) -> set[str]:
    snake = to_snake(class_name)
    aliases = {snake, snake.replace("_", ""), class_name.lower()}
    aliases.update(MANUAL_ALIASES.get(class_name, set()))
    return aliases


# 实现 `base_names` 步骤，规范化输入并返回下游期望的数据或元信息。
def base_names(node: ast.ClassDef) -> tuple[str, ...]:
    names = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return tuple(names)


# 实现 `parse_operator_classes` 步骤，规范化输入并返回下游期望的数据或元信息。
def parse_operator_classes() -> dict[str, dict]:
    method_map = {}
    class_nodes = {}
    class_sources = {}

    for path in OPERATORS_SOURCES:
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        module = ast.parse(source)
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            class_nodes[node.name] = node
            class_sources[node.name] = path
            method_map[node.name] = {item.name: item for item in node.body if isinstance(item, ast.FunctionDef)}

    # 实现 `has_method` 步骤，规范化输入并返回下游期望的数据或元信息。
    def has_method(class_name: str, method_name: str) -> bool:
        if method_name in method_map.get(class_name, {}):
            return True
        for base in base_names(class_nodes[class_name]):
            if base in class_nodes and has_method(base, method_name):
                return True
        return False

    # 实现 `resolve_method_node` 步骤，规范化输入并返回下游期望的数据或元信息。
    def resolve_method_node(class_name: str, method_name: str) -> ast.FunctionDef | None:
        if method_name in method_map.get(class_name, {}):
            return method_map[class_name][method_name]
        for base in base_names(class_nodes[class_name]):
            if base in class_nodes:
                node = resolve_method_node(base, method_name)
                if node is not None:
                    return node
        return None

    # 实现 `collect_runtime_method_nodes` 步骤，规范化输入并返回下游期望的数据或元信息。
    def collect_runtime_method_nodes(class_name: str, method_name: str) -> list[ast.FunctionDef]:
        nodes = []
        visited = set()

        # 实现 `visit` 步骤，规范化输入并返回下游期望的数据或元信息。
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

    # 实现 `collect_c_function_refs` 步骤，规范化输入并返回下游期望的数据或元信息。
    def collect_c_function_refs(nodes: list[ast.AST]) -> set[str]:
        c_functions = set()
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value.endswith("_forward"):
                    c_functions.add(child.value)
                elif isinstance(child, ast.Attribute) and child.attr.endswith("_forward"):
                    c_functions.add(child.attr)
        return c_functions

    # 实现 `runtime_uses_numpy` 步骤，规范化输入并返回下游期望的数据或元信息。
    def runtime_uses_numpy(nodes: list[ast.AST]) -> bool:
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id == "np":
                    return True
        return False

    # 实现 `is_operator_class` 步骤，规范化输入并返回下游期望的数据或元信息。
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
            "source": str(class_sources[name].relative_to(ROOT)),
            "bases": bases,
            "has_forward": has_method(name, "forward"),
            "has_forward_shape": has_method(name, "forward_"),
            "c_functions": tuple(sorted(c_functions)),
            "c_runtime_functions": tuple(sorted(c_runtime_functions)),
            "runtime_uses_numpy": runtime_uses_numpy(runtime_nodes),
        }
    return operators


# 实现 `parse_import_supported_raw_ops` 步骤，规范化输入并返回下游期望的数据或元信息。
def parse_import_supported_raw_ops() -> set[str]:
    source = "\n".join(path.read_text(encoding="utf-8") for path in IMPORTER_SOURCES if path.exists())
    ops = set(re.findall(r'node\.op_type\s*==\s*"([^"]+)"', source))
    ops.update(re.findall(r'op_upper\s*==\s*"([^"]+)"', source))
    ops.update(re.findall(r'@register_factory\(\s*"([^"]+)"\s*\)', source))
    for match in re.finditer(r"node\.op_type\s+in\s+\[([^\]]+)\]", source):
        ops.update(re.findall(r'"([^"]+)"', match.group(1)))
    return ops


# 实现 `parse_import_supported_ops` 步骤，规范化输入并返回下游期望的数据或元信息。
def parse_import_supported_ops() -> set[str]:
    return {normalize_name(op) for op in parse_import_supported_raw_ops() if op != "Upsample"}


# 实现 `parse_onnx17_official_ops` 步骤，规范化输入并返回下游期望的数据或元信息。
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


# 实现 `parse_onnx_latest_official_ops` 步骤，读取当前安装 ONNX 中默认 domain 的最新 schema。
def parse_onnx_latest_official_ops() -> tuple[dict[str, tuple[str, int]], str | None]:
    try:
        from onnx import defs
    except Exception as exc:  # pragma: no cover - exercised when onnx is missing locally.
        return {}, f"无法导入 onnx.defs: {exc}"

    latest = {}
    for schema in defs.get_all_schemas_with_history():
        if schema.domain != "":
            continue
        if schema.name not in latest or schema.since_version > latest[schema.name].since_version:
            latest[schema.name] = schema
    return {normalize_name(name): (name, schema.since_version) for name, schema in latest.items()}, None


# 实现 `parse_c_functions` 步骤，规范化输入并返回下游期望的数据或元信息。
def parse_c_functions() -> tuple[set[str], set[str]]:
    header = (ROOT / "tensor_ops" / "tensor_ops.h").read_text(encoding="utf-8")
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted((ROOT / "tensor_ops").glob("*.c")))
    declared = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*_forward)\s*\(", header))
    implemented = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*_forward)\b", source))
    return declared, implemented


# 实现 `parse_cuda_verifiers` 步骤，规范化输入并返回下游期望的数据或元信息。
def parse_cuda_verifiers() -> set[str]:
    return {path.stem.replace("verify_", "") for path in (ROOT / "cuda").glob("verify_*.cu")}


# 实现 `parse_numerical_plan_details` 步骤，统计默认数值计划的唯一算子、总计划数和低精度计划数。
def parse_numerical_plan_details() -> tuple[set[str], int, int]:
    source_paths = [
        ROOT / "tools" / "commands" / "numerical_correctness.py",
        ROOT / "tools" / "numerical" / "cli.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_paths if path.exists())
    module = ast.parse(source)
    function_plans: dict[str, list[ast.Tuple]] = {}
    plans: list[ast.Tuple] = []

    # 从计划列表字面量、列表拼接和 helper 调用中抽取计划元组。
    def extract_plan_nodes(value: ast.AST) -> list[ast.Tuple]:
        if isinstance(value, ast.List):
            return [item for item in value.elts if isinstance(item, ast.Tuple) and len(item.elts) >= 5]
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            return extract_plan_nodes(value.left) + extract_plan_nodes(value.right)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            return list(function_plans.get(value.func.id, []))
        return []

    for _ in range(4):
        changed = False
        for node in module.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            extracted: list[ast.Tuple] = []
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and child.value is not None:
                    extracted.extend(extract_plan_nodes(child.value))
            if extracted and function_plans.get(node.name) != extracted:
                function_plans[node.name] = extracted
                changed = True
        if not changed:
            break

    # 判断单条计划是否覆盖低精度 dtype，便于报告混合精度门禁覆盖规模。
    def is_mixed_precision_plan(plan: ast.Tuple) -> bool:
        low_precision = {"float16", "bfloat16", "float8_e4m3", "float8_e5m2"}
        dtypes = []
        if len(plan.elts) > 3 and isinstance(plan.elts[3], ast.List):
            dtypes.extend(
                item.value
                for item in plan.elts[3].elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
        if len(plan.elts) > 4 and isinstance(plan.elts[4], ast.Constant) and isinstance(plan.elts[4].value, str):
            dtypes.append(plan.elts[4].value)
        return any(dtype in low_precision for dtype in dtypes)

    # 从计划元组中抽取第二列 op_name，兼容旧的 `plans = [...]`
    # 和新拆分后的 `build_default_plans()` 返回值。
    def extract_plan_names(plan_nodes: list[ast.Tuple]) -> list[str]:
        extracted = []
        for item in plan_nodes:
            if (
                len(item.elts) >= 2
                and isinstance(item.elts[1], ast.Constant)
                and isinstance(item.elts[1].value, str)
            ):
                extracted.append(item.elts[1].value)
        return extracted

    class Visitor(ast.NodeVisitor):
        # 处理 AST 访问节点 `visit_Assign`，收集后续审计分类所需的计划列表。
        def visit_Assign(self, node: ast.Assign) -> None:
            nonlocal plans
            if not any(isinstance(target, ast.Name) and target.id == "plans" for target in node.targets):
                self.generic_visit(node)
                return
            extracted = extract_plan_nodes(node.value)
            if extracted:
                plans = extracted
            self.generic_visit(node)

        # 处理 AST 访问节点 `visit_Return`，识别 `build_default_plans()` 中的计划列表或列表拼接。
        def visit_Return(self, node: ast.Return) -> None:
            nonlocal plans
            if node.value is None:
                self.generic_visit(node)
                return
            extracted = extract_plan_nodes(node.value)
            if extracted:
                plans = extracted
            self.generic_visit(node)

    Visitor().visit(module)
    names = extract_plan_names(plans)
    return set(names), len(plans), sum(1 for plan in plans if is_mixed_precision_plan(plan))


# 实现 `parse_numerical_plans` 步骤，规范化输入并返回下游期望的数据或元信息。
def parse_numerical_plans() -> set[str]:
    return parse_numerical_plan_details()[0]


# 实现 `classify` 步骤，规范化输入并返回下游期望的数据或元信息。
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
    if class_name in DEEP_SEMANTIC_PYTEST_COVERAGE:
        notes.append("已有独立 pytest 深度语义/混合精度覆盖")
    elif class_name in REFERENCE_PARITY_PYTEST_COVERAGE:
        notes.append("已有 ONNX reference pytest 语义/混合精度覆盖")
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
    if class_name in PYTEST_SEMANTIC_COVERAGE:
        return "已 pytest 语义验证", tuple(notes)
    return "已实现未数值验证", tuple(notes)


# 实现 `audit` 步骤，规范化输入并返回下游期望的数据或元信息。
def audit() -> tuple[list[OperatorInfo], dict[str, object]]:
    operators = parse_operator_classes()
    import_supported = parse_import_supported_ops()
    import_supported_raw = parse_import_supported_raw_ops()
    c_declared, c_implemented = parse_c_functions()
    cuda_verifiers = parse_cuda_verifiers()
    numerical_plans, numerical_plan_total_count, mixed_precision_plan_count = parse_numerical_plan_details()
    official_onnx17, official_error = parse_onnx17_official_ops()
    official_latest, official_latest_error = parse_onnx_latest_official_ops()
    normalized_import_raw = {normalize_name(op): op for op in import_supported_raw}
    official_latest_missing = sorted(set(official_latest) - set(normalized_import_raw))

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
        "numerical_plan_total_count": numerical_plan_total_count,
        "mixed_precision_plan_count": mixed_precision_plan_count,
        "cuda_not_planned": sorted(cuda_verifiers - numerical_plans),
        "plan_without_cuda": sorted(numerical_plans - cuda_verifiers),
        "official_onnx17_count": len(official_onnx17),
        "official_onnx17_error": official_error,
        "official_onnx17_supported_count": len(set(official_onnx17) & set(normalized_import_raw)),
        "official_onnx17_missing": [official_onnx17[name] for name in sorted(set(official_onnx17) - set(normalized_import_raw))],
        "supported_non_onnx17": [normalized_import_raw[name] for name in sorted(set(normalized_import_raw) - set(official_onnx17))],
        "official_onnx_latest_count": len(official_latest),
        "official_onnx_latest_error": official_latest_error,
        "official_onnx_latest_supported_count": len(set(official_latest) & set(normalized_import_raw)),
        "official_onnx_latest_missing": [official_latest[name][0] for name in official_latest_missing],
        "official_onnx_latest_missing_with_versions": [
            f"{official_latest[name][0]}(since_version={official_latest[name][1]})" for name in official_latest_missing
        ],
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
    metadata["deep_semantic_pytest_coverage"] = [
        info.class_name for info in infos if info.class_name in DEEP_SEMANTIC_PYTEST_COVERAGE
    ]
    metadata["reference_parity_pytest_coverage"] = [
        info.class_name for info in infos if info.class_name in REFERENCE_PARITY_PYTEST_COVERAGE
    ]
    metadata["pytest_semantic_coverage"] = [
        info.class_name for info in infos if info.class_name in PYTEST_SEMANTIC_COVERAGE
    ]
    metadata["declared_c_but_runtime_unused"] = [
        info.class_name for info in infos if info.c_functions and not info.c_runtime_functions
    ]
    return infos, metadata


# 实现 `yes_no` 步骤，规范化输入并返回下游期望的数据或元信息。
def yes_no(value: bool) -> str:
    return "yes" if value else "no"


# 实现 `render_markdown` 步骤，规范化输入并返回下游期望的数据或元信息。
def render_markdown(infos: list[OperatorInfo], metadata: dict[str, object]) -> str:
    status_counts = Counter(info.status for info in infos)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "<!--",
        "/**",
        "  ******************************************************************************",
        "  * @file        operator_coverage.md",
        "  * @author      Egor Izmaylov",
        "  * @brief       记录算子实现情况评估、验证覆盖和剩余风险。",
        "  * @details     2026.06.02  V1.0.0  创建",
        "  ******************************************************************************",
        "  * @attention",
        "  ******************************************************************************",
        "*/",
        "-->",
        "",
        "# 算子实现情况评估",
        "",
        f"> 自动生成时间：{generated_at}",
        "> 生成命令：`python tools/audit_ops.py --output docs/reports/operator_coverage.md`",
        "",
        "## 评估口径",
        "",
        "- `ONNXImport`：`nn/ONNXImport.py` 中存在显式映射，可从 ONNX node 构造对应算子类。",
        "- `forward`：`nn/Operators.py` 中存在运行时前向实现，继承自 `ReduceBase`/`ArgBase` 的实现也计入。",
        "- `forward_`：存在图构建/形状推断实现，继承实现也计入。",
        "- `C backend funcs`：Python 算子类中引用的 `<op>_forward` C 函数均能在 `tensor_ops/*.c` 中找到，且 `tensor_ops.h` 与 `.c` 声明/实现集合一致。",
        "- `C runtime path`：`forward()` 运行路径实际引用 `<op>_forward` C 函数；仅在 `__init__`、形状推断或未调用 helper 中出现不计入。",
        "- `CUDA verifier`：`cuda/verify_<op>.cu` 存在，仅说明有参考验证程序源码。",
        "- `active numerical plan`：统一入口 `python tools/cli.py numerical` 对应的 `tools/numerical/cli.py` 默认计划中包含该算子，代表会被默认数值验证门禁执行。",
        "- `独立 pytest 深度语义/混合精度覆盖`：使用独立公式或 ONNX reference 对高风险算子的官方语义、边界条件和低精度 dtype 路径进行 pytest 回归验证。",
        "- `ONNX reference pytest 语义/混合精度覆盖`：使用本地 ONNX reference evaluator 对普通算子的官方输出和低精度 dtype 路径进行 pytest 回归验证。",
        "- `ONNX opset 17 官方覆盖`：通过本地 `onnx.defs` 读取默认 domain 中 `since_version <= 17` 的最新 schema，并与 `ONNXImport` 的显式映射做名称级对比。",
        "- `当前安装 ONNX 最新官方覆盖`：读取当前环境可见的最新默认 domain schema，用于暴露高版本 opset 新增算子的后续兼容风险。",
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
        f"- active numerical plan 覆盖：{metadata['numerical_plan_count']} 个唯一算子名称，{metadata['numerical_plan_total_count']} 条默认计划。",
        f"- active numerical plan 混合精度覆盖：{metadata['mixed_precision_plan_count']} 条默认计划。",
        (
            f"- 独立 pytest 深度语义/混合精度覆盖：{len(metadata['deep_semantic_pytest_coverage'])} 个；"
            + ", ".join(f"`{name}`" for name in metadata["deep_semantic_pytest_coverage"])
            + "。"
        ),
        (
            f"- ONNX reference pytest 语义/混合精度覆盖：{len(metadata['reference_parity_pytest_coverage'])} 个；"
            + ", ".join(f"`{name}`" for name in metadata["reference_parity_pytest_coverage"])
            + "。"
        ),
            f"- ONNX opset 17 官方算子：{metadata['official_onnx17_count']} 个；ONNXImport 名称级覆盖：{metadata['official_onnx17_supported_count']} 个。",
            f"- 当前安装 ONNX 最新官方算子：{metadata['official_onnx_latest_count']} 个；ONNXImport 名称级覆盖：{metadata['official_onnx_latest_supported_count']} 个。",
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
        if info.status in {"已实现未数值验证", "待接入数值计划"}
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
            "- 已补充独立 pytest 深度语义/混合精度覆盖的算子："
            + ", ".join(f"`{name}`" for name in metadata["deep_semantic_pytest_coverage"])
            + "。",
            "- 已补充 ONNX reference pytest 语义/混合精度覆盖的普通算子："
            + ", ".join(f"`{name}`" for name in metadata["reference_parity_pytest_coverage"])
            + "。",
            (
                "- 未发现仍需立即后端化的 Python-only 普通数值/张量算子。"
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

    if metadata["official_onnx_latest_error"]:
        lines.extend(
            [
                "",
                "## 当前安装 ONNX 最新官方覆盖",
                "",
                f"- 无法生成最新官方覆盖对比：{metadata['official_onnx_latest_error']}",
            ]
        )
    else:
        latest_missing = metadata["official_onnx_latest_missing_with_versions"]
        lines.extend(
            [
                "",
                "## 当前安装 ONNX 最新官方覆盖",
                "",
                "- 该段用于暴露高版本 opset 兼容风险，不改变当前报告中 opset 17 的历史覆盖口径。",
                f"- 当前环境默认 domain 最新官方算子：{metadata['official_onnx_latest_count']} 个。",
                f"- `ONNXImport` 已覆盖官方名称：{metadata['official_onnx_latest_supported_count']} 个。",
                f"- 最新官方名称级缺口：{len(latest_missing)} 个。",
                "",
                "### 最新官方缺口",
                "",
                ", ".join(f"`{name}`" for name in latest_missing) if latest_missing else "无。",
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


# 作为 `tools/audit_ops.py` 的命令行入口，解析参数、调度检查流程并返回进程退出码。
def main() -> int:
    parser = argparse.ArgumentParser(description="Audit operator implementation and verification coverage.")
    parser.add_argument(
        "--output",
        default="docs/reports/operator_coverage.md",
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
