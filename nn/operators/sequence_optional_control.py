# /**
#   ******************************************************************************
#   * @file        sequence_optional_control.py
#   * @author      Egor Izmaylov
#   * @brief       按算子职责分组保存 `sequence_optional_control` 相关 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *


def _tensor_metadata(value):
    if isinstance(value, Tensor_):
        return value
    if isinstance(value, Tensor):
        return Tensor_(*value.size, dtype=value.dtype)
    return None


def _merge_control_metadata(left, right):
    """Merge values that may reach the same carried output across zero or more steps."""
    if isinstance(right, Optional_):
        if isinstance(left, Optional_):
            element = _merge_control_metadata(left.element, right.element)
            initial_presence = left.present
        elif left is None:
            element = right.element
            initial_presence = False
        else:
            element = _merge_control_metadata(left, right.element)
            initial_presence = True
        presence = (
            initial_presence if initial_presence == right.present else None
        )
        return Optional_(element, presence)

    left_tensor = _tensor_metadata(left)
    right_tensor = _tensor_metadata(right)
    if left_tensor is not None and right_tensor is not None:
        if left_tensor.dtype != right_tensor.dtype:
            raise TypeError(
                f"control-flow carried dtype mismatch: {left_tensor.dtype} != {right_tensor.dtype}"
            )
        if left_tensor.size is None or right_tensor.size is None:
            return Tensor_(dtype=left_tensor.dtype, rank_known=False)
        if len(left_tensor.size) != len(right_tensor.size):
            return Tensor_(dtype=left_tensor.dtype, rank_known=False)
        shape = [
            left_dim if left_dim == right_dim else None
            for left_dim, right_dim in zip(left_tensor.size, right_tensor.size)
        ]
        return Tensor_(*shape, dtype=left_tensor.dtype)

    if isinstance(right, Sequence_):
        if isinstance(left, Sequence_):
            left_element = left.element
        elif isinstance(left, list):
            left_element = None
            for element in left:
                left_element = (
                    element if left_element is None
                    else _merge_control_metadata(left_element, element)
                )
        else:
            raise TypeError(
                f"control-flow carried kind mismatch: {type(left).__name__} != Sequence_"
            )
        element = (
            right.element if left_element is None
            else _merge_control_metadata(left_element, right.element)
        )
        return Sequence_(element)

    raise TypeError(
        f"unsupported control-flow carried metadata: "
        f"{type(left).__name__}, {type(right).__name__}"
    )


class SequenceEmpty(Ops):
    # 初始化 `SequenceEmpty` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `SequenceEmpty` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self):
        return {"tensor": [], "parameters": None}

    # 执行 `SequenceEmpty` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self):
        return {
            "tensor": Sequence_(Tensor_(dtype=self.dtype, rank_known=False), length=0),
            "parameters": None,
        }


class SequenceConstruct(Ops):
    # 初始化 `SequenceConstruct` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `SequenceConstruct` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, *inputs):
        return {"tensor": list(inputs), "parameters": None}

    # 执行 `SequenceConstruct` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, *inputs):
        return {"tensor": list(inputs), "parameters": None}


class SequenceAt(Ops):
    # 初始化 `SequenceAt` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `SequenceAt` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input_sequence, position):
        return {"tensor": input_sequence[_sequence_position(position, len(input_sequence))], "parameters": None}

    # 执行 `SequenceAt` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input_sequence, position):
        if isinstance(input_sequence, Sequence_):
            if input_sequence.length == 0:
                raise IndexError("cannot select from an empty sequence")
            if input_sequence.length is not None and hasattr(position, "data"):
                _sequence_position(position, input_sequence.length)
            return {"tensor": input_sequence.element, "parameters": None}
        return self.forward(input_sequence, position)


class SequenceInsert(Ops):
    # 初始化 `SequenceInsert` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `SequenceInsert` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input_sequence, tensor, position=None):
        output = list(input_sequence)
        pos = _sequence_position(position, len(output), default=len(output), allow_end=True)
        output.insert(pos, tensor)
        return {"tensor": output, "parameters": None}

    # 执行 `SequenceInsert` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input_sequence, tensor, position=None):
        if isinstance(input_sequence, Sequence_):
            element = _merge_control_metadata(input_sequence.element, tensor)
            length = None if input_sequence.length is None else input_sequence.length + 1
            return {"tensor": Sequence_(element, length), "parameters": None}
        return self.forward(input_sequence, tensor, position)


class SequenceErase(Ops):
    # 初始化 `SequenceErase` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `SequenceErase` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input_sequence, position=None):
        output = list(input_sequence)
        pos = _sequence_position(position, len(output), default=len(output) - 1)
        del output[pos]
        return {"tensor": output, "parameters": None}

    # 执行 `SequenceErase` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input_sequence, position=None):
        if isinstance(input_sequence, Sequence_):
            length = input_sequence.length
            if length is not None:
                if length == 0:
                    raise IndexError("cannot erase from an empty sequence")
                if position is not None and hasattr(position, "data"):
                    _sequence_position(position, length)
                length -= 1
            return {"tensor": Sequence_(input_sequence.element, length), "parameters": None}
        return self.forward(input_sequence, position)


class SequenceLength(Ops):
    # 初始化 `SequenceLength` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="int64", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = "int64"
        self.version = version

    # 执行 `SequenceLength` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input_sequence):
        return {"tensor": Tensor(dtype=self.dtype, data=np.array(len(input_sequence), dtype=np.int64)), "parameters": None}

    # 执行 `SequenceLength` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input_sequence):
        if isinstance(input_sequence, list):
            return self.forward(input_sequence)
        if isinstance(input_sequence, Sequence_) and input_sequence.length is not None:
            return {"tensor": Tensor(dtype=self.dtype, data=np.array(input_sequence.length, dtype=np.int64)), "parameters": None}
        return {"tensor": Tensor_(dtype=self.dtype), "parameters": None}


class ConcatFromSequence(Ops):
    # 初始化 `ConcatFromSequence` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=0, new_axis=0, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.new_axis = new_axis
        self.dtype = dtype
        self.version = version

    # 执行 `ConcatFromSequence` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input_sequence):
        if not input_sequence:
            raise ValueError("ConcatFromSequence requires a non-empty sequence")
        arrays = [tensor.data for tensor in input_sequence]
        if self.new_axis:
            out_data = np.stack(arrays, axis=self.axis)
        else:
            out_data = np.concatenate(arrays, axis=self.axis)
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `ConcatFromSequence` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input_sequence):
        if isinstance(input_sequence, Sequence_):
            if input_sequence.length == 0:
                raise ValueError("ConcatFromSequence requires a non-empty sequence")
            element = input_sequence.element
            if not isinstance(element, Tensor_):
                raise TypeError("ConcatFromSequence requires tensor sequence metadata")
            if element.size is None:
                return {"tensor": Tensor_(dtype=self.dtype, rank_known=False), "parameters": None}
            shapes = [tuple(element.size)]
            length = input_sequence.length
        else:
            if not input_sequence:
                raise ValueError("ConcatFromSequence requires a non-empty sequence")
            if any(tensor.size is None for tensor in input_sequence):
                return {"tensor": Tensor_(dtype=self.dtype, rank_known=False), "parameters": None}
            shapes = [tuple(tensor.size) for tensor in input_sequence]
            length = len(shapes)

        rank = len(shapes[0])
        output_rank = rank + int(bool(self.new_axis))
        axis = self.axis if self.axis >= 0 else self.axis + output_rank
        if axis < 0 or axis >= output_rank:
            raise ValueError(
                f"ConcatFromSequence axis {self.axis} is out of range for rank {output_rank}"
            )
        out_shape = list(shapes[0])
        for dim_index in range(rank):
            dimensions = [shape[dim_index] for shape in shapes]
            if any(dimension != dimensions[0] for dimension in dimensions):
                out_shape[dim_index] = None
        if self.new_axis:
            out_shape.insert(axis, length)
        else:
            sizes = [shape[axis] for shape in shapes]
            out_shape[axis] = (
                None if length is None or any(size is None for size in sizes)
                else sizes[0] * length if isinstance(input_sequence, Sequence_)
                else sum(sizes)
            )
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class SplitToSequence(Ops):
    # 初始化 `SplitToSequence` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=0, keepdims=1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.keepdims = keepdims
        self.dtype = dtype
        self.version = version

    # 封装 `_split_sizes` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _split_sizes(self, axis_dim, split=None):
        if split is None:
            step = 1
            return [1] * axis_dim
        values = np.asarray(split.data).astype(np.int64).reshape(-1)
        if values.size == 1:
            step = int(values[0])
            if step <= 0:
                raise ValueError("SplitToSequence split values must be positive")
            return [min(step, axis_dim - start) for start in range(0, axis_dim, step)]
        sizes = values.astype(int).tolist()
        if any(size <= 0 for size in sizes) or sum(sizes) != axis_dim:
            raise ValueError("SplitToSequence 1-D split must contain positive sizes that sum to the axis dimension")
        return sizes

    # 执行 `SplitToSequence` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input, split=None):
        axis = self.axis if self.axis >= 0 else self.axis + len(input.size)
        sizes = self._split_sizes(input.size[axis], split)
        result = []
        start = 0
        for size in sizes:
            slc = [slice(None)] * len(input.size)
            slc[axis] = slice(start, start + size)
            data = input.data[tuple(slc)]
            if split is None and not self.keepdims:
                data = np.squeeze(data, axis=axis)
            result.append(Tensor(*data.shape, dtype=self.dtype, data=data.copy()))
            start += size
        return {"tensor": result, "parameters": None}

    # 执行 `SplitToSequence` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input, split=None):
        axis = self.axis if self.axis >= 0 else self.axis + len(input.size)
        sizes = self._split_sizes(input.size[axis], split) if split is not None and hasattr(split, "data") and split.data is not None else [1] * input.size[axis]
        result = []
        for size in sizes:
            shape = list(input.size)
            shape[axis] = size
            if split is None and not self.keepdims:
                shape.pop(axis)
            result.append(Tensor_(*tuple(shape), dtype=self.dtype))
        return {"tensor": result, "parameters": None}


class Optional(Ops):
    # 初始化 `Optional` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", element_type=None, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.element_type = element_type
        self.version = version

    # 执行 `Optional` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input=None):
        return {"tensor": input, "parameters": None}

    # 执行 `Optional` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input=None):
        if input is None:
            element = (
                _graph_type_metadata(self.element_type)
                if self.element_type is not None
                else Tensor_(dtype=self.dtype, rank_known=False)
            )
            return {"tensor": Optional_(element, present=False), "parameters": None}
        return {"tensor": Optional_(input, present=True), "parameters": None}


class OptionalGetElement(Ops):
    # 初始化 `OptionalGetElement` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `OptionalGetElement` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        if input is None:
            raise ValueError("OptionalGetElement cannot read an empty optional")
        return {"tensor": input, "parameters": None}

    # 执行 `OptionalGetElement` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        if isinstance(input, Optional_):
            if input.present is False:
                raise ValueError("OptionalGetElement cannot read an empty optional")
            return {"tensor": input.element, "parameters": None}
        return self.forward(input)


class OptionalHasElement(Ops):
    # 初始化 `OptionalHasElement` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype="bool", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = "bool"
        self.version = version

    # 执行 `OptionalHasElement` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        return {"tensor": Tensor(dtype=self.dtype, data=np.array(input is not None, dtype=np.bool_)), "parameters": None}

    # 执行 `OptionalHasElement` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        return {"tensor": Tensor_(dtype=self.dtype), "parameters": None}


class If(Ops):
    # 初始化 `If` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, then_branch, else_branch, version="17", opset_imports=None):
        super().__init__(inputs, outputs)
        self.then_branch = then_branch
        self.else_branch = else_branch
        self.version = version
        self.opset_imports = opset_imports
        self.outer_scope_names = sorted(
            _graph_external_names(then_branch) | _graph_external_names(else_branch)
        )

    # 执行 `If` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, cond):
        return self.forward_with_context(None, cond)

    # 在外层作用域上下文中执行 `If` 的子图逻辑，用于控制流算子解析捕获值。
    def forward_with_context(self, outer_scope, cond):
        condition = bool(np.asarray(cond.data).item())
        graph = self.then_branch if condition else self.else_branch
        values = _run_graph_proto(graph, {}, outer_scope, self.opset_imports)
        if len(values) != len(graph.output):
            raise RuntimeError(
                f"If branch {graph.name!r} returned {len(values)} values for "
                f"{len(graph.output)} declared outputs"
            )
        outputs = tuple(
            _value_from_reference(value, value_info.type)
            for value, value_info in zip(values, graph.output)
        )
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

    @classmethod
    def _merge_declared_type(cls, then_type, else_type, output_index):
        then_kind = then_type.WhichOneof("value")
        else_kind = else_type.WhichOneof("value")
        if then_kind != else_kind:
            raise TypeError(
                f"If output {output_index} branch type mismatch: "
                f"{then_kind} != {else_kind}"
            )
        if then_type.HasField("sequence_type"):
            then_element = then_type.sequence_type.elem_type
            else_element = else_type.sequence_type.elem_type
            return Sequence_(
                cls._merge_declared_type(then_element, else_element, output_index)
            )
        if then_type.HasField("optional_type"):
            then_element = then_type.optional_type.elem_type
            else_element = else_type.optional_type.elem_type
            return Optional_(
                cls._merge_declared_type(then_element, else_element, output_index)
            )
        if not then_type.HasField("tensor_type"):
            raise TypeError(
                f"If output {output_index} has unsupported branch type {then_kind!r}"
            )

        then_tensor = then_type.tensor_type
        else_tensor = else_type.tensor_type
        if then_tensor.elem_type != else_tensor.elem_type:
            raise TypeError(
                f"If output {output_index} branch dtype mismatch: "
                f"{then_tensor.elem_type} != {else_tensor.elem_type}"
            )
        dtype = nn.onnx_dtype_mapping.get(then_tensor.elem_type)
        if dtype is None:
            raise TypeError(
                f"If output {output_index} has unsupported element type "
                f"{then_tensor.elem_type}"
            )

        if not then_tensor.HasField("shape") or not else_tensor.HasField("shape"):
            return Tensor_(dtype=dtype, rank_known=False)
        then_dims = then_tensor.shape.dim
        else_dims = else_tensor.shape.dim
        if len(then_dims) != len(else_dims):
            return Tensor_(dtype=dtype, rank_known=False)

        merged = []
        for then_dim, else_dim in zip(then_dims, else_dims):
            then_value = int(then_dim.dim_value) if then_dim.HasField("dim_value") else None
            else_value = int(else_dim.dim_value) if else_dim.HasField("dim_value") else None
            merged.append(then_value if then_value == else_value else None)
        return Tensor_(*merged, dtype=dtype)

    @classmethod
    def _merge_tensor_output(cls, then_info, else_info, output_index):
        return cls._merge_declared_type(then_info.type, else_info.type, output_index)

    # 执行 `If` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, cond):
        if len(self.then_branch.output) != len(self.else_branch.output):
            raise ValueError(
                "If branches declare different output counts: "
                f"{len(self.then_branch.output)} != {len(self.else_branch.output)}"
            )
        outputs = tuple(
            self._merge_tensor_output(then_info, else_info, index)
            for index, (then_info, else_info) in enumerate(
                zip(self.then_branch.output, self.else_branch.output)
            )
        )
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}


class Loop(Ops):
    # 初始化 `Loop` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, body, version="17", opset_imports=None):
        super().__init__(inputs, outputs)
        self.body = body
        self.version = version
        self.opset_imports = opset_imports
        self.outer_scope_names = sorted(_graph_external_names(body))

    # 封装 `_trip_count` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _trip_count(m):
        if m is None:
            return None
        return int(np.asarray(m.data).item())

    # 封装 `_condition` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _condition(cond):
        if cond is None:
            return True
        return bool(np.asarray(cond.data).item())

    # 执行 `Loop` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, m=None, cond=None, *loop_vars):
        return self.forward_with_context(None, m, cond, *loop_vars)

    # 在外层作用域上下文中执行 `Loop` 的子图逻辑，用于控制流算子解析捕获值。
    def forward_with_context(self, outer_scope, m=None, cond=None, *loop_vars):
        trip_count = self._trip_count(m)
        condition = self._condition(cond)
        if trip_count is None and cond is None:
            raise ValueError("Loop without trip count or condition would be unbounded")
        body_inputs = [value.name for value in self.body.input]
        carried_input_types = [
            value_info.type for value_info in self.body.input[2:2 + len(loop_vars)]
        ]
        state_values = [
            _reference_value_for_type(value, type_proto)
            for value, type_proto in zip(loop_vars, carried_input_types)
        ]
        scan_outputs = None
        iteration = 0
        last_outputs = None
        while condition and (trip_count is None or iteration < trip_count):
            feeds = {}
            if body_inputs:
                feeds[body_inputs[0]] = np.asarray(iteration, dtype=np.int64)
            if len(body_inputs) > 1:
                feeds[body_inputs[1]] = np.asarray(condition, dtype=np.bool_)
            for name, value in zip(body_inputs[2:], state_values):
                feeds[name] = value
            last_outputs = list(_run_graph_proto(self.body, feeds, outer_scope, self.opset_imports))
            expected_outputs = len(self.body.output)
            if len(last_outputs) != expected_outputs:
                raise RuntimeError(
                    f"Loop body returned {len(last_outputs)} values for "
                    f"{expected_outputs} declared outputs"
                )
            condition = bool(np.asarray(last_outputs[0]).item())
            state_values = last_outputs[1:1 + len(state_values)]
            produced_scan = last_outputs[1 + len(state_values):]
            if scan_outputs is None:
                scan_outputs = [[] for _ in produced_scan]
            for bucket, value in zip(scan_outputs, produced_scan):
                bucket.append(np.asarray(value))
            iteration += 1
        if last_outputs is None:
            final_values = state_values
        else:
            final_values = state_values
        if scan_outputs is None:
            scan_value_infos = self.body.output[1 + len(state_values):]
            symbol_bindings = _graph_symbol_bindings(
                self.body.input[2:], loop_vars, "Loop body"
            )
            stacked_scan = []
            for value_info in scan_value_infos:
                element_shape, dtype = _graph_tensor_metadata(
                    value_info, symbol_bindings, "Loop empty scan output"
                )
                stacked_scan.append(
                    np.empty((0, *element_shape), dtype=nn.DTYPE_TO_NUMPY[dtype])
                )
        else:
            stacked_scan = [np.stack(bucket, axis=0) for bucket in scan_outputs]
        final_value_infos = self.body.output[1:1 + len(final_values)]
        converted_states = [
            _value_from_reference(value, value_info.type)
            for value, value_info in zip(final_values, final_value_infos)
        ]
        converted_scans = [_tensor_from_numpy(value) for value in stacked_scan]
        outputs = tuple(converted_states + converted_scans)
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

    # 执行 `Loop` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, m=None, cond=None, *loop_vars):
        outputs = []
        state_count = len(loop_vars)
        symbol_bindings = _graph_symbol_bindings(
            self.body.input[2:2 + state_count], loop_vars, "Loop body"
        )
        for loop_var, value_info in zip(
            loop_vars, self.body.output[1:1 + state_count]
        ):
            declared = _graph_value_shape(value_info, symbol_bindings)
            outputs.append(_merge_control_metadata(loop_var, declared))
        for value_info in self.body.output[1 + state_count:]:
            scan = _graph_value_shape(value_info, symbol_bindings)
            if scan.size is None:
                outputs.append(Tensor_(dtype=scan.dtype, rank_known=False))
            else:
                outputs.append(Tensor_(None, *scan.size, dtype=scan.dtype))
        return {"tensor": outputs[0] if len(outputs) == 1 else tuple(outputs), "parameters": None}


class Scan(Ops):
    # 初始化 `Scan` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(
        self,
        inputs,
        outputs,
        body,
        num_scan_inputs,
        scan_input_axes=None,
        scan_input_directions=None,
        scan_output_axes=None,
        scan_output_directions=None,
        version="17",
        opset_imports=None,
    ):
        super().__init__(inputs, outputs)
        self.body = body
        self.num_scan_inputs = int(num_scan_inputs)
        self.scan_input_axes = list(scan_input_axes or [0] * self.num_scan_inputs)
        self.scan_input_directions = list(scan_input_directions or [0] * self.num_scan_inputs)
        self.scan_output_axes = list(scan_output_axes or [])
        self.scan_output_directions = list(scan_output_directions or [])
        self.version = version
        self.opset_imports = opset_imports
        self.outer_scope_names = sorted(_graph_external_names(body))

    # 执行 `Scan` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, *inputs):
        return self.forward_with_context(None, *inputs)

    @staticmethod
    def _normalized_axis(axis, rank, label):
        normalized = axis + rank if axis < 0 else axis
        if normalized < 0 or normalized >= rank:
            raise ValueError(f"Scan {label} axis {axis} is out of range for rank {rank}")
        return normalized

    def _empty_scan_output(self, value_info, output_index, body_input_shapes):
        tensor_type = value_info.type.tensor_type
        if not value_info.type.HasField("tensor_type") or tensor_type.elem_type == 0:
            raise ValueError(
                f"Scan cannot construct empty output {output_index}: body output "
                f"{value_info.name!r} has no declared tensor type"
            )

        symbol_values = _graph_symbol_bindings(
            self.body.input,
            [Tensor_(*shape) for shape in body_input_shapes],
            "Scan body",
        )
        element_shape, dtype = _graph_tensor_metadata(
            value_info, symbol_values, "Scan empty output"
        )

        output_rank = len(element_shape) + 1
        requested_axis = (
            self.scan_output_axes[output_index]
            if output_index < len(self.scan_output_axes) else 0
        )
        output_axis = self._normalized_axis(requested_axis, output_rank, "output")
        output_shape = list(element_shape)
        output_shape.insert(output_axis, 0)
        return np.empty(output_shape, dtype=nn.DTYPE_TO_NUMPY[dtype])

    # 在外层作用域上下文中执行 `Scan` 的子图逻辑，用于控制流算子解析捕获值。
    def forward_with_context(self, outer_scope, *inputs):
        num_states = len(inputs) - self.num_scan_inputs
        states = [_tensor_to_numpy(value) for value in inputs[:num_states]]
        scan_inputs = [_tensor_to_numpy(value) for value in inputs[num_states:]]
        body_inputs = [value.name for value in self.body.input]
        body_outputs = [value.name for value in self.body.output]
        scan_axes = [
            self._normalized_axis(
                self.scan_input_axes[index] if index < len(self.scan_input_axes) else 0,
                value.ndim,
                "input",
            )
            for index, value in enumerate(scan_inputs)
        ]
        scan_lengths = [value.shape[axis] for value, axis in zip(scan_inputs, scan_axes)]
        if not scan_lengths:
            raise ValueError("Scan requires at least one scan input")
        if len(set(scan_lengths)) != 1:
            raise ValueError(f"Scan inputs have different sequence lengths: {scan_lengths}")
        trip_count = scan_lengths[0]
        collected = None
        for iteration in range(trip_count):
            feeds = {name: value for name, value in zip(body_inputs[:num_states], states)}
            for index, value in enumerate(scan_inputs):
                axis = scan_axes[index]
                take_index = trip_count - 1 - iteration if self.scan_input_directions[index] else iteration
                feeds[body_inputs[num_states + index]] = np.take(value, take_index, axis=axis)
            result = list(_run_graph_proto(self.body, feeds, outer_scope, self.opset_imports))
            output_map = dict(zip(body_outputs, result))
            states = [np.asarray(output_map[name]) for name in body_outputs[:num_states]]
            scan_values = [np.asarray(output_map[name]) for name in body_outputs[num_states:]]
            if collected is None:
                collected = [[] for _ in scan_values]
            for bucket, value in zip(collected, scan_values):
                bucket.append(value)
        scan_outputs = []
        if collected is None:
            body_input_shapes = [value.shape for value in states]
            body_input_shapes.extend(
                value.shape[:axis] + value.shape[axis + 1:]
                for value, axis in zip(scan_inputs, scan_axes)
            )
            try:
                inferred_outputs = _infer_graph_outputs_for_shapes(
                    self.body, body_input_shapes, self.opset_imports
                )
            except Exception:
                inferred_outputs = list(self.body.output)
            for index, value_info in enumerate(inferred_outputs[num_states:]):
                scan_outputs.append(self._empty_scan_output(value_info, index, body_input_shapes))
        else:
            for index, bucket in enumerate(collected):
                values = list(reversed(bucket)) if index < len(self.scan_output_directions) and self.scan_output_directions[index] else bucket
                requested_axis = self.scan_output_axes[index] if index < len(self.scan_output_axes) else 0
                axis = self._normalized_axis(
                    requested_axis, bucket[0].ndim + 1, "output"
                )
                scan_outputs.append(np.stack(values, axis=axis))
        outputs = tuple(_tensor_from_numpy(value) for value in states + scan_outputs)
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

    # 执行 `Scan` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, *inputs):
        num_states = len(inputs) - self.num_scan_inputs
        outputs = []
        body_values = list(inputs[:num_states])
        scan_inputs = inputs[num_states:]
        scan_axes = []
        for idx, scan_input in enumerate(scan_inputs):
            requested_axis = (
                self.scan_input_axes[idx] if idx < len(self.scan_input_axes) else 0
            )
            axis = self._normalized_axis(requested_axis, len(scan_input.size), "input")
            scan_axes.append(axis)
            element_shape = list(scan_input.size)
            del element_shape[axis]
            body_values.append(Tensor_(*element_shape, dtype=scan_input.dtype))
        symbol_bindings = _graph_symbol_bindings(
            self.body.input, body_values, "Scan body"
        )
        for state, value_info in zip(inputs[:num_states], self.body.output[:num_states]):
            declared = _graph_value_shape(value_info, symbol_bindings)
            outputs.append(_merge_control_metadata(state, declared))
        for idx, value_info in enumerate(self.body.output[num_states:]):
            elem = _graph_value_shape(value_info, symbol_bindings)
            if elem.size is None:
                outputs.append(Tensor_(dtype=elem.dtype, rank_known=False))
                continue
            scan_input = inputs[num_states + min(idx, self.num_scan_inputs - 1)]
            axis = scan_axes[min(idx, self.num_scan_inputs - 1)]
            length = scan_input.size[axis]
            requested_out_axis = self.scan_output_axes[idx] if idx < len(self.scan_output_axes) else 0
            shape = list(elem.size)
            out_axis = self._normalized_axis(
                requested_out_axis, len(shape) + 1, "output"
            )
            shape.insert(out_axis, length)
            outputs.append(Tensor_(*shape, dtype=elem.dtype))
        return {"tensor": outputs[0] if len(outputs) == 1 else tuple(outputs), "parameters": None}


class SequenceMap(Ops):
    # 初始化 `SequenceMap` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, body, version="17", opset_imports=None):
        super().__init__(inputs, outputs)
        self.body = body
        self.version = version
        self.opset_imports = opset_imports
        self.outer_scope_names = sorted(_graph_external_names(body))

    # 执行 `SequenceMap` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input_sequence, *additional_inputs):
        return self.forward_with_context(None, input_sequence, *additional_inputs)

    # 在外层作用域上下文中执行 `SequenceMap` 的子图逻辑，用于控制流算子解析捕获值。
    def forward_with_context(self, outer_scope, input_sequence, *additional_inputs):
        body_inputs = [value.name for value in self.body.input]
        collected = [[] for _ in self.body.output]
        for idx, item in enumerate(input_sequence):
            feeds = {body_inputs[0]: _tensor_to_numpy(item)}
            for name, value in zip(body_inputs[1:], additional_inputs):
                feeds[name] = _tensor_to_numpy(value[idx]) if isinstance(value, list) else _tensor_to_numpy(value)
            reference_values = list(_run_graph_proto(
                self.body, feeds, outer_scope, self.opset_imports
            ))
            if len(reference_values) != len(self.body.output):
                raise RuntimeError(
                    f"SequenceMap body returned {len(reference_values)} values for "
                    f"{len(self.body.output)} declared outputs"
                )
            result = [
                _value_from_reference(value, value_info.type)
                for value, value_info in zip(reference_values, self.body.output)
            ]
            for bucket, value in zip(collected, result):
                bucket.append(value)
        outputs = tuple(collected)
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

    # 执行 `SequenceMap` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input_sequence, *additional_inputs):
        def body_metadata(item, index):
            values = [item]
            for additional in additional_inputs:
                if isinstance(additional, Sequence_):
                    values.append(additional.element)
                elif isinstance(additional, list):
                    if index is None:
                        element = None
                        for candidate in additional:
                            element = (
                                candidate if element is None
                                else _merge_control_metadata(element, candidate)
                            )
                        values.append(element)
                    else:
                        values.append(additional[index])
                else:
                    values.append(additional)
            bindings = _graph_symbol_bindings(self.body.input, values, "SequenceMap body")
            return tuple(
                _graph_value_shape(value_info, bindings)
                for value_info in self.body.output
            )

        if isinstance(input_sequence, Sequence_):
            elements = body_metadata(input_sequence.element, None)
            outputs = tuple(
                Sequence_(element, input_sequence.length) for element in elements
            )
        elif input_sequence:
            outputs = tuple([] for _ in self.body.output)
            for index, item in enumerate(input_sequence):
                elements = body_metadata(item, index)
                for bucket, element in zip(outputs, elements):
                    bucket.append(element)
        else:
            # An empty sequence still carries the declared output element kind.
            elements = tuple(_graph_value_shape(info) for info in self.body.output)
            outputs = tuple(Sequence_(element, 0) for element in elements)
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}
