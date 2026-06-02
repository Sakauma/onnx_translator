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
        return {"tensor": [], "parameters": None}


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
        if not input_sequence:
            return {"tensor": Tensor_(0, dtype=self.dtype), "parameters": None}
        shapes = [tuple(tensor.size) for tensor in input_sequence]
        if self.new_axis:
            axis = self.axis if self.axis >= 0 else self.axis + len(shapes[0]) + 1
            out_shape = list(shapes[0])
            out_shape.insert(axis, len(shapes))
        else:
            axis = self.axis if self.axis >= 0 else self.axis + len(shapes[0])
            out_shape = list(shapes[0])
            out_shape[axis] = sum(shape[axis] for shape in shapes)
        return {"tensor": Tensor_(*tuple(out_shape), dtype=self.dtype), "parameters": None}


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
    def __init__(self, inputs, outputs, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.dtype = dtype
        self.version = version

    # 执行 `Optional` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input=None):
        return {"tensor": input, "parameters": None}

    # 执行 `Optional` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input=None):
        return {"tensor": input, "parameters": None}


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
        if input is None:
            return self.forward(input)
        return {"tensor": Tensor_(dtype=self.dtype), "parameters": None}


class If(Ops):
    # 初始化 `If` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, then_branch, else_branch, version="17"):
        super().__init__(inputs, outputs)
        self.then_branch = then_branch
        self.else_branch = else_branch
        self.version = version
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
        outputs = tuple(_tensor_from_numpy(value) for value in _run_graph_proto(graph, {}, outer_scope))
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

    # 执行 `If` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, cond):
        graph = self.then_branch
        outputs = tuple(_graph_value_shape(value_info) for value_info in graph.output)
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}


class Loop(Ops):
    # 初始化 `Loop` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, body, version="17"):
        super().__init__(inputs, outputs)
        self.body = body
        self.version = version
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
        state_values = [_tensor_to_numpy(value) for value in loop_vars]
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
            last_outputs = list(_run_graph_proto(self.body, feeds, outer_scope))
            condition = bool(np.asarray(last_outputs[0]).item())
            state_values = [np.asarray(value) for value in last_outputs[1:1 + len(state_values)]]
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
            stacked_scan = []
            for value_info in scan_value_infos:
                inferred = _graph_value_shape(value_info)
                np_dtype = nn.DTYPE_TO_NUMPY.get(inferred.dtype, np.float32)
                stacked_scan.append(np.empty((0, *inferred.size), dtype=np_dtype))
        else:
            stacked_scan = [np.stack(bucket, axis=0) for bucket in scan_outputs]
        outputs = tuple(_tensor_from_numpy(value) for value in final_values + stacked_scan)
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

    # 执行 `Loop` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, m=None, cond=None, *loop_vars):
        outputs = []
        state_count = max(0, len(self.body.input) - 2)
        for value_info in self.body.output[1:1 + state_count]:
            outputs.append(_graph_value_shape(value_info))
        for value_info in self.body.output[1 + state_count:]:
            scan = _graph_value_shape(value_info)
            outputs.append(Tensor_(1, *scan.size, dtype=scan.dtype))
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
    ):
        super().__init__(inputs, outputs)
        self.body = body
        self.num_scan_inputs = int(num_scan_inputs)
        self.scan_input_axes = list(scan_input_axes or [0] * self.num_scan_inputs)
        self.scan_input_directions = list(scan_input_directions or [0] * self.num_scan_inputs)
        self.scan_output_axes = list(scan_output_axes or [])
        self.scan_output_directions = list(scan_output_directions or [])
        self.version = version
        self.outer_scope_names = sorted(_graph_external_names(body))

    # 执行 `Scan` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, *inputs):
        return self.forward_with_context(None, *inputs)

    # 在外层作用域上下文中执行 `Scan` 的子图逻辑，用于控制流算子解析捕获值。
    def forward_with_context(self, outer_scope, *inputs):
        num_states = len(inputs) - self.num_scan_inputs
        states = [_tensor_to_numpy(value) for value in inputs[:num_states]]
        scan_inputs = [_tensor_to_numpy(value) for value in inputs[num_states:]]
        body_inputs = [value.name for value in self.body.input]
        body_outputs = [value.name for value in self.body.output]
        trip_count = scan_inputs[0].shape[self.scan_input_axes[0]]
        collected = None
        for iteration in range(trip_count):
            feeds = {name: value for name, value in zip(body_inputs[:num_states], states)}
            for index, value in enumerate(scan_inputs):
                axis = self.scan_input_axes[index] if index < len(self.scan_input_axes) else 0
                take_index = trip_count - 1 - iteration if self.scan_input_directions[index] else iteration
                feeds[body_inputs[num_states + index]] = np.take(value, take_index, axis=axis)
            result = list(_run_graph_proto(self.body, feeds, outer_scope))
            output_map = dict(zip(body_outputs, result))
            states = [np.asarray(output_map[name]) for name in body_outputs[:num_states]]
            scan_values = [np.asarray(output_map[name]) for name in body_outputs[num_states:]]
            if collected is None:
                collected = [[] for _ in scan_values]
            for bucket, value in zip(collected, scan_values):
                bucket.append(value)
        scan_outputs = []
        for index, bucket in enumerate(collected or []):
            values = list(reversed(bucket)) if index < len(self.scan_output_directions) and self.scan_output_directions[index] else bucket
            axis = self.scan_output_axes[index] if index < len(self.scan_output_axes) else 0
            scan_outputs.append(np.stack(values, axis=axis))
        outputs = tuple(_tensor_from_numpy(value) for value in states + scan_outputs)
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

    # 执行 `Scan` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, *inputs):
        num_states = len(inputs) - self.num_scan_inputs
        outputs = []
        for value_info in self.body.output[:num_states]:
            outputs.append(_graph_value_shape(value_info))
        for idx, value_info in enumerate(self.body.output[num_states:]):
            elem = _graph_value_shape(value_info)
            scan_input = inputs[num_states + min(idx, self.num_scan_inputs - 1)]
            axis = self.scan_input_axes[min(idx, len(self.scan_input_axes) - 1)] if self.scan_input_axes else 0
            length = scan_input.size[axis]
            out_axis = self.scan_output_axes[idx] if idx < len(self.scan_output_axes) else 0
            shape = list(elem.size)
            shape.insert(out_axis, length)
            outputs.append(Tensor_(*shape, dtype=elem.dtype))
        return {"tensor": outputs[0] if len(outputs) == 1 else tuple(outputs), "parameters": None}


class SequenceMap(Ops):
    # 初始化 `SequenceMap` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, body, version="17"):
        super().__init__(inputs, outputs)
        self.body = body
        self.version = version
        self.outer_scope_names = sorted(_graph_external_names(body))

    # 执行 `SequenceMap` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input_sequence, *additional_inputs):
        return self.forward_with_context(None, input_sequence, *additional_inputs)

    # 在外层作用域上下文中执行 `SequenceMap` 的子图逻辑，用于控制流算子解析捕获值。
    def forward_with_context(self, outer_scope, input_sequence, *additional_inputs):
        body_inputs = [value.name for value in self.body.input]
        collected = None
        for idx, item in enumerate(input_sequence):
            feeds = {body_inputs[0]: _tensor_to_numpy(item)}
            for name, value in zip(body_inputs[1:], additional_inputs):
                feeds[name] = _tensor_to_numpy(value[idx]) if isinstance(value, list) else _tensor_to_numpy(value)
            result = [_tensor_from_numpy(value) for value in _run_graph_proto(self.body, feeds, outer_scope)]
            if collected is None:
                collected = [[] for _ in result]
            for bucket, value in zip(collected, result):
                bucket.append(value)
        outputs = tuple(collected or [])
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}

    # 执行 `SequenceMap` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input_sequence, *additional_inputs):
        outputs = tuple([] for _ in self.body.output)
        return {"tensor": outputs[0] if len(outputs) == 1 else outputs, "parameters": None}
