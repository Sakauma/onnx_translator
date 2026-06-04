# /**
#   ******************************************************************************
#   * @file        loss_ranking_ops.py
#   * @author      Egor Izmaylov
#   * @brief       保存 `loss_ranking_ops` 分组中的 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *

class Multinomial(Ops):
    # 初始化 `Multinomial` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, dtype=6, sample_size=1, seed=None, version="17"):
        super().__init__(inputs, outputs)
        self.dtype = nn.onnx_dtype_mapping.get(dtype, "int32") if isinstance(dtype, int) else dtype
        self.sample_size = int(sample_size)
        self.seed = seed
        self.version = version
        if self.lib:
            self.lib.multinomial_forward.argtypes = [
                ctypes.POINTER(CTensor), ctypes.POINTER(CTensor), ctypes.c_int, ctypes.c_uint32
            ]

    # 执行 `Multinomial` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input):
        probs = np.asarray(input.data, dtype=np.float64)
        if probs.ndim != 2:
            raise ValueError(f"Multinomial expects rank-2 input, got shape {input.size}")
        if self.sample_size < 0:
            raise ValueError(f"Multinomial sample_size must be non-negative, got {self.sample_size}")
        if np.any(probs < 0):
            raise ValueError("Multinomial probabilities must be non-negative")
        if np.any(probs.sum(axis=1) <= 0):
            raise ValueError("Multinomial probabilities must have a positive sum")

        if self.lib is not None and input.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            out_shape = (probs.shape[0], self.sample_size)
            output_shape_c = (ctypes.c_int * 2)(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, 2, nn.DTYPE_MAP[self.dtype])
            seed = 0 if self.seed is None else int(self.seed)
            self.lib.multinomial_forward(input_c, output_c, ctypes.c_int(self.sample_size), ctypes.c_uint32(seed))
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        rng = np.random.default_rng(None if self.seed is None else int(self.seed))
        class_count = probs.shape[1]
        out = np.empty((probs.shape[0], self.sample_size), dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, np.int32))
        for row_idx, row in enumerate(probs):
            total = row.sum()
            out[row_idx] = rng.choice(class_count, size=self.sample_size, replace=True, p=row / total)
        return {"tensor": Tensor(*out.shape, dtype=self.dtype, data=out), "parameters": None}

    # 执行 `Multinomial` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input):
        if len(input.size) != 2:
            raise ValueError(f"Multinomial expects rank-2 input, got shape {input.size}")
        return {"tensor": Tensor_(input.size[0], self.sample_size, dtype=self.dtype), "parameters": None}


class NegativeLogLikelihoodLoss(Ops):
    # 初始化 `NegativeLogLikelihoodLoss` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, reduction="mean", ignore_index=None, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.negative_log_likelihood_loss_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int64,
            ]

    # 封装 `_reduction_code` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _reduction_code(self):
        if self.reduction == "none":
            return 0
        if self.reduction == "mean":
            return 1
        if self.reduction == "sum":
            return 2
        raise ValueError(f"Unsupported loss reduction {self.reduction!r}")

    # 封装 `_target_shape` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _target_shape(input_shape):
        if len(input_shape) < 2:
            raise ValueError(f"Loss input expects rank >= 2, got shape {input_shape}")
        return (input_shape[0],) + tuple(input_shape[2:])

    # 封装 `_gather_negative_scores` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _gather_negative_scores(log_probs, target, ignore_index):
        input_shape = log_probs.shape
        target_shape = target.shape
        n, c = input_shape[0], input_shape[1]
        reshaped = log_probs.reshape((n, c, -1))
        target_2d = target.reshape((n, -1))
        loss_2d = np.zeros((n, target_2d.shape[1]), dtype=log_probs.dtype)
        for i in range(n):
            for j in range(target_2d.shape[1]):
                cls = int(target_2d[i, j])
                if ignore_index is not None and cls == ignore_index:
                    continue
                if cls < 0 or cls >= c:
                    raise ValueError(f"Target class {cls} is out of range [0, {c})")
                loss_2d[i, j] = -reshaped[i, cls, j]
        return loss_2d.reshape(target_shape)

    # 封装 `_reduce` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _reduce(self, loss, target, weight=None):
        gather_weight = None
        if weight is not None:
            target_i = np.asarray(target, dtype=np.int64)
            clipped = np.clip(target_i, 0, len(weight.data) - 1)
            gather_weight = np.take(weight.data, clipped).astype(loss.dtype, copy=False)
            if self.ignore_index is not None:
                gather_weight = np.where(target_i == self.ignore_index, 0, gather_weight).astype(loss.dtype, copy=False)
        elif self.ignore_index is not None:
            gather_weight = np.where(target == self.ignore_index, 0, 1).astype(loss.dtype, copy=False)

        if gather_weight is not None:
            loss = loss * gather_weight
            if self.reduction == "mean":
                denom = gather_weight.sum()
                return loss.sum() / denom

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unsupported loss reduction {self.reduction!r}")

    # 执行 `NegativeLogLikelihoodLoss` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, input, target, weight=None):
        data = np.asarray(input.data)
        labels = np.asarray(target.data)
        expected_target_shape = self._target_shape(data.shape)
        if labels.shape != expected_target_shape:
            raise ValueError(f"Target shape {labels.shape} does not match expected {expected_target_shape}")
        invalid = (labels < 0) | (labels >= data.shape[1])
        if self.ignore_index is not None:
            invalid = invalid & (labels != self.ignore_index)
        if np.any(invalid):
            raise ValueError(f"Target class is out of range [0, {data.shape[1]})")

        if (
            self.lib is not None
            and input.dtype in nn.DTYPE_MAP
            and target.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
            and (weight is None or weight.dtype in nn.DTYPE_MAP)
        ):
            out_shape = target.size if self.reduction == "none" else ()
            input_c = self._numpy_to_ctensor(np.ascontiguousarray(input.data), input.dtype)
            target_c = self._numpy_to_ctensor(np.ascontiguousarray(target.data), target.dtype)
            weight_c = self._numpy_to_ctensor(np.ascontiguousarray(weight.data), weight.dtype) if weight is not None else None
            output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
            output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
            self.lib.negative_log_likelihood_loss_forward(
                input_c,
                target_c,
                weight_c,
                output_c,
                ctypes.c_int(self._reduction_code()),
                ctypes.c_int(1 if self.ignore_index is not None else 0),
                ctypes.c_int64(0 if self.ignore_index is None else int(self.ignore_index)),
            )
            out_data = self._ctensor_to_numpy(output_c, self.dtype)
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(target_c)
            if weight_c is not None:
                self.lib.free_tensor(weight_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_shape, dtype=self.dtype, data=out_data), "parameters": None}

        loss = self._gather_negative_scores(data, labels, self.ignore_index)
        reduced = self._reduce(loss, labels, weight)
        out_data = np.asarray(reduced, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `NegativeLogLikelihoodLoss` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, input, target, weight=None):
        out_shape = target.size if self.reduction == "none" else ()
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}


class SoftmaxCrossEntropyLoss(NegativeLogLikelihoodLoss):
    # 初始化 `SoftmaxCrossEntropyLoss` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, reduction="mean", ignore_index=None, dtype="float32", version="17"):
        super().__init__(inputs, outputs, reduction=reduction, ignore_index=ignore_index, dtype=dtype, version=version)
        if self.lib:
            self.lib.softmax_cross_entropy_loss_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int64,
            ]

    # 封装 `_log_softmax` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _log_softmax(scores):
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

    # 执行 `SoftmaxCrossEntropyLoss` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, scores, labels, weights=None):
        data = np.asarray(scores.data)
        target = np.asarray(labels.data)
        expected_target_shape = self._target_shape(data.shape)
        if target.shape != expected_target_shape:
            raise ValueError(f"Label shape {target.shape} does not match expected {expected_target_shape}")
        invalid = (target < 0) | (target >= data.shape[1])
        if self.ignore_index is not None:
            invalid = invalid & (target != self.ignore_index)
        if np.any(invalid):
            raise ValueError(f"Target class is out of range [0, {data.shape[1]})")
        out_dtype = nn.DTYPE_TO_NUMPY.get(self.dtype, data.dtype)
        if (
            self.lib is not None
            and scores.dtype in nn.DTYPE_MAP
            and labels.dtype in nn.DTYPE_MAP
            and self.dtype in nn.DTYPE_MAP
            and (weights is None or weights.dtype in nn.DTYPE_MAP)
        ):
            loss_shape = labels.size if self.reduction == "none" else ()
            scores_c = self._numpy_to_ctensor(np.ascontiguousarray(scores.data), scores.dtype)
            labels_c = self._numpy_to_ctensor(np.ascontiguousarray(labels.data), labels.dtype)
            weights_c = self._numpy_to_ctensor(np.ascontiguousarray(weights.data), weights.dtype) if weights is not None else None
            loss_shape_c = (ctypes.c_int * len(loss_shape))(*loss_shape)
            loss_c = self.lib.create_tensor(loss_shape_c, len(loss_shape), nn.DTYPE_MAP[self.dtype])
            log_c = None
            want_log_prob = len(self.outputs) > 1 and self.outputs[1]
            if want_log_prob:
                log_shape_c = (ctypes.c_int * len(scores.size))(*scores.size)
                log_c = self.lib.create_tensor(log_shape_c, len(scores.size), nn.DTYPE_MAP[self.dtype])
            self.lib.softmax_cross_entropy_loss_forward(
                scores_c,
                labels_c,
                weights_c,
                loss_c,
                log_c,
                ctypes.c_int(self._reduction_code()),
                ctypes.c_int(1 if self.ignore_index is not None else 0),
                ctypes.c_int64(0 if self.ignore_index is None else int(self.ignore_index)),
            )
            loss_data = self._ctensor_to_numpy(loss_c, self.dtype)
            loss_tensor = Tensor(*loss_shape, dtype=self.dtype, data=loss_data)
            log_tensor = None
            if want_log_prob:
                log_data = self._ctensor_to_numpy(log_c, self.dtype)
                log_tensor = Tensor(*scores.size, dtype=self.dtype, data=log_data)
            self.lib.free_tensor(scores_c)
            self.lib.free_tensor(labels_c)
            if weights_c is not None:
                self.lib.free_tensor(weights_c)
            self.lib.free_tensor(loss_c)
            if log_c is not None:
                self.lib.free_tensor(log_c)
            if want_log_prob:
                return {"tensor": (loss_tensor, log_tensor), "parameters": None}
            return {"tensor": loss_tensor, "parameters": None}

        log_prob = self._log_softmax(data)
        loss = self._gather_negative_scores(log_prob, target, self.ignore_index)
        reduced = self._reduce(loss, target, weights)
        loss_tensor = Tensor(*np.asarray(reduced).shape, dtype=self.dtype, data=np.asarray(reduced, dtype=out_dtype))
        if len(self.outputs) > 1 and self.outputs[1]:
            log_tensor = Tensor(*log_prob.shape, dtype=self.dtype, data=log_prob.astype(out_dtype, copy=False))
            return {"tensor": (loss_tensor, log_tensor), "parameters": None}
        return {"tensor": loss_tensor, "parameters": None}

    # 执行 `SoftmaxCrossEntropyLoss` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, scores, labels, weights=None):
        loss_shape = labels.size if self.reduction == "none" else ()
        loss_tensor = Tensor_(*loss_shape, dtype=self.dtype)
        if len(self.outputs) > 1 and self.outputs[1]:
            return {"tensor": (loss_tensor, Tensor_(*scores.size, dtype=self.dtype)), "parameters": None}
        return {"tensor": loss_tensor, "parameters": None}


class Unique(Ops):
    # 初始化 `Unique` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, axis=None, sorted=1, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.axis = axis
        self.sorted = sorted
        self.dtype = dtype
        self.version = version
        if self.lib:
            self.lib.unique_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
            ]
            self.lib.unique_forward.restype = ctypes.c_int

    # 封装 `_reorder_unique` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _reorder_unique(self, values, indices, inverse, counts, axis=None):
        if self.sorted:
            return values, indices, inverse, counts
        order = np.argsort(indices)
        remap = np.empty_like(order)
        remap[order] = np.arange(order.size)
        if axis is None:
            values = values[order]
        else:
            values = np.take(values, order, axis=axis)
        return values, indices[order], remap[inverse], counts[order]

    # 封装 `_compute` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _compute(self, x):
        data = x.data
        if self.axis is None:
            values, indices, inverse, counts = np.unique(
                data.reshape(-1), return_index=True, return_inverse=True, return_counts=True
            )
            values, indices, inverse, counts = self._reorder_unique(values, indices, inverse, counts)
        else:
            axis = self.axis if self.axis >= 0 else self.axis + data.ndim
            values, indices, inverse, counts = np.unique(
                data, axis=axis, return_index=True, return_inverse=True, return_counts=True
            )
            values, indices, inverse, counts = self._reorder_unique(values, indices, inverse, counts, axis=axis)
        return (
            values.astype(nn.DTYPE_TO_NUMPY.get(self.dtype, values.dtype), copy=False),
            indices.astype(np.int64, copy=False),
            inverse.astype(np.int64, copy=False),
            counts.astype(np.int64, copy=False),
        )

    # 执行 `Unique` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        if self.axis is None and self.lib is not None and x.dtype in nn.DTYPE_MAP and self.dtype in nn.DTYPE_MAP:
            flat = np.ascontiguousarray(np.asarray(x.data).reshape(-1))
            n = int(flat.size)
            input_c = self._numpy_to_ctensor(flat, x.dtype)
            buffer_shape_c = (ctypes.c_int * 1)(n)
            values_c = self.lib.create_tensor(buffer_shape_c, 1, nn.DTYPE_MAP[self.dtype])
            indices_c = self.lib.create_tensor(buffer_shape_c, 1, nn.DTYPE_MAP["int64"])
            inverse_c = self.lib.create_tensor(buffer_shape_c, 1, nn.DTYPE_MAP["int64"])
            counts_c = self.lib.create_tensor(buffer_shape_c, 1, nn.DTYPE_MAP["int64"])
            unique_count = int(self.lib.unique_forward(
                input_c,
                values_c,
                indices_c,
                inverse_c,
                counts_c,
                ctypes.c_int(int(self.sorted)),
            ))
            values = self._ctensor_to_numpy(values_c, self.dtype)[:unique_count]
            indices = self._ctensor_to_numpy(indices_c, "int64")[:unique_count]
            inverse = self._ctensor_to_numpy(inverse_c, "int64")[:n]
            counts = self._ctensor_to_numpy(counts_c, "int64")[:unique_count]
            self.lib.free_tensor(input_c)
            self.lib.free_tensor(values_c)
            self.lib.free_tensor(indices_c)
            self.lib.free_tensor(inverse_c)
            self.lib.free_tensor(counts_c)
        else:
            values, indices, inverse, counts = self._compute(x)
        tensors = [
            Tensor(*values.shape, dtype=self.dtype, data=values),
            Tensor(*indices.shape, dtype="int64", data=indices),
            Tensor(*inverse.shape, dtype="int64", data=inverse),
            Tensor(*counts.shape, dtype="int64", data=counts),
        ]
        selected = [tensor for name, tensor in zip(self.outputs, tensors) if name]
        return {"tensor": selected[0] if len(selected) == 1 else tuple(selected), "parameters": None}

    # 执行 `Unique` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        if hasattr(x, "data") and x.data is not None:
            return self.forward(x)
        if self.axis is None:
            unique_dim = x.data_size
            inverse_dim = x.data_size
            value_shape = (unique_dim,)
        else:
            axis = self.axis if self.axis >= 0 else self.axis + len(x.size)
            unique_dim = x.size[axis]
            inverse_dim = x.size[axis]
            value_shape = tuple(x.size)
        tensors = [
            Tensor_(*value_shape, dtype=self.dtype),
            Tensor_(unique_dim, dtype="int64"),
            Tensor_(inverse_dim, dtype="int64"),
            Tensor_(unique_dim, dtype="int64"),
        ]
        selected = [tensor for name, tensor in zip(self.outputs, tensors) if name]
        return {"tensor": selected[0] if len(selected) == 1 else tuple(selected), "parameters": None}


class NonMaxSuppression(Ops):
    # 初始化 `NonMaxSuppression` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, center_point_box=0, dtype="int64", version="17"):
        super().__init__(inputs, outputs)
        self.center_point_box = center_point_box
        self.dtype = "int64"
        self.version = version
        if self.lib:
            self.lib.non_max_suppression_forward.argtypes = [
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.POINTER(CTensor),
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
            ]
            self.lib.non_max_suppression_forward.restype = ctypes.c_int

    # 执行 `NonMaxSuppression` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(
        self,
        boxes,
        scores,
        max_output_boxes_per_class=None,
        iou_threshold=None,
        score_threshold=None,
    ):
        boxes_data = boxes.data.astype(np.float32, copy=False)
        scores_data = scores.data.astype(np.float32, copy=False)
        if boxes_data.ndim != 3 or boxes_data.shape[2] != 4:
            raise ValueError(f"NonMaxSuppression boxes must have shape [batch, boxes, 4], got {boxes.size}")
        if scores_data.ndim != 3 or scores_data.shape[0] != boxes_data.shape[0] or scores_data.shape[2] != boxes_data.shape[1]:
            raise ValueError(f"NonMaxSuppression scores must have shape [batch, classes, boxes], got {scores.size}")

        max_output = 0 if max_output_boxes_per_class is None else int(max_output_boxes_per_class.data.item())
        iou = 0.0 if iou_threshold is None else float(iou_threshold.data.item())
        score_min = -np.inf if score_threshold is None else float(score_threshold.data.item())
        if (
            self.lib is not None
            and boxes.dtype in nn.DTYPE_MAP
            and scores.dtype in nn.DTYPE_MAP
            and max_output > 0
        ):
            max_rows = boxes_data.shape[0] * scores_data.shape[1] * min(max_output, boxes_data.shape[1])
            boxes_c = self._numpy_to_ctensor(np.ascontiguousarray(boxes_data.astype(nn.DTYPE_TO_NUMPY[boxes.dtype], copy=False)), boxes.dtype)
            scores_c = self._numpy_to_ctensor(np.ascontiguousarray(scores_data.astype(nn.DTYPE_TO_NUMPY[scores.dtype], copy=False)), scores.dtype)
            output_shape_c = (ctypes.c_int * 2)(max_rows, 3)
            output_c = self.lib.create_tensor(output_shape_c, 2, nn.DTYPE_MAP[self.dtype])
            count = int(self.lib.non_max_suppression_forward(
                boxes_c,
                scores_c,
                output_c,
                ctypes.c_int(max_output),
                ctypes.c_float(iou),
                ctypes.c_float(score_min),
                ctypes.c_int(int(self.center_point_box)),
            ))
            out_data = self._ctensor_to_numpy(output_c, self.dtype)[:count]
            self.lib.free_tensor(boxes_c)
            self.lib.free_tensor(scores_c)
            self.lib.free_tensor(output_c)
            return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

        selected = []
        if max_output > 0:
            for batch in range(scores_data.shape[0]):
                for cls in range(scores_data.shape[1]):
                    class_scores = scores_data[batch, cls]
                    candidate_indices = np.where(class_scores >= score_min)[0]
                    order = candidate_indices[np.argsort(-class_scores[candidate_indices], kind="mergesort")]
                    kept = []
                    for box_idx in order:
                        if len(kept) >= max_output:
                            break
                        box = boxes_data[batch, box_idx]
                        if all(_nms_iou(box, boxes_data[batch, kept_idx], self.center_point_box) <= iou for kept_idx in kept):
                            kept.append(int(box_idx))
                            selected.append([batch, cls, int(box_idx)])

        selected_arr = np.asarray(selected, dtype=np.int64).reshape(-1, 3)
        return {"tensor": Tensor(*selected_arr.shape, dtype=self.dtype, data=selected_arr), "parameters": None}

    # 执行 `NonMaxSuppression` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, boxes, scores, max_output_boxes_per_class=None, iou_threshold=None, score_threshold=None):
        if hasattr(boxes, "data") and boxes.data is not None and hasattr(scores, "data") and scores.data is not None:
            return self.forward(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
        if max_output_boxes_per_class is not None and hasattr(max_output_boxes_per_class, "data") and max_output_boxes_per_class.data is not None:
            max_output = int(max_output_boxes_per_class.data.item())
        else:
            max_output = 0
        first_dim = int(scores.size[0] * scores.size[1] * max_output) if len(scores.size) >= 2 else 0
        return {"tensor": Tensor_(first_dim, 3, dtype=self.dtype), "parameters": None}


class Einsum(Ops):
    # 初始化 `Einsum` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(self, inputs, outputs, equation, dtype="float32", version="17"):
        super().__init__(inputs, outputs)
        self.equation = equation
        self.dtype = dtype
        self.version = version
        
        if self.lib:
            self.lib.einsum_forward.argtypes = [
                ctypes.POINTER(ctypes.POINTER(CTensor)), ctypes.c_int, ctypes.POINTER(CTensor),
                ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
            ]

    # 封装 `_parse_equation` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _parse_equation(self, shapes):
        equation = self.equation.replace(" ", "")
        if "->" in equation:
            lhs, rhs = equation.split("->")
            input_labels, output_labels, ellipsis_labels = self._expand_ellipsis_labels(lhs.split(","), shapes, rhs)
        else:
            input_labels, output_labels, ellipsis_labels = self._expand_ellipsis_labels(equation.split(","), shapes, None)
        if len(input_labels) != len(shapes):
            raise ValueError(f"Einsum: Equation expects {len(input_labels)} inputs, got {len(shapes)}")
        if len(set(output_labels)) != len(output_labels):
            raise ValueError(f"Einsum: Output labels cannot repeat: {output_labels}")
            
        # 收集所有唯一标签及其维度大小；ellipsis 标签允许广播，显式标签必须维度完全一致。
        unique_labels = sorted(list(set("".join(input_labels) + output_labels)))
        unique_labels = [l for l in unique_labels if l.strip()] # 去除空格
        
        label_to_dim = {}
        for i, labels in enumerate(input_labels):
            labels = labels.strip()
            shape = shapes[i]
            if len(labels) != len(shape):
                raise ValueError(f"Einsum: Labels {labels} mismatch shape {shape}")
            for l, dim in zip(labels, shape):
                dim = int(dim)
                if l not in label_to_dim:
                    label_to_dim[l] = dim
                    continue
                if l in ellipsis_labels:
                    old_dim = label_to_dim[l]
                    if old_dim == dim or dim == 1:
                        continue
                    if old_dim == 1:
                        label_to_dim[l] = dim
                        continue
                if label_to_dim[l] != dim:
                    raise ValueError(f"Einsum: Label {l!r} has inconsistent dimensions {label_to_dim[l]} and {dim}")

        for label in output_labels:
            if label not in label_to_dim:
                raise ValueError(f"Einsum: Output label {label!r} does not appear in any input")
        
        # 生成 C 需要的 loop_limits
        loop_limits = [label_to_dim[l] for l in unique_labels]
        
        # 计算 Strides
        # 这是一个映射：Label -> Stride (在 Input X 中)
        # 如果 Label 不在 Input X 中，Stride = 0 (广播语义)
        
        # 实现 `get_tensor_strides` 步骤，规范化输入并返回下游期望的数据或元信息。
        def get_tensor_strides(shape):
            # 计算 contigous strides
            strides = []
            st = 1
            for d in reversed(shape):
                strides.append(st)
                st *= d
            return list(reversed(strides))

        input_strides_flat = []
        for i, labels in enumerate(input_labels):
            labels = labels.strip()
            native_strides = get_tensor_strides(shapes[i])
            # 映射到 unique_labels 顺序
            current_tensor_strides = []
            for u_label in unique_labels:
                if u_label in labels:
                    # 同一个输入中的重复标签表示取对角线，偏移步长需要累加所有匹配轴的 stride。
                    # ellipsis 展开的轴在原始维度为 1 时按广播处理，不参与实际地址递增。
                    stride = 0
                    for idx, label in enumerate(labels):
                        if label != u_label:
                            continue
                        if u_label in ellipsis_labels and int(shapes[i][idx]) == 1 and label_to_dim[u_label] != 1:
                            continue
                        stride += native_strides[idx]
                    current_tensor_strides.append(stride)
                else:
                    current_tensor_strides.append(0) # 广播/无关维度
            input_strides_flat.extend(current_tensor_strides)
            
        output_strides_flat = []
        native_out_strides = get_tensor_strides([label_to_dim[l] for l in output_labels])
        for u_label in unique_labels:
            if u_label in output_labels:
                idx = output_labels.index(u_label)
                output_strides_flat.append(native_out_strides[idx])
            else:
                output_strides_flat.append(0) # 归约维度
                
        # 计算输出形状
        out_shape = tuple([label_to_dim[l] for l in output_labels])
        
        return unique_labels, loop_limits, input_strides_flat, output_strides_flat, out_shape

    # 将官方 Einsum ellipsis 语法展开成内部单字符标签，便于复用 C 后端 stride planner。
    def _expand_ellipsis_labels(self, input_specs, shapes, output_spec):
        if len(input_specs) != len(shapes):
            raise ValueError(f"Einsum: Equation expects {len(input_specs)} inputs, got {len(shapes)}")
        used_labels = set("".join(input_specs) + (output_spec or ""))
        used_labels.discard(".")
        label_pool = [label for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" if label not in used_labels]

        parsed_inputs = []
        max_ellipsis_rank = 0
        for labels, shape in zip(input_specs, shapes):
            if labels.count("...") > 1:
                raise ValueError(f"Einsum: Operand {labels!r} contains multiple ellipses")
            if "..." not in labels:
                if "." in labels:
                    raise ValueError(f"Einsum: Invalid ellipsis syntax in operand {labels!r}")
                if len(labels) != len(shape):
                    raise ValueError(f"Einsum: Labels {labels} mismatch shape {shape}")
                parsed_inputs.append((labels, "", 0, False))
                continue
            prefix, suffix = labels.split("...")
            if "." in prefix + suffix:
                raise ValueError(f"Einsum: Invalid ellipsis syntax in operand {labels!r}")
            ellipsis_rank = len(shape) - len(prefix) - len(suffix)
            if ellipsis_rank < 0:
                raise ValueError(f"Einsum: Labels {labels} mismatch shape {shape}")
            max_ellipsis_rank = max(max_ellipsis_rank, ellipsis_rank)
            parsed_inputs.append((prefix, suffix, ellipsis_rank, True))

        if len(label_pool) < max_ellipsis_rank:
            raise ValueError("Einsum: Not enough internal labels to expand ellipsis")
        ellipsis_labels = "".join(label_pool[:max_ellipsis_rank])

        expanded_inputs = []
        for prefix, suffix, ellipsis_rank, has_ellipsis in parsed_inputs:
            if has_ellipsis:
                expanded_inputs.append(prefix + ellipsis_labels[max_ellipsis_rank - ellipsis_rank:] + suffix)
            else:
                expanded_inputs.append(prefix)

        if output_spec is None:
            counts = {}
            for label in "".join(expanded_inputs):
                if label in ellipsis_labels:
                    continue
                counts[label] = counts.get(label, 0) + 1
            return expanded_inputs, ellipsis_labels + "".join(sorted(label for label, count in counts.items() if count == 1)), set(ellipsis_labels)

        if output_spec.count("...") > 1:
            raise ValueError(f"Einsum: Output {output_spec!r} contains multiple ellipses")
        if "..." in output_spec:
            prefix, suffix = output_spec.split("...")
            if "." in prefix + suffix:
                raise ValueError(f"Einsum: Invalid ellipsis syntax in output {output_spec!r}")
            output_labels = prefix + ellipsis_labels + suffix
        else:
            if "." in output_spec:
                raise ValueError(f"Einsum: Invalid ellipsis syntax in output {output_spec!r}")
            output_labels = output_spec
        return expanded_inputs, output_labels, set(ellipsis_labels)

    # 封装 `_forward_ij_jk_to_ik` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _forward_ij_jk_to_ik(self, left, right):
        if len(left.size) != 2 or len(right.size) != 2:
            return None
        m, k = left.size
        k2, n = right.size
        if k != k2:
            raise ValueError(f"Einsum ij,jk->ik shape mismatch: {left.size} vs {right.size}")

        a = left.data.astype(np.float32, copy=False)
        b = right.data.astype(np.float32, copy=False)
        out = np.empty((m, n), dtype=np.float32)
        for i in range(m):
            for j in range(n):
                acc = np.float32(0.0)
                for kk in range(k):
                    acc = np.float32(acc + np.float32(a[i, kk] * b[kk, j]))
                out[i, j] = acc
        return out

    # 执行 `Einsum` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, *inputs):
        equation = self.equation.replace(" ", "")
        out_data = None
        if self.lib is not None and self.dtype in nn.DTYPE_MAP and all(x.dtype in nn.DTYPE_MAP for x in inputs):
            try:
                _labels, loop_limits, input_strides, output_strides, out_shape = self._parse_equation([x.size for x in inputs])
                input_ctensors = [self._numpy_to_ctensor(np.ascontiguousarray(x.data), x.dtype) for x in inputs]
                input_array = (ctypes.POINTER(CTensor) * len(input_ctensors))(*input_ctensors)
                output_shape_c = (ctypes.c_int * len(out_shape))(*out_shape)
                output_c = self.lib.create_tensor(output_shape_c, len(out_shape), nn.DTYPE_MAP[self.dtype])
                loop_limits_c = (ctypes.c_int * len(loop_limits))(*loop_limits)
                input_strides_c = (ctypes.c_int * len(input_strides))(*input_strides)
                output_strides_c = (ctypes.c_int * len(output_strides))(*output_strides)
                self.lib.einsum_forward(
                    input_array,
                    len(input_ctensors),
                    output_c,
                    len(loop_limits),
                    loop_limits_c,
                    input_strides_c,
                    output_strides_c,
                )
                out_data = self._ctensor_to_numpy(output_c, self.dtype)
                for c_tensor in input_ctensors:
                    self.lib.free_tensor(c_tensor)
                self.lib.free_tensor(output_c)
            except ValueError:
                out_data = None
        if out_data is None and equation == "ij,jk->ik" and len(inputs) == 2:
            out_data = self._forward_ij_jk_to_ik(inputs[0], inputs[1])
        if out_data is None:
            out_data = np.einsum(self.equation, *(x.data for x in inputs))
        out_data = np.asarray(out_data, dtype=nn.DTYPE_TO_NUMPY.get(self.dtype, out_data.dtype))
        return {"tensor": Tensor(*out_data.shape, dtype=self.dtype, data=out_data), "parameters": None}

    # 执行 `Einsum` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, *inputs):
        if not inputs:
            raise ValueError("Einsum requires at least one input")
        dummy_inputs = [np.empty(x.size, dtype=np.float32) for x in inputs]
        out_shape = np.einsum(self.equation, *dummy_inputs).shape
        return {"tensor": Tensor_(*out_shape, dtype=self.dtype), "parameters": None}
