# /**
#   ******************************************************************************
#   * @file        text.py
#   * @author      Egor Izmaylov
#   * @brief       按算子职责分组保存 `text` 相关 ONNX 算子实现。
#   * @details     2026.06.02  V1.0.0  创建
#   ******************************************************************************
#   * @attention
#   ******************************************************************************
# */

from .common import *
import re


def _as_string_array(value):
    return np.asarray(value.data, dtype=np.str_)


def _pad_split_lists(split_lists, padding_requirement):
    if isinstance(split_lists, list):
        return split_lists + ["" for _ in range(int(padding_requirement))]
    if isinstance(split_lists, np.ndarray):
        return [_pad_split_lists(item, pad) for item, pad in zip(split_lists, padding_requirement)]
    raise TypeError(f"Invalid split list type {type(split_lists)!r}")


def _split_with_padding(data, delimiter=None, maxsplit=None):
    separator = None if delimiter in (None, "") else delimiter
    split_lists = np.char.split(data.astype(np.str_), separator, maxsplit)
    num_splits = np.vectorize(len, otypes=[np.int64])(split_lists)
    padding_requirement = (np.max(num_splits, initial=0) - num_splits).tolist()
    padded = np.array(_pad_split_lists(split_lists, padding_requirement), dtype=object)
    if data.size == 0:
        padded = padded.reshape(*data.shape, 0)
    return padded.astype(np.str_), num_splits.astype(np.int64)


class RegexFullMatch(Ops):
    # 初始化 `RegexFullMatch` 的正则表达式 pattern，输出固定为 bool 张量。
    def __init__(self, inputs, outputs, pattern=None, version="20"):
        super().__init__(inputs, outputs)
        self.pattern = pattern
        self.dtype = "bool"
        self.version = version

    # 执行 `RegexFullMatch` 的字符串 fullmatch 语义，逐元素输出 bool 结果。
    def forward(self, x):
        if self.pattern is None:
            raise ValueError("RegexFullMatch requires a regex pattern attribute")
        try:
            regex = re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern {self.pattern!r}") from exc
        data = _as_string_array(x)
        matcher = np.vectorize(lambda item: regex.fullmatch(str(item)) is not None, otypes=[np.bool_])
        out_data = matcher(data)
        return {"tensor": Tensor(*out_data.shape, dtype="bool", data=out_data), "parameters": None, "graph": None}

    # 执行 `RegexFullMatch` 的形状推断路径，输出 shape 与输入一致。
    def forward_(self, x):
        return {"tensor": Tensor_(*x.size, dtype="bool"), "parameters": None, "graph": None}


class StringConcat(Ops):
    # 初始化 `StringConcat` 算子，记录字符串输出 dtype 和 opset 版本。
    def __init__(self, inputs, outputs, version="20"):
        super().__init__(inputs, outputs)
        self.dtype = "string"
        self.version = version

    # 执行 `StringConcat` 的逐元素字符串拼接，并遵循 NumPy/ONNX 多向广播。
    def forward(self, x, y):
        x_data = _as_string_array(x)
        y_data = _as_string_array(y)
        out_data = np.char.add(x_data, y_data).astype(np.str_)
        return {"tensor": Tensor(*out_data.shape, dtype="string", data=out_data), "parameters": None, "graph": None}

    # 执行 `StringConcat` 的形状推断路径，根据输入 shape 推导广播后 shape。
    def forward_(self, x, y):
        try:
            out_shape = np.broadcast_shapes(tuple(x.size), tuple(y.size))
        except ValueError:
            out_shape = x.size
        return {"tensor": Tensor_(*out_shape, dtype="string"), "parameters": None, "graph": None}


class StringSplit(Ops):
    # 初始化 `StringSplit` 的 delimiter 和 maxsplit 属性，输出为字符串张量与 int64 计数张量。
    def __init__(self, inputs, outputs, delimiter=None, maxsplit=None, version="20"):
        super().__init__(inputs, outputs)
        self.delimiter = delimiter
        self.maxsplit = None if maxsplit is None else int(maxsplit)
        self.dtype = "string"
        self.version = version

    # 执行 `StringSplit`，按官方语义将较短拆分结果用空字符串补齐。
    def forward(self, x):
        data = _as_string_array(x)
        split_data, counts = _split_with_padding(data, self.delimiter, self.maxsplit)
        return {
            "tensor": [
                Tensor(*split_data.shape, dtype="string", data=split_data),
                Tensor(*counts.shape, dtype="int64", data=counts),
            ],
            "parameters": None,
            "graph": None,
        }

    # 执行 `StringSplit` 的形状推断路径；真实 Tensor 可用时返回精确最后维，否则以 1 作为占位。
    def forward_(self, x):
        if isinstance(x, Tensor):
            result = self.forward(x)["tensor"]
            return {"tensor": [Tensor_(*result[0].size, dtype="string"), Tensor_(*result[1].size, dtype="int64")], "parameters": None}
        return {"tensor": [Tensor_(*x.size, 1, dtype="string"), Tensor_(*x.size, dtype="int64")], "parameters": None}


class StringNormalizer(Ops):
    # 初始化 `StringNormalizer` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(
        self,
        inputs,
        outputs,
        case_change_action="NONE",
        is_case_sensitive=0,
        locale="",
        stopwords=None,
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.case_change_action = case_change_action
        self.is_case_sensitive = bool(is_case_sensitive)
        self.locale = locale
        self.stopwords = list(stopwords or [])
        self.dtype = "string"
        self.version = version

    # 封装 `_strip_accents` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _strip_accents(text):
        try:
            text.encode("ASCII", errors="strict")
            return text
        except UnicodeEncodeError:
            normalized = unicodedata.normalize("NFKD", text)
            return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    # 封装 `_remove_stopwords` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    @staticmethod
    def _remove_stopwords(text, stops):
        return " ".join(token for token in text.split(" ") if token not in stops)

    # 封装 `_normalize_text` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _normalize_text(self, value):
        if isinstance(value, float) and np.isnan(value):
            return ""
        text = self._strip_accents(str(value))
        raw_stops = set(self.stopwords)
        if self.case_change_action == "LOWER":
            stops = {word.lower() for word in self.stopwords}
        elif self.case_change_action == "UPPER":
            stops = {word.upper() for word in self.stopwords}
        elif self.case_change_action == "NONE":
            stops = raw_stops
        else:
            raise ValueError(f"Unknown case_change_action {self.case_change_action!r}")

        if self.is_case_sensitive and raw_stops:
            text = self._remove_stopwords(text, raw_stops)
        if self.case_change_action == "LOWER":
            text = text.lower()
        elif self.case_change_action == "UPPER":
            text = text.upper()
        if not self.is_case_sensitive and stops:
            text = self._remove_stopwords(text, stops)
        return text

    # 执行 `StringNormalizer` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        data = np.asarray(x.data, dtype=np.str_)
        if data.ndim == 1:
            normalized = [self._normalize_text(value) for value in data.tolist()]
            normalized = [value for value in normalized if len(value) > 0]
            if not normalized:
                normalized = [""]
            out_data = np.asarray(normalized, dtype=np.str_)
        elif data.ndim == 2 and data.shape[0] == 1:
            normalized = [self._normalize_text(value) for value in data[0].tolist()]
            normalized = [value for value in normalized if len(value) > 0]
            if not normalized:
                normalized = [""]
            out_data = np.asarray([normalized], dtype=np.str_)
        else:
            raise ValueError(f"StringNormalizer expects shape [C] or [1, C], got {x.size}")
        return {"tensor": Tensor(*out_data.shape, dtype="string", data=out_data), "parameters": None}

    # 执行 `StringNormalizer` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        if isinstance(x, Tensor):
            return {"tensor": Tensor_(*self.forward(x)["tensor"].size, dtype="string"), "parameters": None}
        return {"tensor": Tensor_(*x.size, dtype="string"), "parameters": None}


class TfIdfVectorizer(Ops):
    # 初始化 `TfIdfVectorizer` 的构造参数，保存后续运行、形状推断或验证所需的状态。
    def __init__(
        self,
        inputs,
        outputs,
        mode,
        ngram_counts,
        ngram_indexes,
        max_skip_count,
        min_gram_length,
        max_gram_length,
        pool_int64s=None,
        pool_strings=None,
        weights=None,
        version="17",
    ):
        super().__init__(inputs, outputs)
        self.mode = mode
        self.ngram_counts = list(ngram_counts)
        self.ngram_indexes = list(ngram_indexes)
        self.max_skip_count = int(max_skip_count)
        self.min_gram_length = int(min_gram_length)
        self.max_gram_length = int(max_gram_length)
        self.pool_int64s = list(pool_int64s or [])
        self.pool_strings = list(pool_strings or [])
        self.weights = list(weights or [])
        self.dtype = "float32"
        self.version = version
        self._ngram_map = self._build_ngram_map()
        self.output_size = max(self.ngram_indexes) + 1 if self.ngram_indexes else 0

    # 封装 `_build_ngram_map` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _build_ngram_map(self):
        pool = self.pool_strings if self.pool_strings else self.pool_int64s
        ngram_map = {}
        ngram_id = 0
        for size_index, start in enumerate(self.ngram_counts):
            ngram_size = size_index + 1
            end = self.ngram_counts[size_index + 1] if size_index + 1 < len(self.ngram_counts) else len(pool)
            item_count = max(0, end - start)
            ngram_count = item_count // ngram_size if ngram_size > 0 else 0
            for idx in range(ngram_count):
                gram_start = start + idx * ngram_size
                gram = tuple(pool[gram_start:gram_start + ngram_size])
                if self.min_gram_length <= ngram_size <= self.max_gram_length and ngram_id < len(self.ngram_indexes):
                    ngram_map[gram] = self.ngram_indexes[ngram_id]
                ngram_id += 1
        return ngram_map

    # 封装 `_rows` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _rows(self, data):
        if data.ndim == 0:
            return [data.reshape(1)], False
        if data.ndim == 1:
            return [data], False
        if data.ndim == 2:
            if data.shape[0] < 1:
                raise ValueError("TfIdfVectorizer 2-D input must have B > 0")
            return [data[i] for i in range(data.shape[0])], True
        raise ValueError(f"TfIdfVectorizer expects scalar, 1-D, or 2-D input, got shape {data.shape}")

    # 封装 `_count_row` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _count_row(self, row):
        counts = np.zeros((self.output_size,), dtype=np.float32)
        if self.output_size == 0:
            return counts
        row_values = row.tolist()
        min_size = self.min_gram_length
        for skip_distance in range(1, self.max_skip_count + 2):
            start_size = min_size
            if skip_distance > 1 and start_size == 1:
                start_size = 2
            for start in range(len(row_values)):
                for ngram_size in range(start_size, self.max_gram_length + 1):
                    last = start + skip_distance * (ngram_size - 1)
                    if last >= len(row_values):
                        break
                    gram = tuple(row_values[start + skip_distance * offset] for offset in range(ngram_size))
                    out_idx = self._ngram_map.get(gram)
                    if out_idx is not None:
                        counts[out_idx] += 1.0
        return counts

    # 封装 `_apply_mode` 辅助逻辑，统一边界条件处理并保持调用方实现简洁。
    def _apply_mode(self, counts):
        mode = self.mode.upper()
        if mode == "TF":
            return counts
        if mode == "IDF":
            if self.weights:
                weights = np.asarray(self.weights, dtype=np.float32)
                return np.where(counts > 0, weights[:self.output_size], 0.0).astype(np.float32)
            return (counts > 0).astype(np.float32)
        if mode == "TFIDF":
            if self.weights:
                weights = np.asarray(self.weights, dtype=np.float32)
                return counts * weights[:self.output_size]
            return counts
        raise ValueError(f"Unsupported TfIdfVectorizer mode {self.mode!r}")

    # 执行 `TfIdfVectorizer` 的真实张量计算路径，读取输入数据并返回图运行器约定的结果结构。
    def forward(self, x):
        data = np.asarray(x.data)
        rows, batched = self._rows(data)
        vectors = [self._apply_mode(self._count_row(row)) for row in rows]
        out_data = np.stack(vectors, axis=0).astype(np.float32)
        if not batched:
            out_data = out_data.reshape((self.output_size,))
        return {"tensor": Tensor(*out_data.shape, dtype="float32", data=out_data), "parameters": None}

    # 执行 `TfIdfVectorizer` 的形状推断路径，只生成 `Tensor_` 元数据，不访问真实数值缓冲区。
    def forward_(self, x):
        if len(x.size) == 2:
            out_shape = (x.size[0], self.output_size)
        else:
            out_shape = (self.output_size,)
        return {"tensor": Tensor_(*out_shape, dtype="float32"), "parameters": None}
