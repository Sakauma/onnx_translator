import onnx

s = onnx.defs.get_schema("LayerNormalization", 17)
print([(t.type_param_str, list(t.allowed_type_strs)) for t in s.type_constraints])
print(s.attributes["stash_type"].description)
