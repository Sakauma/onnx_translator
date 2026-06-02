"""文件功能：根据本地图结构生成可视化图，帮助观察算子节点和边的连接关系。
作者：Egor Izmaylov
时间：2026-06-02
"""

from graphviz import Digraph
import os

# 实现 `GraphGenerate` 步骤，规范化输入并返回下游期望的数据或元信息。
def GraphGenerate(input_graph, model_name, fast_mode=True):
    """
    生成计算图的可视化图形
    
    Args:
        input_graph: 自定义的 Graph 对象，包含 ops 字典和 input_name 列表
        model_name: 任务/模型名称，用于命名输出文件
        fast_mode: 是否开启极速渲染模式（牺牲布局质量换取速度）
    """
    mode_info = "极速渲染 (nslimit=2)" if fast_mode else "标准渲染"
    print(f"   [GraphViz] 正在构建图结构 (布局: Top-Down, 模式: {mode_info})...")
    
    dot = Digraph(comment=model_name)
    dot.attr(rankdir='TB') # Top-to-Bottom 布局
    
    if fast_mode:
        dot.attr(nslimit='2.0')
        dot.attr(mclimit='2.0')
        dot.attr(overlap='false', splines='true', sep='+10') 
    else:
        dot.attr(overlap='false', splines='true')

    # 设置默认节点样式
    dot.attr('node', shape='record', style='filled,rounded', fontname='Consolas', fontsize='12')

    # 筛选有效节点
    valid_nodes = []
    skipped_nodes = set()
    
    for node_name, op in input_graph.ops.items():
        op_type = op.__class__.__name__
        if op_type == 'Constant':
            skipped_nodes.add(node_name)
            continue
        valid_nodes.append(node_name)

    print(f"   (已过滤 {len(skipped_nodes)} 个常量节点，保留 {len(valid_nodes)} 个计算节点)")

    tensor_producer_map = {}
    for op_name in valid_nodes:
        op = input_graph.ops[op_name]
        
        outputs = op.outputs 
        
        for out_tensor in outputs:
            tensor_producer_map[str(out_tensor)] = op_name

    for node_name in valid_nodes:
        op = input_graph.ops[node_name]
        
        is_generic = (op.__class__.__name__ == 'GenericNode')
        op_type = op.op_type if is_generic else op.__class__.__name__

        info_str = ""
        try:
            if hasattr(op, "parameters") and isinstance(op.parameters, dict):
                if "info" in op.parameters: 
                    info_str = op.parameters["info"]
                elif "values" in op.parameters and "tensor" in op.parameters["values"]: 
                    t_data = op.parameters["values"]["tensor"]
                    if isinstance(t_data, list):
                        sizes = [str(t.size) for t in t_data]
                        info_str = "\\n".join([f"Out{i}: {s}" for i, s in enumerate(sizes)])
                    elif hasattr(t_data, "size"):
                        info_str = f"Out: {t_data.size}"
        except: 
            pass
        
        safe_name = node_name.replace('{', '\\{').replace('}', '\\}').replace('|', '\\|')
        safe_info = info_str.replace('{', '\\{').replace('}', '\\}').replace('|', '\\|')

        label = f"{{ {safe_name} | {op_type} }}"
        if safe_info: 
            label += f" | {safe_info} }}"

        if is_generic:
            # 降级节点：橙色、虚线
            dot.node(node_name, label=label, fillcolor='orange', style='filled,dashed')
        else:
            # 正常节点：有形状信息的标蓝，否则标灰
            fillcolor = 'lightblue' if safe_info else 'lightgrey'
            dot.node(node_name, label=label, fillcolor=fillcolor)
        
    drawn_edges = set()
    
    model_real_inputs = []
    if hasattr(input_graph, 'input_name'):
        if isinstance(input_graph.input_name, list):
            model_real_inputs = input_graph.input_name
        elif isinstance(input_graph.input_name, str):
            model_real_inputs = [input_graph.input_name]

    for op_name in valid_nodes:
        op = input_graph.ops[op_name]
        
        inputs = op.inputs
        
        for in_tensor in inputs:
            tensor_name = str(in_tensor)
            
            if tensor_name in tensor_producer_map:
                producer_name = tensor_producer_map[tensor_name]
                
                if producer_name in valid_nodes:
                    edge_key = (producer_name, op_name, tensor_name)
                    if edge_key not in drawn_edges:
                        dot.edge(producer_name, op_name)
                        drawn_edges.add(edge_key)
            else:
                is_key_input = (tensor_name in model_real_inputs)
                
                input_node_name = f"Input_{tensor_name}"
                input_node_name_clean = input_node_name.replace(':', '_').replace('[', '_').replace(']', '_').replace('.', '_')
                
                if input_node_name_clean not in drawn_edges:
                    if is_key_input:
                        dot.node(input_node_name_clean, label=f"{tensor_name}", shape='ellipse', fillcolor='lightgreen', style='filled', fontsize='12')
                    else:
                        dot.node(input_node_name_clean, label=f"{tensor_name}", shape='plain', fontcolor='gray50', fontsize='8')
                    
                    drawn_edges.add(input_node_name_clean)
                
                # 连线样式
                edge_color = "green" if is_key_input else "gray80"
                edge_style = "solid" if is_key_input else "dashed"
                edge_arrowsize = "1.0" if is_key_input else "0.5"
                
                dot.edge(input_node_name_clean, op_name, color=edge_color, style=edge_style, arrowsize=edge_arrowsize)

    # 保存与渲染
    save_dir = os.path.join("./result", model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
            
    file_path = os.path.join(save_dir, model_name + "_ops_graph")
    
    print(f"   [GraphViz] 正在渲染 PDF...")
    try:
        path = dot.render(file_path, format='svg', cleanup=True)
        print(f"✅ 可视化文件生成成功！请查看: {path}")
    except Exception as e:
        print(f"❌ 渲染失败: {e}")
        print("提示: 请确保系统已安装 Graphviz (apt install graphviz / brew install graphviz)")