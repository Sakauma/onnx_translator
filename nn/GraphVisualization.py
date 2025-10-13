from graphviz import Digraph


def GraphGenerate(input_graph, model_name):
    dot = Digraph()
    nodes_data = []
    for item in input_graph.ops.keys():
        nodes_data.append(item)
    temp_nodes = []

    for item in nodes_data:
        for i in input_graph.ops[item].inputs:
            temp_nodes.append(str(i))
        for i in input_graph.ops[item].outputs:
            temp_nodes.append(str(i))
    list(set(temp_nodes))

    for node_data in nodes_data:
        node_name = node_data
        if isinstance(input_graph.ops[node_name].parameters["values"]["tensor"], list):
            if len(input_graph.ops[node_name].parameters["values"]["tensor"]) == 2:
                label = (f'{{ input node: {input_graph.ops[node_name].inputs} | {node_name}| '
                         f'output node: {input_graph.ops[node_name].outputs}| '
                         f'output tensor: {[input_graph.ops[node_name].parameters["values"]["tensor"][0].size,input_graph.ops[node_name].parameters["values"]["tensor"][1].size]} }}')
                dot.node(node_name, label=label, shape='record', width='1.6', height='0.8', colorr="grey",
                         fontsize="12",
                         style='rounded',
                         fillcolor='lightblue', color='blue', fontname='Consolas')

            elif len(input_graph.ops[node_name].parameters["values"]["tensor"]) == 3:
                label = (f'{{ input node: {input_graph.ops[node_name].inputs} | {node_name}| '
                         f'output node: {input_graph.ops[node_name].outputs}| '
                         f'output tensor: {[input_graph.ops[node_name].parameters["values"]["tensor"][0].size, input_graph.ops[node_name].parameters["values"]["tensor"][1].size, input_graph.ops[node_name].parameters["values"]["tensor"][2].size]} }}')
                dot.node(node_name, label=label, shape='record', width='1.6', height='0.8', colorr="grey",
                         fontsize="12",
                         style='rounded',
                         fillcolor='lightblue', color='blue', fontname='Consolas')

            else:
                label = (f'{{ input node: {input_graph.ops[node_name].inputs} | {node_name}| '
                         f'output node: {input_graph.ops[node_name].outputs}| '
                         f'output tensor: {input_graph.ops[node_name].parameters["values"]["tensor"].size} }}')
                dot.node(node_name, label=label, shape='record', width='1.6', height='0.8', colorr="grey",
                         fontsize="12",
                         style='rounded',
                         fillcolor='lightblue', color='blue', fontname='Consolas')

    for node_name in temp_nodes:
        dot.node(node_name, shape='point', width='0.05', height='0.05', fontsize='8', fontname='Consolas')

    for item in nodes_data:
        node_name = item
        for i in input_graph.ops[item].inputs:
            dot.edge(str(i), node_name)
        for i in input_graph.ops[item].outputs:
            dot.edge(node_name, str(i))

    dot.attr(rankdir='TB')
    file_path = "./result/" + model_name + "/" + model_name + "_ops_graph"
    dot.render(file_path, format='png', cleanup=True)
