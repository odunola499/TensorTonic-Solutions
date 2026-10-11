def build_expression_graph(leaves: list, operations: list) -> tuple:
    """
    Returns a tuple of the node list and final node ID.
    """
    node_list = []
    node_data_dict = dict()

    for leave in leaves:
        node_list.append({
            "id": leave.get('id'),
            "data": float(leave['data']),
            "grad": float(0.0),
            "op":"",
            "parents":[]
        })
        node_data_dict[leave['id']] = float(leave['data'])

    for operation in operations:
        node = dict()
        id = operation['id']
        left = operation['left']
        right = operation['right']
        op = operation['op']
        node['id'] = id

        if op == "*":
            data = node_data_dict[left] * node_data_dict[right]
        elif op == "+":
            data = node_data_dict[left] + node_data_dict[right]
            
        node['data'] = data
        node['grad'] = 0.0
        node['op'] = op
        node['parents'] = [left, right]

        node_data_dict[id] = data
        node_list.append(node)
        
    final_id = node_list[-1]['id']
    return (node_list, final_id)
            
