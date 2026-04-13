import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(data):
    # Convert torch_geometric.data.Data to networkx.Graph
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.t().tolist())

    # # Set node attributes (e.g., node features)
    # nx.set_node_attributes(G, {i: {'feature': data.x[i].tolist()} for i in range(data.num_nodes)})

    # # Set edge attributes (e.g., edge features)
    # if data.edge_attr is not None:
    #     nx.set_edge_attributes(G, {tuple(edge): data.edge_attr[idx].tolist() for idx, edge in enumerate(data.edge_index.t().tolist())})

    # 调整图形的尺寸
    plt.figure(figsize=(20, 2))  # 宽度为10英寸，高度为8英寸
    # Plot the graph
    pos = {i: (data.x[i][0].item(), data.x[i][1].item()) for  i in range(data.num_nodes)}
    nx.draw(G, pos, with_labels=False, 
            node_color=[data.x[i][-3].item()  for i in range(data.num_nodes)], alpha = 0.5,   #-1--global index    -2--if ev   -3----time index
            node_shape = '>', node_size = 100, linewidths = 5,
            width = 2.5, edge_color = [data.x[i][-2].item()  for i in range(data.num_edges)] )
    
    # nx.draw_networkx_labels(G, pos)
    
#     plt.show()
    plt.savefig("/home/chwei/AutoVehicle_DataAndOther/myData/RESULTS_2024_2025/others/vector_input_4.svg")
    return 1
    


