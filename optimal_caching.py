import networkx as nx
import matplotlib.pyplot as plt

LARGE = 10000000000


def generate_graph_from_requests(requests, page_weights, k, draw=False):
    n = len(page_weights)
    requests = ["stam"] + requests

    G = nx.DiGraph()
    pos = 0
    G.add_nodes_from([(0, {"pos": (pos, 0.5)})])
    pos += 1

    x = 1
    current_k = k
    for request_num, request in enumerate(requests):

        if request_num == 0:
            G.add_edges_from(
                [(0, x + n + i, {"capacity": 1, "weight": 0}) for i in range(k)])
        else:
            G.add_edges_from(
                [(x - n - k + j, x + i, {"capacity": 1, "weight": page_weights[i] + LARGE * (i != request) if i != j else 0}) for i in range(n) for j
                 in range(n)])
            G.add_edges_from(
                [(x - k + j, x + i, {"capacity": 1, "weight": page_weights[i] + LARGE * (i != request)}) for i in range(n) for j
                 in range(k)])

            G.add_edges_from(
                [(x - k - n + j, x + n + i, {"capacity": 1, "weight": LARGE * (request != "-")}) for i in range(k) for j
                 in range(n)])

            G.add_edges_from(
                [(x - k + i, x + n + i, {"capacity": 1, "weight": 0}) for i in range(k)])

        G.add_nodes_from([(x+i, {"pos": (pos, i/(n+k))}) for i in range(n+k)])
        pos += 1

        if request == "-":
            current_k -= 1
        if request == "+":
            current_k += 1

        x += n + k
        G.add_edges_from(
            [(x - k + i, x + n + i, {"capacity": 1, "weight": -LARGE if i < k - current_k else 0}) for i in range(k)])
        G.add_edges_from(
            [(x - n - k + i, x + i, {"capacity": 1, "weight": -LARGE if i == request else 0}) for i in
             range(n)])


        G.add_nodes_from([(x+i, {"pos": (pos, i/(n+k))}) for i in range(n+k)])
        pos += 1

        x += n + k

    G.add_edges_from(
        [(x - n - k + j, x, {"capacity": 1, "weight": 0}) for j in range(n+k)])
    G.add_nodes_from([(x, {"pos": (pos, 0.5)})])
    pos += 1

    # print(f"Generated {G}")
    # Draw the graph
    if draw:
        nx.draw(G, nx.get_node_attributes(G, "pos"), with_labels=True)
        # nx.draw_networkx_edge_labels(G, nx.get_node_attributes(G, "pos"))
        plt.show()



    # Display the graph
    return G, 0, x
