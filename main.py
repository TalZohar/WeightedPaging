import networkx as nx
import matplotlib.pyplot as plt
from random import randint
from itertools import chain

LARGE = 10000000
N = 5
K = 2
REQUEST_NUM = 10


def max_index(my_list):
    return my_list.index(max(my_list))


def generate_graph_from_requests(requests, page_weights, k, required_cache_states=None):
    n = len(page_weights)
    if not required_cache_states:
        required_cache_states = [{request} for request in requests]

    G = nx.DiGraph()
    pos = 0
    G.add_nodes_from([(0, {"pos": (pos, 0.5)})])
    pos += 1

    G.add_edges_from([(0, i + 1, {"capacity": 1, "weight": 0}) for i in range(k)])
    G.add_nodes_from([(i+1, {"pos": (pos, i/k)}) for i in range(k)])
    pos += 1
    x = k + 1
    for request_num, required_cache_state, request in zip(range(len(required_cache_states)), required_cache_states,
                                                          requests):
        assert (len(required_cache_state) <= k)
        if request_num == 0:
            G.add_edges_from(
                [(x - k + j, x + i, {"capacity": 1, "weight": 0 if i != j else 0}) for i in range(n) for j
                 in range(k)])
        else:
            G.add_edges_from(
                [(x - n + j, x + i, {"capacity": 1, "weight": page_weights[i] + LARGE * (i not in required_cache_state) if i != j else 0}) for i in range(n) for j
                 in range(n)])
        G.add_nodes_from([(x+i, {"pos": (pos, i/n)}) for i in range(n)])
        pos += 1

        x += n
        G.add_edges_from(
            [(x - n + i, x + i, {"capacity": 1, "weight": -LARGE if i in required_cache_state else 0}) for i in
             range(n)])
        G.add_nodes_from([(x+i, {"pos": (pos, i/n)}) for i in range(n)])
        pos += 1

        x += n

    G.add_edges_from(
        [(x - n + j, x, {"capacity": 1, "weight": 0}) for j in range(n)])
    G.add_nodes_from([(x, {"pos": (pos, 0.5)})])
    pos += 1

    # print(f"Generated {G}")
    # Draw the graph
    #nx.draw(G, nx.get_node_attributes(G, "pos"), with_labels=True)
    #nx.draw_networkx_edge_labels(G, nx.get_node_attributes(G, "pos"))

    # Display the graph
    plt.show()
    return G, 0, x


def get_solution_from_flow(G, mincostFlow, n, k):
    cache_states = []
    evictions = []
    x = k + 1 + n
    for request_num in range(len(requests)):
        if request_num < len(requests) - 1:
            evictions.append([{"new": max_index(list(mincostFlow[x + i].values())),
                               "evicted": i}
                              for i in range(n) if
                              sum(list(mincostFlow[x + i].values())) == 1 and list(mincostFlow[x + i].values())[i] != 1])
        cache_states.append([i for i in range(n) if sum(list(mincostFlow[x + i].values())) == 1])
        x += 2 * n
    return cache_states, evictions

m = 0
while True:
    page_weights = [2 ** i for i in range(N)]
    requests = [randint(0, N - 1) for i in range(REQUEST_NUM)]
    print(f"requests: {requests}")
    G, source, sink = generate_graph_from_requests(requests, page_weights, K)
    mincostFlow = nx.max_flow_min_cost(G, source, sink)
    mincost = nx.cost_of_flow(G, mincostFlow)
    print(f"mincost: {mincost}")
    cache_states, evictions = get_solution_from_flow(G, mincostFlow, N, K)
    print(f"cache state solution: {list(zip(requests,cache_states))}")
    print(f"evictions: {evictions}")
    print(sum([page_weights[x["new"]] for x in chain.from_iterable(evictions)]))

    for request_num, eviction in enumerate(evictions):
        if len(eviction) == 0:
            continue

        page_eviction = eviction[0]["evicted"]
        last_request = request_num
        while requests[last_request] != page_eviction:
            cache_states[last_request].remove(page_eviction)
            if last_request == 0:
                break
            last_request -= 1

    print(f"required: {list(zip(requests,cache_states))}")

    G, source, sink = generate_graph_from_requests(requests, page_weights, K + 1, cache_states)
    mincostFlow = nx.max_flow_min_cost(G, source, sink)
    mincost = nx.cost_of_flow(G, mincostFlow)
    print(f"mincost: {mincost}")
    cache_states, evictions = get_solution_from_flow(G, mincostFlow, N, K + 1)
    print(f"cache state solution: {list(zip(requests,cache_states))}")
    print(f"evictions: {evictions}")
    cost1 = sum([page_weights[x["new"]] for x in chain.from_iterable(evictions)])
    print(cost1)


    G, source, sink = generate_graph_from_requests(requests, page_weights, K + 1)
    mincostFlow = nx.max_flow_min_cost(G, source, sink)
    mincost = nx.cost_of_flow(G, mincostFlow)
    print(f"mincost: {mincost}")
    cache_states, evictions = get_solution_from_flow(G, mincostFlow, N, K + 1)
    print(f"cache state solution: {list(zip(requests,cache_states))}")
    print(f"evictions: {evictions}")
    cost2 = sum([page_weights[x["new"]] for x in chain.from_iterable(evictions)])
    print(cost2)
    if (cost2 == 0):
        continue
    if m < cost1 / cost2:
        m = cost1 / cost2
    if m > 1:
        break
    print(m)
    print("\n\n\n")