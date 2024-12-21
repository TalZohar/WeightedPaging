import networkx as nx
import matplotlib.pyplot as plt
from random import randint
from itertools import chain

LARGE = 10000000
N = 6
K = 3
REQUEST_NUM = 10
RESULT_FILE = 'result.txt'

def max_index(my_list):
    return my_list.index(max(my_list))


def generate_graph_from_requests(requests, page_weights, k, required_cache_states=None):
    n = len(page_weights)
    if not required_cache_states:
        required_cache_states = [set() for _ in requests]

    required_cache_states = [current_required | {request} for current_required, request in
                             zip(required_cache_states, requests)]

    G = nx.DiGraph()
    pos = 0
    G.add_nodes_from([(0, {"pos": (pos, 0.5)})])
    pos += 1

    G.add_edges_from([(0, i + 1, {"capacity": 1, "weight": 0}) for i in range(k)])
    G.add_nodes_from([(i + 1, {"pos": (pos, i / k)}) for i in range(k)])
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
                [(x - n + j, x + i,
                  {"capacity": 1,
                   # Explanation: adding LARGE factor to change page only when necessary.
                   "weight": (page_weights[i] + LARGE * (i not in required_cache_state)) if i != j else 0})
                 # "weight": page_weights[i] if i != j else 0})
                 for i in range(n) for j in range(n)])
        G.add_nodes_from([(x + i, {"pos": (pos, i / n)}) for i in range(n)])
        pos += 1

        x += n
        G.add_edges_from(
            [(x - n + i, x + i, {"capacity": 1, "weight": -LARGE if i in required_cache_state else 0}) for i in
             range(n)])
        G.add_nodes_from([(x + i, {"pos": (pos, i / n)}) for i in range(n)])
        pos += 1

        x += n

    G.add_edges_from(
        [(x - n + j, x, {"capacity": 1, "weight": 0}) for j in range(n)])
    G.add_nodes_from([(x, {"pos": (pos, 0.5)})])
    pos += 1

    # print(f"Generated {G}")
    # Draw the graph
    # nx.draw(G, nx.get_node_attributes(G, "pos"), with_labels=True)
    # nx.draw_networkx_edge_labels(G, nx.get_node_attributes(G, "pos"))

    # Display the graph
    plt.show()
    return G, 0, x


def get_cache_states_from_flow(G, mincostFlow, n, k, request_len):
    cache_states = []
    x = k + 1 + n
    for request_num in range(request_len):
        cache_states.append({i for i in range(n) if sum(list(mincostFlow[x + i].values())) == 1})
        x += 2 * n
    return cache_states


def solve_weighted_paging(requests, page_weights, k, required_cache_states=None):
    G, source, sink = generate_graph_from_requests(requests, page_weights, k,
                                                   required_cache_states=required_cache_states)
    mincostFlow = nx.max_flow_min_cost(G, source, sink)
    mincost_val = nx.cost_of_flow(G, mincostFlow)
    paging_cost = (mincost_val % LARGE)
    cache_states = get_cache_states_from_flow(G, mincostFlow, len(page_weights), k, len(requests))
    return cache_states, paging_cost


def convert_cache_state_to_persistent_cache_state(requests, cache_states):
    """
    Converts cache state solution to contain only pages that remain in the cache between requests
    """
    new_cache_state = [set() for i in range(len(cache_states))]
    for page in set(requests):
        page_occurences = [i for i, x in enumerate(requests) if page == x]
        page_occurences = [0, *page_occurences, len(requests) - 1]
        for start, end in zip(page_occurences, page_occurences[1:]):
            preserve = True
            for state in cache_states[start:end + 1]:
                if page not in state:
                    preserve = False

            if preserve:
                for i in range(start, end + 1):
                    new_cache_state[i].add(page)
    return new_cache_state


def main():
    while True:
        page_weights = [2 ** i for i in range(N)]
        requests = [randint(0, N - 1) for i in range(REQUEST_NUM)]
        print(f"page_weight: {page_weights}")
        print(f"requests: {requests}")
        cs1_cache_states, cs1_paging_cost = solve_weighted_paging(requests, page_weights, K)
        print(f"k OPT cache_states {cs1_cache_states}")
        print(f"k OPT cost {cs1_paging_cost}")
        cs2_cache_states, cs2_paging_cost = solve_weighted_paging(requests, page_weights, K + 1)
        print(f"k+1 OPT cache_states {cs2_cache_states}")
        print(f"k+1 OPT cost {cs2_paging_cost}")
        persistent_cs1_cache_state = convert_cache_state_to_persistent_cache_state(requests, cs1_cache_states)
        print(f"persistent cache state {persistent_cs1_cache_state}")
        new_cs2_cache_states, new_cs2_paging_cost = solve_weighted_paging(requests, page_weights, K + 1,
                                                                          required_cache_states=persistent_cs1_cache_state)
        print(f"k+1 cache_states {new_cs2_cache_states}")
        print(f"k+1 cost {new_cs2_paging_cost}")
        if cs2_paging_cost != new_cs2_paging_cost:
            print(f"SUCCESS")
            with open(RESULT_FILE, 'a') as file:
                file.write("INCOMING RESULT\n")
                file.write(f"page_weight: {page_weights}\n")
                file.write(f"requests: {requests}\n")
                file.write(f"k OPT cache_states {cs1_cache_states}\n")
                file.write(f"k OPT cost {cs1_paging_cost}\n")
                file.write(f"k+1 OPT cache_states {cs2_cache_states}\n")
                file.write(f"k+1 OPT cost {cs2_paging_cost}\n")
                file.write(f"persistent cache state {persistent_cs1_cache_state}\n")
                file.write(f"k+1 cache_states {new_cs2_cache_states}\n")
                file.write(f"k+1 cost {new_cs2_paging_cost}\n\n")
if __name__ == '__main__':
    requests = [0, 4, 2, 3, 2, 0, 3]
    page_weights = [2 ** i for i in range(N)]
    states, cost = (solve_weighted_paging(requests, page_weights, K))
    states = convert_cache_state_to_persistent_cache_state(requests, states)
    print(states, cost)
    states, cost = (solve_weighted_paging(requests, page_weights, K + 1,
                                          required_cache_states=states))
    print(states, cost)
    states, cost = (solve_weighted_paging(requests, page_weights, K + 1))
    print(states, cost)
    main()
