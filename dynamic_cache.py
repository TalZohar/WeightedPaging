from copy import deepcopy

import networkx as nx

from optimal_caching import generate_graph_from_requests

L = 2
N = 9
PAGE_WEIGHTS = {i: L ** i for i in range(N)}
PAGE_WEIGHTS[1] = PAGE_WEIGHTS[0]


class CacheState:
    def __init__(self, init_state: set, k=None):
        self.state = init_state
        self.k = k or len(init_state)

    def get_free_slots(self):
        return self.k - len(self.state)

    def print(self):
        assert len(self.state) <= self.k
        print(f"k[{self.k}] state[{self.state}]", end=", ")


def max_index(my_list):
    return my_list.index(max(my_list))


def evict_min_alg(cache_state: CacheState, request):
    new_cache_state = deepcopy(cache_state)
    if len(new_cache_state.state) > 0:
        evict_page = min([(i, PAGE_WEIGHTS[i]) for i in new_cache_state.state], key=lambda x: x[1])[0]
    if request == "+":
        new_cache_state.k += 1
    elif request == "-":
        if new_cache_state.get_free_slots() > 0:
            new_cache_state.k -= 1
        else:
            new_cache_state.state.remove(evict_page)
            new_cache_state.k -= 1
    else:
        if request in new_cache_state.state:
            return new_cache_state

        if new_cache_state.get_free_slots() > 0:
            new_cache_state.state.add(request)
        else:
            new_cache_state.state.remove(evict_page)
            new_cache_state.state.add(request)
    return new_cache_state


def evict_max_alg(cache_state: CacheState, request):
    new_cache_state = deepcopy(cache_state)
    if len(new_cache_state.state) > 0:
        evict_page = max([(i, PAGE_WEIGHTS[i]) for i in new_cache_state.state], key=lambda x: x[1])[0]
    if request == "+":
        new_cache_state.k += 1
    elif request == "-":
        if new_cache_state.get_free_slots() > 0:
            new_cache_state.k -= 1
        else:
            new_cache_state.state.remove(evict_page)
            new_cache_state.k -= 1
    else:
        if request in new_cache_state.state:
            return new_cache_state

        if new_cache_state.get_free_slots() > 0:
            new_cache_state.state.add(request)
        else:
            new_cache_state.state.remove(evict_page)
            new_cache_state.state.add(request)
    return new_cache_state


def generate_adv_requests(alg, init_cache: CacheState, k):
    if k == 1:
        requests = [1, 0]
        new_cache = alg(init_cache, 1)
        new_new_cache = alg(new_cache, 0)

        return requests, [new_cache, new_new_cache]

    requests = [k]
    new_cache = alg(init_cache, k)
    cache_states = [new_cache]

    # if alg == evict_max_alg:
    #     requests += ["-"]
    #     new_cache = alg(init_cache, "-")
    #     cache_states += [new_cache]

    for i in range(L):
        rec_requests, rec_cache_states = generate_adv_requests(alg, new_cache, k - 1)
        requests += rec_requests
        cache_states += rec_cache_states
        new_cache = rec_cache_states[len(rec_cache_states) - 1]

    # if alg == evict_max_alg:
    #     requests += ["+"]
    #     new_cache = alg(init_cache, "+")
    #     cache_states += [new_cache]

    curr_cache = cache_states[len(cache_states) - 1]
    if k in curr_cache.state:
        for i in range(k):
            requests += ["-"]
            cache_states += [alg(curr_cache, "-")]
            curr_cache = cache_states[len(cache_states) - 1]

        for i in range(k):
            requests += ["+"]
            cache_states += [alg(curr_cache, "+")]
            curr_cache = cache_states[len(cache_states) - 1]

    curr_cache = cache_states[len(cache_states) - 1]
    new_cache = alg(curr_cache, k)
    requests += [k]
    cache_states += [new_cache]

    return requests, cache_states


def evaluate_cost(cache_states):
    cost = 0
    for i in range(1, len(cache_states)):
        new_pages = [page for page in cache_states[i].state if page not in cache_states[i - 1].state]
        cost += sum([PAGE_WEIGHTS[i] for i in new_pages])
    return cost


def get_solution(init_cache_state: CacheState, alg, requests):
    cache_states = [init_cache_state]
    for request in requests:
        curr_cache_state = cache_states[len(cache_states) - 1]
        cache_states += [alg(curr_cache_state, request)]
    return cache_states


def get_solution_from_flow(requests, mincostFlow, n, k):
    cache_states = []
    x = k + 1 + n
    curr_k = k
    for request_num, request in enumerate(requests):
        cache_states.append(CacheState({i for i in range(n) if sum(list(mincostFlow[x + i].values())) == 1}, curr_k))
        if request == "-":
            curr_k -= 1
        if request == "+":
            curr_k += 1
        x += 2 * (n + k)
    return cache_states


def get_optimal_solution(requests, k):
    G, source, sink = generate_graph_from_requests(requests, PAGE_WEIGHTS, k)
    mincostFlow = nx.max_flow_min_cost(G, source, sink)
    mincost = nx.cost_of_flow(G, mincostFlow)
    return get_solution_from_flow(requests, mincostFlow, len(PAGE_WEIGHTS), k)


def main():
    k = N - 1
    ALG = evict_min_alg
    OPT = evict_max_alg
    init_cache_state = CacheState(set(), k)
    requests, cache_states = generate_adv_requests(ALG, init_cache_state, k)
    cache_states = [init_cache_state] + cache_states
    print(f"Requests: {requests}")
    print(evaluate_cost(cache_states))
    alg_cost = evaluate_cost(get_solution(init_cache_state, ALG, requests))
    opt_cost = evaluate_cost(get_solution(init_cache_state, OPT, requests))
    real_opt = evaluate_cost(get_optimal_solution(requests, k))
    ratio1 = alg_cost / opt_cost
    print(f"ALG[{alg_cost}] OPT[{opt_cost}] REAL_OPT[{real_opt}] RATIO[{ratio1} REAL_OPT_RATIO[{alg_cost / real_opt}]")

    ALG = evict_max_alg
    OPT = evict_min_alg

    requests, cache_states = generate_adv_requests(ALG, init_cache_state, k)
    cache_states = [init_cache_state] + cache_states
    print(f"Requests: {requests}")
    print(evaluate_cost(cache_states))
    alg_cost = evaluate_cost(get_solution(init_cache_state, ALG, requests))
    opt_cost = evaluate_cost(get_solution(init_cache_state, OPT, requests))
    real_opt = evaluate_cost(get_optimal_solution(requests, k))

    [cache_state.print() for cache_state in get_solution(init_cache_state, ALG, requests)]
    print()
    [cache_state.print() for cache_state in get_solution(init_cache_state, OPT, requests)]
    print()
    [cache_state.print() for cache_state in get_optimal_solution(requests, k)]
    print()

    ratio2 = alg_cost / opt_cost
    print(f"ALG[{alg_cost}] OPT[{opt_cost}] REAL_OPT[{real_opt}] RATIO[{ratio2} REAL_OPT_RATIO[{alg_cost / real_opt}]")
    print(f"RATIO[{min(ratio1, ratio2)}]")


if __name__ == '__main__':
    main()
