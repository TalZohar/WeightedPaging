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

def check_monotonicity(requests, cache_state_1, cache_state_2) -> bool:
    """
    checks if cache_state_1 solution contains cache_state_2 solution
    """
    consistent_cache_state_1 = convert_cache_state_to_persistent_cache_state(requests, cache_state_1)
    consistent_cache_state_2 = convert_cache_state_to_persistent_cache_state(requests, cache_state_2)
    return all(s1.issuperset(s2) for s1, s2 in zip(consistent_cache_state_1, consistent_cache_state_2))

def check_monotonicity_differ_by_one(requests, cache_state_1, cache_state_2) -> bool:
    """
    checks if cache_state_1 solution contains cache_state_2 solution
    """
    # consistent_cache_state_1 = convert_cache_state_to_persistent_cache_state(requests, cache_state_1)
    # consistent_cache_state_2 = convert_cache_state_to_persistent_cache_state(requests, cache_state_2)
    consistent_cache_state_1 = cache_state_1
    consistent_cache_state_2 = cache_state_2
    for s1, s2 in zip(consistent_cache_state_1, consistent_cache_state_2):
        if len(s2 - s1) > 1:
            return False
    return True
