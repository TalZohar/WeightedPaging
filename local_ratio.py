import copy
import dataclasses
import uuid
from collections import defaultdict
from typing import List

from matplotlib import pyplot as plt
from networkx.algorithms.traversal import dfs_edges


@dataclasses.dataclass
class Interval:
    start_time: int
    end_time: int
    weight: float
    page: int
    request: int
    id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4, init=False)

    @property
    def length(self):
        return self.end_time - self.start_time

    def contains(self, t: int):
        return self.start_time <= t <= self.end_time


@dataclasses.dataclass
class IntervalProblem:
    def __init__(self, time_horizon: int, width: int):
        self.time_horizon = time_horizon
        self.interval_list = []
        self.interval_per_time = [[] for i in range(time_horizon)]
        self.width: int = width
        self.start_cover = 0

    def add_interval(self, interval: Interval):
        self.interval_list.append(interval)
        for i in range(interval.start_time, interval.end_time + 1):
            self.interval_per_time[i].append(interval)

    def find_weightless_intervals(self, time: int):
        weightless_intervals = []
        for interval in self.interval_per_time[time]:
            if interval.weight == 0:
                weightless_intervals.append(interval)
        return weightless_intervals

    def find_non_weightless_intervals(self, time: int):
        non_weightless_intervals = []
        for interval in self.interval_per_time[time]:
            if interval.weight > 0:
                non_weightless_intervals.append(interval)
        return non_weightless_intervals

    def find_tau(self):
        """
        Tau is the time with the minimum excess
        :return: time
        """
        return min(range(self.time_horizon), key=lambda t: len(self.find_weightless_intervals(t)))


def translate_paging_to_interval_problem(requests, page_weights, k, start_cache_state) -> IntervalProblem:
    """
    Translate paging to intervals.
    Note, an interval begins a timestep after a request and ends a timestep before
    """
    time_horizon = len(requests)
    n = len(set(requests))
    interval_problem = IntervalProblem(time_horizon, n - k)
    last_occurrences = [0] * len(page_weights)
    for request_num, request_page in enumerate(requests):
        if last_occurrences[request_page] <= request_num - 1:
            interval_problem.add_interval(Interval(start_time=last_occurrences[request_page], end_time=request_num - 1,
                                                   weight=page_weights[request_page], page=request_page,
                                                   request=request_num))
        last_occurrences[request_page] = request_num + 1

    # add requests at the end
    for request_page in set(requests):
        if last_occurrences[request_page] <= time_horizon - 1:
            interval_problem.add_interval(Interval(start_time=last_occurrences[request_page], end_time=time_horizon - 1,
                                                   weight=page_weights[request_page], page=request_page,
                                                   request="end"))

    return interval_problem


def get_interval_sol_cost(interval_solution: List[Interval]):
    cost = 0
    for interval in interval_solution:
        cost += interval.weight
    return cost


def translate_interval_solution_to_paging_solution(interval_prob: IntervalProblem, interval_solution: List[Interval],
                                                   requests):
    sol_uuids = {interval.id for interval in interval_solution}
    cache_states = [set() for i in range(interval_prob.time_horizon)]
    for t in range(interval_prob.time_horizon):
        for interval in interval_prob.interval_per_time[t]:
            if interval.id not in sol_uuids:
                cache_states[t].add(interval.page)

        # add the request
        assert requests[t] not in cache_states[t]
        cache_states[t].add(requests[t])
    return cache_states


def solve_local_ratio(interval_prob: IntervalProblem, recursion_depth: int) -> List[Interval]:
    new_interval_prob = copy.deepcopy(interval_prob)
    tau = new_interval_prob.find_tau()
    plot_intervals_by_page(interval_prob.interval_list, save_file=f"[{recursion_depth}]tau{tau}.png")
    print(f"tau: {tau}")

    if len(interval_prob.find_weightless_intervals(tau)) >= interval_prob.width:
        # STOPPING CONDITION
        return []

    tau_intervals = new_interval_prob.find_non_weightless_intervals(tau)
    min_weight = min(tau_intervals, key=lambda interval: interval.weight).weight
    for interval in tau_intervals:
        interval.weight -= min_weight

    # find Q-intervals
    q_intervals = [interval for interval in tau_intervals if interval.weight == 0]
    returned_sol = solve_local_ratio(new_interval_prob, recursion_depth + 1)

    # Add q-intervals to solution as long as it is necessary
    for interval in q_intervals:
        # check if necessary:
        is_necessary = False
        for t in range(interval.start_time, interval.end_time + 1):
            sol_t_intervals = [sol_interval for sol_interval in returned_sol if sol_interval.contains(t)]
            if len(sol_t_intervals) + len(interval_prob.find_weightless_intervals(t)) < interval_prob.width:
                is_necessary = True

        if is_necessary:
            print(f"Adding interval {interval}")
            returned_sol.append(interval)

    return returned_sol


def plot_intervals_by_page(intervals: List[Interval], highlight_ids=None, save_file=None):
    """Plot a list of intervals, grouping them by the same page on the y-axis."""
    # Group intervals by page
    if highlight_ids is None:
        highlight_ids = set()
    page_groups = defaultdict(list)
    for interval in intervals:
        page_groups[interval.page].append(interval)

    fig, ax = plt.subplots()

    # Plot intervals page by page
    y_level = 0
    for page, page_intervals in sorted(page_groups.items()):
        for interval in page_intervals:
            # Check if the interval's UUID is in the highlight set
            if interval.id in highlight_ids:
                color = 'red'  # Highlight color
            else:
                color = 'skyblue'  # Regular color

            # Plot the interval as a horizontal bar
            ax.broken_barh([(interval.start_time, interval.length)], (y_level - 0.4, 0.8), facecolors=color)

            # Annotate the interval with its weight
            ax.text((interval.start_time + interval.end_time) / 2, y_level,
                    f'Weight: {interval.weight:.2f}\n',
                    va='center', ha='center', color='black', fontsize=8)

        y_level += 1  # Move to the next level for the next page group

    # Labeling
    ax.set_xlabel('Time')
    ax.set_ylabel('Pages')
    ax.set_title('Intervals grouped by page as a function of time')
    ax.set_yticks([])  # Hide default y-ticks since we're manually labeling the pages
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()
    plt.clf()
    plt.close()


def main():
    requests = [4, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4]
    page_weights = [3, 2, 3, 3, 3]
    start_cache = [0, 2, 3, 4]
    k = 4
    interval_problem = translate_paging_to_interval_problem(requests, page_weights, k, start_cache)
    sol = solve_local_ratio(interval_problem, 0)
    print(translate_interval_solution_to_paging_solution(interval_problem, sol, requests))
    sol_uuids = {interval.id for interval in sol}
    plot_intervals_by_page(interval_problem.interval_list, sol_uuids)


if __name__ == '__main__':
    main()
