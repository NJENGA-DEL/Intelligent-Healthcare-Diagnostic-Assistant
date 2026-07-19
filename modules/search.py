"""
modules/search.py

SEARCH ALGORITHMS MODULE
=========================
Covers: Week 3 (Uninformed & Informed Search)

This module is not disease-specific -- it's a general-purpose search
toolkit. It exists to serve two roles in the wider system:

  1. A clean, reusable backbone for Module 7 (planner.py)'s BFS state-space
     search, instead of that search logic being hardcoded inline there.

  2. CASE-BASED REASONING: given a new patient's symptom vector, search
     through past patient records (data/patient_records.csv, once populated)
     to find the most similar historical case(s) -- useful as a secondary,
     human-interpretable sanity check alongside the ML/Bayesian modules
     ("here are 3 past patients who looked like this one, and what they
     were diagnosed with").

Algorithms implemented:
    - Breadth-First Search (BFS)      -- uninformed, shortest # of steps
    - Depth-First Search (DFS)        -- uninformed, memory-light
    - Uniform Cost Search (UCS)       -- uninformed, cheapest total cost
    - A* Search                       -- informed, uses a heuristic
    - Nearest-neighbor symptom search -- similarity search over patients
"""

from collections import deque
import heapq
from typing import Callable, Dict, List, Optional, Set, Tuple, Any


# ---------------------------------------------------------------------------
# GENERIC STATE-SPACE SEARCH
# ---------------------------------------------------------------------------
# These functions are intentionally generic -- they don't know anything
# about "patients" or "diseases". They operate on:
#   - a `start` state
#   - a `goal_test(state) -> bool` function
#   - a `get_successors(state) -> List[Tuple[action, new_state, cost]]` function
#
# This mirrors how Module 7's planner already works internally (states are
# frozensets of facts, actions have preconditions/add/delete lists) but
# keeps the search algorithm itself reusable for ANY state-space problem,
# not just treatment planning.
# ---------------------------------------------------------------------------

def bfs(start: Any,
        goal_test: Callable[[Any], bool],
        get_successors: Callable[[Any], List[Tuple[str, Any, float]]]
        ) -> Optional[List[str]]:
    """
    Breadth-First Search.

    Explores states level by level -- guarantees the SHORTEST action
    sequence (fewest steps) to reach a goal state, assuming all actions
    have equal/unweighted cost. This is exactly the algorithm the manual
    specifies for Module 7's treatment planner.

    Returns the list of actions (the plan) to reach a goal state, or None
    if no plan exists.
    """
    frontier = deque([(start, [])])   # (current_state, actions_so_far)
    visited: Set[Any] = {start}

    while frontier:
        state, path = frontier.popleft()

        if goal_test(state):
            return path

        for action, next_state, _cost in get_successors(state):
            if next_state not in visited:
                visited.add(next_state)
                frontier.append((next_state, path + [action]))

    return None  # No plan found


def dfs(start: Any,
        goal_test: Callable[[Any], bool],
        get_successors: Callable[[Any], List[Tuple[str, Any, float]]],
        max_depth: int = 50
        ) -> Optional[List[str]]:
    """
    Depth-First Search.

    Explores as deep as possible down one path before backtracking.
    Uses far less memory than BFS for large state spaces, but does NOT
    guarantee the shortest plan -- useful when you just need *a* valid
    plan quickly, not necessarily the optimal one.

    `max_depth` guards against infinite recursion in cyclic state spaces.
    """
    visited: Set[Any] = set()

    def _dfs_recursive(state, path, depth):
        if depth > max_depth:
            return None
        if goal_test(state):
            return path
        visited.add(state)

        for action, next_state, _cost in get_successors(state):
            if next_state not in visited:
                result = _dfs_recursive(next_state, path + [action], depth + 1)
                if result is not None:
                    return result
        return None

    return _dfs_recursive(start, [], 0)


def uniform_cost_search(start: Any,
                         goal_test: Callable[[Any], bool],
                         get_successors: Callable[[Any], List[Tuple[str, Any, float]]]
                         ) -> Optional[Tuple[List[str], float]]:
    """
    Uniform Cost Search (a.k.a. Dijkstra's algorithm for search).

    Unlike BFS (fewest steps) or DFS (any path), UCS finds the plan with
    the LOWEST TOTAL COST -- relevant here because Module 7's actions each
    have a `cost` and `duration` (e.g. IsolatePatient costs more/takes
    longer than PrescribeAntiviral). If treatment plans should minimize
    total cost/time rather than just step count, this is the algorithm
    to use instead of plain BFS.

    Returns (action_list, total_cost), or None if no plan exists.
    """
    # Min-heap keyed by cumulative cost. A tie-breaking counter avoids
    # comparing (cost, state, path) tuples directly, since states/paths
    # may not be orderable.
    counter = 0
    frontier = [(0.0, counter, start, [])]
    best_cost: Dict[Any, float] = {start: 0.0}

    while frontier:
        cost_so_far, _, state, path = heapq.heappop(frontier)

        if goal_test(state):
            return path, cost_so_far

        # Skip stale entries -- a cheaper path to this state was already found.
        if cost_so_far > best_cost.get(state, float("inf")):
            continue

        for action, next_state, step_cost in get_successors(state):
            new_cost = cost_so_far + step_cost
            if new_cost < best_cost.get(next_state, float("inf")):
                best_cost[next_state] = new_cost
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, next_state, path + [action]))

    return None


def a_star_search(start: Any,
                   goal_test: Callable[[Any], bool],
                   get_successors: Callable[[Any], List[Tuple[str, Any, float]]],
                   heuristic: Callable[[Any], float]
                   ) -> Optional[Tuple[List[str], float]]:
    """
    A* Search.

    Like UCS, but uses a heuristic function h(state) estimating the
    remaining cost to the goal, exploring the most promising states first
    (lowest cost_so_far + heuristic). This is faster than UCS when a good
    heuristic is available.

    Example heuristic for the treatment planner: number of goal facts
    NOT yet satisfied in the current state (the more goal facts missing,
    the further from done). The heuristic must never OVERESTIMATE the
    true remaining cost, or A* can return a suboptimal plan.
    """
    counter = 0
    frontier = [(heuristic(start), counter, 0.0, start, [])]
    best_cost: Dict[Any, float] = {start: 0.0}

    while frontier:
        _priority, _, cost_so_far, state, path = heapq.heappop(frontier)

        if goal_test(state):
            return path, cost_so_far

        if cost_so_far > best_cost.get(state, float("inf")):
            continue

        for action, next_state, step_cost in get_successors(state):
            new_cost = cost_so_far + step_cost
            if new_cost < best_cost.get(next_state, float("inf")):
                best_cost[next_state] = new_cost
                counter += 1
                priority = new_cost + heuristic(next_state)
                heapq.heappush(frontier, (priority, counter, new_cost, next_state, path + [action]))

    return None


# ---------------------------------------------------------------------------
# CASE-BASED SIMILARITY SEARCH
# ---------------------------------------------------------------------------
# This is a different flavor of "search" -- not state-space planning, but
# finding the most similar past patient(s) to a new one. Useful as a
# transparent, human-readable cross-check: "3 similar past patients had
# this diagnosis" is easy for a doctor to sanity-check, unlike a neural
# network's raw confidence score.
# ---------------------------------------------------------------------------

def symptom_similarity(symptoms_a: List[str], symptoms_b: List[str]) -> float:
    """
    Jaccard similarity between two symptom sets:
        |intersection| / |union|

    Returns a value between 0.0 (no overlap) and 1.0 (identical symptom sets).
    Simple, fast, and easy to explain -- appropriate for a case-based
    reasoning sanity check rather than the primary diagnostic method.
    """
    set_a, set_b = set(symptoms_a), set(symptoms_b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def find_similar_patients(query_symptoms: List[str],
                           patient_records: List[Dict],
                           top_k: int = 3
                           ) -> List[Tuple[Dict, float]]:
    """
    Find the top_k most similar past patients to a new symptom set.

    `patient_records` is expected to be a list of dicts loaded from
    data/patient_records.csv, each containing at least a 'symptoms' list
    and a 'diagnosis' field.

    Returns a list of (patient_record, similarity_score) tuples, sorted
    from most to least similar.
    """
    scored = [
        (record, symptom_similarity(query_symptoms, record.get("symptoms", [])))
        for record in patient_records
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Test 1: BFS / DFS / UCS / A* on a tiny toy state-space ---
    # States are just integers 0..5, goal is to reach 5.
    # Each state can move to state+1 (cost 1) or state+2 (cost 2, if it exists).
    def get_successors_demo(state):
        successors = []
        if state + 1 <= 5:
            successors.append((f"+1 -> {state+1}", state + 1, 1))
        if state + 2 <= 5:
            successors.append((f"+2 -> {state+2}", state + 2, 2))
        return successors

    goal_test_demo = lambda s: s == 5
    heuristic_demo = lambda s: 5 - s  # steps remaining, never overestimates

    print("BFS plan :", bfs(0, goal_test_demo, get_successors_demo))
    print("DFS plan :", dfs(0, goal_test_demo, get_successors_demo))
    print("UCS plan :", uniform_cost_search(0, goal_test_demo, get_successors_demo))
    print("A* plan  :", a_star_search(0, goal_test_demo, get_successors_demo, heuristic_demo))

    # --- Test 2: symptom similarity / case-based search ---
    fake_patient_records = [
        {"patient_id": "P010", "symptoms": ["fever", "cough", "fatigue"], "diagnosis": "flu"},
        {"patient_id": "P011", "symptoms": ["fever", "cough", "loss_of_smell"], "diagnosis": "covid19"},
        {"patient_id": "P012", "symptoms": ["headache", "dizziness"], "diagnosis": "migraine"},
    ]
    new_patient_symptoms = ["fever", "cough", "fatigue", "loss_of_smell"]

    print("\nMost similar past patients:")
    for record, score in find_similar_patients(new_patient_symptoms, fake_patient_records, top_k=2):
        print(f"  {record['patient_id']} ({record['diagnosis']}): similarity={score:.2f}")