"""``split_by_feasibility`` must order tied keys by population index.

NumPy's default ``argsort`` (``kind="quicksort"``) is not stable, and NumPy 2.x
dispatches it to different SIMD kernels per CPU. Elements with equal keys can
therefore come back in a different order on different machines, so a seeded run
keeps different solutions on different CPUs. These tests assert the stable
contract directly: ascending key, and ties in original population order.
"""

import numpy as np

from pymoo.core.population import Population
from pymoo.core.survival import split_by_feasibility

# Large enough that NumPy uses its full (unstable) sort path rather than the
# insertion sort it applies to short arrays, with several groups of ties.
N_INDIVIDUALS = 64
N_TIE_GROUPS = 4


def stable_order(indices, keys):
    """Indices sorted by key, ties kept in ascending index order."""
    return np.array(sorted(indices, key=lambda i: (keys[i], i)))


def test_infeasible_ties_keep_population_order():
    index = np.arange(N_INDIVIDUALS)
    # Every third individual is feasible; the rest share N_TIE_GROUPS violation values.
    cv = np.where(index % 3 == 0, 0.0, 1.0 + (index % N_TIE_GROUPS)).reshape(-1, 1)
    pop = Population.new(F=np.zeros((N_INDIVIDUALS, 1)), CV=cv)

    feasible, infeasible = split_by_feasibility(pop, sort_infeas_by_cv=True)

    np.testing.assert_array_equal(feasible, index[index % 3 == 0])
    np.testing.assert_array_equal(infeasible, stable_order(index[index % 3 != 0], cv[:, 0]))


def test_feasible_ties_keep_population_order():
    index = np.arange(N_INDIVIDUALS)
    f = (index % N_TIE_GROUPS).astype(float).reshape(-1, 1)
    cv = np.zeros((N_INDIVIDUALS, 1))
    pop = Population.new(F=f, CV=cv)

    feasible, infeasible = split_by_feasibility(pop, sort_infeas_by_cv=True, sort_feas_by_obj=True)

    np.testing.assert_array_equal(feasible, stable_order(index, f[:, 0]))
    assert len(infeasible) == 0
