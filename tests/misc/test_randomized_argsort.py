"""``randomized_argsort`` must break ties by its seeded permutation alone.

The function shuffles with ``random_state`` and then sorts. If that sort is not
stable, the order of tied values also depends on the CPU's sort kernel, so the
same seed can rank tied solutions differently on different machines.
"""

import numpy as np

from pymoo.util.randomized_argsort import randomized_argsort

SEED = 1
# Large enough that NumPy uses its full (unstable) sort path rather than the
# insertion sort it applies to short arrays, with several groups of ties.
N_VALUES = 64
N_TIE_GROUPS = 4


def test_ties_keep_seeded_permutation_order():
    values = (np.arange(N_VALUES) % N_TIE_GROUPS).astype(float)
    # Boundary solutions get infinite crowding distance, so ties at inf are common.
    values[::7] = np.inf
    permutation = np.random.default_rng(SEED).permutation(N_VALUES)
    position_in_permutation = np.argsort(permutation)

    order = randomized_argsort(values, method="numpy", random_state=np.random.default_rng(SEED))

    expected = sorted(range(N_VALUES), key=lambda i: (values[i], position_in_permutation[i]))
    np.testing.assert_array_equal(order, expected)
