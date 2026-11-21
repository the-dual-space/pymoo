"""Multi-population based Differential Evolution (MDE).

References:
    [1] Karkinli, A.E. Detection of object boundary from point cloud by using multi-population based
        differential evolution algorithm. Neural Computing and Applications 35, 5193-5206 (2023).
        https://doi.org/10.1007/s00521-022-07969-w

The reference implementation (MATLAB and Python) is maintained by the author of the algorithm at
https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm.
"""

import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.algorithm import Algorithm
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.core.replacement import ImprovementReplacement
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util import default_random_state
from pymoo.util.display.single import SingleObjectiveOutput

# =========================================================================================================
# Operators
# =========================================================================================================


@default_random_state
def mde_scale(random_state=None):
    """Draw the random scale factor of one mutant vector (part of the mutation, Eqs. 7-14 in [1]).

    ``scale = |a - rand^b| * randn^d`` with ``a ~ U{0, 1}``, ``b ~ U{1, ..., 10}`` and ``d ~ U{1, ..., 5}``.
    """
    a = random_state.integers(0, 2)
    u = random_state.random()
    b = random_state.integers(1, 11)
    n = random_state.standard_normal()
    d = random_state.integers(1, 6)
    return abs(a - u**b) * n**d


@default_random_state
def mde_trial(B, best, noise, random_state=None):
    """Create the trial vectors of a sub-population (mutation + parameter-free crossover, Eqs. 7-14 in [1]).

    Args:
        B: Decision variables of the sub-population, shape ``(N, D)``.
        best: Decision variables of the best member of the full population, shape ``(D,)``.
        noise: Noise term carried over from the previous generation, shape ``(N, D)``.
        random_state: Random number generator.

    Returns:
        The (not yet bounded) trial vectors, shape ``(N, D)``.
    """
    N, D = B.shape
    temp = np.empty_like(B)

    for i in range(N):
        scale = mde_scale(random_state=random_state)

        # two distinct partners, both different from the target vector i
        while True:
            r1, r2 = random_state.choice(N, 2, replace=False)
            if r1 != i and r2 != i:
                break

        # per variable: elitist (best) or random component, and target or random component
        dx = np.where(random_state.random(D) < 0.5, B[r1], best)
        dy = np.where(random_state.random(D) < 0.5, B[i], B[r2])
        temp[i] = B[r2] + scale * (dx - dy)

    # parameter-free crossover map: either one random threshold per vector (c = 1) or one per variable (c = D)
    c = 1 if random_state.random() ** random_state.integers(1, 6) < 0.5 else D
    M = np.abs(random_state.integers(0, 2, (N, c)) - random_state.random((N, D)) ** random_state.integers(1, 6, (N, c)))
    M = M < 0.5

    return B + M * (temp + noise - B)


@default_random_state
def mde_noise(B, random_state=None):
    """Noise term derived from the updated sub-population (Eq. 13 in [1]).

    ``noise = B * 10^phi * (rand(N, 1) - 0.5)`` with ``phi ~ U{-12, ..., -9}`` drawn per entry.
    """
    N, D = B.shape
    return B * 10.0 ** random_state.integers(-12, -8, (N, D)) * (random_state.random((N, 1)) - 0.5)


@default_random_state
def mde_bounds_resample(X, xl, xu, random_state=None):
    """Re-sample every variable outside ``[xl, xu]`` uniformly within the bounds (Eq. 15 in [1])."""
    out = (X < xl) | (X > xu)
    if out.any():
        X = np.where(out, xl + random_state.random(X.shape) * (xu - xl), X)
    return X


# =========================================================================================================
# Implementation
# =========================================================================================================


class MDE(Algorithm):
    def __init__(
        self,
        pop_size=90,
        sampling=None,
        repair=None,
        output=None,
        **kwargs,
    ):
        """Multi-population based Differential Evolution (MDE).

        MDE [1] is a differential evolution variant without algorithm-specific control parameters:
        there is no scale factor F and no crossover rate CR to tune. The algorithm keeps a full
        population of ``pop_size`` (= 3N) individuals. In each generation, N = ``pop_size // 3``
        distinct individuals are drawn at random to form a sub-population, which is evolved by

        - a mutation whose scale factor is redrawn for every vector (``|a - rand^b| * randn^d``) and
          whose difference vector mixes, per variable, the best individual of the full population,
          random partners and the target vector,
        - a parameter-free crossover map (one threshold per vector or per variable, chosen at random),
        - a small multiplicative noise term derived from the sub-population of the previous generation,
        - re-sampling of variables that leave the bounds uniformly within ``[xl, xu]``,
        - greedy one-to-one replacement of each target by its trial vector.

        The improved sub-population is written back to the full population. Each generation thus
        costs N function evaluations (the initial population costs ``pop_size``).

        Constraints are handled by pymoo's feasibility-first comparison: in the greedy replacement a
        trial vector replaces its target if both are feasible and it has a smaller objective value,
        if it is feasible and the target is not, or if both are infeasible and it has a smaller
        constraint violation (see :class:`~pymoo.core.replacement.ImprovementReplacement`). The best
        individual used by the mutation is the first individual by constraint violation, then by
        objective value.

        Args:
            pop_size: Size of the full population (3N). Must be at least 9. A sub-population of
                ``pop_size // 3`` individuals is evolved, and thus evaluated, in each generation.
            sampling: Sampling strategy used to create the initial population. Defaults to
                :class:`~pymoo.operators.sampling.rnd.FloatRandomSampling` (uniform random sampling
                within the bounds), as in the original algorithm.
            repair: Optional repair operator applied to the trial vectors after the bound handling.
                Defaults to no repair.
            output: Output shown during the run when ``verbose=True``. Defaults to
                :class:`~pymoo.util.display.single.SingleObjectiveOutput`.
            **kwargs: Further arguments passed to :class:`~pymoo.core.algorithm.Algorithm`
                (for instance ``termination``, ``seed`` or ``save_history``).

        References:
            [1] Karkinli, A.E. Detection of object boundary from point cloud by using
            multi-population based differential evolution algorithm. Neural Computing and
            Applications 35, 5193-5206 (2023). https://doi.org/10.1007/s00521-022-07969-w
        """
        super().__init__(output=output if output is not None else SingleObjectiveOutput(), **kwargs)

        if pop_size < 9:
            raise ValueError(f"MDE requires pop_size >= 9 (a sub-population of at least 3), got {pop_size}.")

        self.pop_size = int(pop_size)
        self.n_sub = self.pop_size // 3

        self.repair = repair if repair is not None else NoRepair()
        sampling = sampling if sampling is not None else FloatRandomSampling()
        self.initialization = Initialization(sampling, repair=self.repair)
        self.survival = FitnessSurvival()

        # noise term carried from one generation to the next (Eq. 13)
        self.noise = None

    def _setup(self, problem, **kwargs):
        if problem.n_obj != 1:
            raise ValueError("MDE is a single-objective algorithm.")
        if not problem.has_bounds():
            raise ValueError("MDE requires a problem with lower and upper bounds (xl and xu).")

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.pop_size, algorithm=self, random_state=self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        self.survival.do(self.problem, self.pop, return_indices=True)
        self.noise = np.zeros((self.n_sub, self.problem.n_var))

    def _infill(self):
        random_state = self.random_state

        # draw the sub-population of N distinct individuals from the full population
        J = random_state.permutation(len(self.pop))[: self.n_sub]

        X = self.pop.get("X")
        best = X[np.argmin(self.pop.get("rank"))]

        T = mde_trial(X[J], best, self.noise, random_state=random_state)
        T = mde_bounds_resample(T, *self.problem.bounds(), random_state=random_state)

        off = Population.new(X=T)
        off = self.repair(self.problem, off, random_state=random_state)

        # remember which individual of the full population each trial vector competes with
        off.set("index", J)

        return off

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithm uses the AskAndTell interface thus infills must be provided."

        J = infills.get("index")

        # greedy one-to-one replacement with feasibility-first comparison, written back to the full population
        improved = ImprovementReplacement().do(self.problem, self.pop[J], infills, return_indices=True)
        self.pop[J[improved]] = infills[improved]

        # update the ranking (rank 0 is the best individual) of the full population
        self.survival.do(self.problem, self.pop, return_indices=True)

        # noise for the next generation is derived from the updated sub-population
        self.noise = mde_noise(self.pop[J].get("X"), random_state=self.random_state)

    def _set_optimum(self):
        k = self.pop.get("rank") == 0
        self.opt = self.pop[k]
