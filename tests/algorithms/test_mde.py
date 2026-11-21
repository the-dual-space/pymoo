"""Tests for Multi-population based Differential Evolution (MDE)."""

import numpy as np
import pytest

from pymoo.algorithms.soo.nonconvex.mde import MDE, mde_bounds_resample, mde_trial
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.single import Rastrigin, Sphere


class SphereWithConstraints(Problem):

    def __init__(self, n_var=10):
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=1, xl=0, xu=1, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(np.square(x - 0.5), axis=1)
        out["G"] = 0.1 - out["F"]


@pytest.mark.parametrize("seed", [1, 2])
def test_mde_sphere(seed):
    problem = Sphere(n_var=10)
    res = minimize(problem, MDE(), ("n_evals", 15000), seed=seed)
    np.testing.assert_allclose(res.F[0], 0.0, atol=1e-8)


def test_mde_rastrigin():
    problem = Rastrigin(n_var=5)
    res = minimize(problem, MDE(), ("n_evals", 20000), seed=1)
    assert res.F[0] < 1.0


def test_mde_same_seed_same_result():
    problem = get_problem("ackley", n_var=10)
    res1 = minimize(problem, MDE(), ("n_gen", 50), seed=7)
    res2 = minimize(problem, MDE(), ("n_gen", 50), seed=7)
    np.testing.assert_array_equal(res1.X, res2.X)
    np.testing.assert_array_equal(res1.pop.get("X"), res2.pop.get("X"))


def test_mde_different_seeds_differ():
    problem = get_problem("ackley", n_var=10)
    res1 = minimize(problem, MDE(), ("n_gen", 20), seed=1)
    res2 = minimize(problem, MDE(), ("n_gen", 20), seed=2)
    assert not np.array_equal(res1.X, res2.X)


@pytest.mark.parametrize("pop_size", [9, 90, 100])
def test_mde_evaluations_per_generation(pop_size):
    # the initial population costs pop_size evaluations, every further generation pop_size // 3
    problem = Sphere(n_var=5)
    n_gen = 25
    res = minimize(problem, MDE(pop_size=pop_size), ("n_gen", n_gen), seed=1)
    assert res.algorithm.evaluator.n_eval == pop_size + (n_gen - 1) * (pop_size // 3)
    assert len(res.pop) == pop_size


def test_mde_evaluation_budget():
    problem = Sphere(n_var=5)
    res = minimize(problem, MDE(pop_size=90), ("n_evals", 3090), seed=1)
    assert res.algorithm.evaluator.n_eval == 3090


def test_mde_pop_size_too_small():
    with pytest.raises(ValueError):
        MDE(pop_size=8)


def test_mde_requires_bounds():
    problem = Sphere(n_var=5)
    problem.xl, problem.xu = None, None
    with pytest.raises(ValueError):
        minimize(problem, MDE(), ("n_gen", 5), seed=1)


def test_mde_population_stays_within_bounds():
    problem = get_problem("rastrigin", n_var=10)

    def check(algorithm):
        X = algorithm.pop.get("X")
        assert np.all(X >= problem.xl) and np.all(X <= problem.xu)

    minimize(problem, MDE(), ("n_gen", 100), seed=1, callback=check)


def test_mde_constrained_sphere():
    problem = SphereWithConstraints()
    res = minimize(problem, MDE(), ("n_evals", 20000), seed=1)
    assert res.CV[0] <= 0.0
    np.testing.assert_allclose(res.F[0], 0.1, atol=1e-4)


def test_mde_constrained_g1():
    problem = get_problem("g1")
    res = minimize(problem, MDE(), ("n_evals", 30000), seed=1)
    assert res.CV[0] <= 0.0
    assert res.F[0] < -14.0


def test_mde_ask_and_tell():
    problem = Sphere(n_var=5)
    algorithm = MDE(pop_size=30)
    algorithm.setup(problem, termination=("n_gen", 30), seed=1)

    while algorithm.has_next():
        pop = algorithm.ask()
        algorithm.evaluator.eval(problem, pop)
        algorithm.tell(infills=pop)

    res = algorithm.result()
    ref = minimize(problem, MDE(pop_size=30), ("n_gen", 30), seed=1)
    np.testing.assert_array_equal(res.X, ref.X)


def test_mde_bounds_resample():
    rng = np.random.default_rng(1)
    xl, xu = np.full(4, -1.0), np.full(4, 2.0)
    X = np.array([[0.0, 3.0, -5.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    Y = mde_bounds_resample(X, xl, xu, random_state=rng)
    assert np.all((Y >= xl) & (Y <= xu))
    # variables inside the bounds are left untouched
    np.testing.assert_array_equal(Y[0, [0, 3]], X[0, [0, 3]])
    np.testing.assert_array_equal(Y[1], X[1])


def test_mde_trial_without_differences_is_identity():
    # if all members of the sub-population coincide with the best one, the trial vectors do too
    B = np.tile(np.array([0.3, -0.2, 0.7]), (6, 1))
    T = mde_trial(B, B[0], np.zeros_like(B), random_state=np.random.default_rng(1))
    np.testing.assert_allclose(T, B)
