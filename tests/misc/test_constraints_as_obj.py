"""Tests for ConstraintsAsObjective problem transformation."""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.constraints.as_obj import ConstraintsAsObjective
from pymoo.util.normalization import normalize


class Quadratic(Problem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=1, n_ieq_constr=1, xl=-2, xu=2)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = (X + 1) ** 2
        out["G"] = 1 - X

    def _calc_pareto_front(self, *args, **kwargs):
        return np.array([[4.0]])


def test_constraints_as_objective_pareto_front_append_true():
    problem = ConstraintsAsObjective(Quadratic(), append=True)
    assert problem.n_obj == 2
    pf = problem.pareto_front()
    assert pf.shape == (1, 2)
    np.testing.assert_array_equal(pf, [[0.0, 4.0]])
    np.testing.assert_array_equal(problem.ideal_point(), [0.0, 4.0])
    np.testing.assert_array_equal(problem.nadir_point(), [0.0, 4.0])


def test_constraints_as_objective_pareto_front_append_false():
    problem = ConstraintsAsObjective(Quadratic(), append=False)
    assert problem.n_obj == 1
    pf = problem.pareto_front()
    assert pf.shape == (1, 1)
    np.testing.assert_array_equal(pf, [[0.0]])
    np.testing.assert_array_equal(problem.ideal_point(), [0.0])
    np.testing.assert_array_equal(problem.nadir_point(), [0.0])

    # Normalization should work without dimension mismatch
    values = problem.evaluate(
        np.array([[0.0], [1.0], [2.0]]),
        return_values_of=["F", "G", "H"],
        return_as_dictionary=True,
    )["F"].reshape(-1, 1)
    norm = normalize(values, xl=problem.ideal_point(), xu=problem.nadir_point())
    assert norm.shape == (3, 1)
