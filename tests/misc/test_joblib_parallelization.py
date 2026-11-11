import copy

import numpy as np
import pytest

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.parallelization.joblib import JoblibParallelization


pytest.importorskip("joblib")


class Sphere(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(n_var=3, n_obj=1, xl=-1, xu=1, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum(x**2)


def test_joblib_runner_can_be_deepcopied():
    runner = JoblibParallelization(n_jobs=2, prefer="threads")

    copied = copy.deepcopy(runner)

    assert copied.n_jobs == runner.n_jobs
    assert copied.kwargs == runner.kwargs


def test_joblib_runner_supports_saved_history():
    runner = JoblibParallelization(n_jobs=2, prefer="threads")
    problem = Sphere(elementwise_runner=runner)

    result = minimize(
        problem,
        GA(pop_size=6),
        ("n_gen", 2),
        seed=1,
        save_history=True,
        verbose=False,
    )

    assert len(result.history) == 2
