"""Minimize the Rastrigin function with Multi-population based Differential Evolution (MDE)."""

from pymoo.algorithms.soo.nonconvex.mde import MDE
from pymoo.optimize import minimize
from pymoo.problems import get_problem

problem = get_problem("rastrigin", n_var=10)

algorithm = MDE(pop_size=90)

res = minimize(problem,
               algorithm,
               ("n_evals", 30000),
               seed=1,
               verbose=True)

print(f"Best solution found: \nX = {res.X}\nF = {res.F}")
