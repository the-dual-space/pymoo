---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. _nb_mde:
```

.. meta::
   :description: An implementation of MDE (Multi-population based Differential Evolution), a differential evolution variant without control parameters for continuous single-objective optimization.

+++

.. meta::
   :keywords: MDE, Multi-population based Differential Evolution, Differential Evolution, Parameter-free, Single-objective Optimization, Python

+++

# MDE: Multi-population based Differential Evolution

+++

Multi-population based Differential Evolution (MDE) <cite data-cite="Karkinli2023"></cite> is a differential evolution variant that has no algorithm-specific control parameters: there is no scale factor `F` and no crossover rate `CR` to tune.

The algorithm keeps a full population of `pop_size` (= 3N) individuals. In each generation, N = `pop_size // 3` distinct individuals are drawn at random to form a sub-population, which is evolved and written back to the full population:

- **Mutation**: the scale factor is redrawn for every vector as $|a - \text{rand}^b| \cdot \text{randn}^d$ with $a \in \{0, 1\}$, $b \in \{1, \dots, 10\}$, $d \in \{1, \dots, 5\}$, and the difference vector mixes, per variable, the best individual of the full population, random partners and the target vector.
- **Crossover**: a parameter-free crossover map with either one random threshold per vector or one per variable.
- **Noise**: a small multiplicative noise term derived from the sub-population of the previous generation.
- **Bounds**: variables that leave the bounds are re-sampled uniformly within `[xl, xu]`.
- **Selection**: greedy one-to-one replacement of each target by its trial vector.

Each generation costs N function evaluations (the initial population costs `pop_size`). Constraints are handled by the feasibility-first comparison used throughout pymoo.

+++

### Example

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.mde import MDE
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("rastrigin", n_var=10)

algorithm = MDE(pop_size=90)

res = minimize(problem,
               algorithm,
               ("n_evals", 30000),
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
```

### Constrained Optimization Example

```{code-cell} ipython3
from pymoo.algorithms.soo.nonconvex.mde import MDE
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("g1")

res = minimize(problem,
               MDE(),
               ("n_evals", 30000),
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s\nCV = %s" % (res.X, res.F, res.CV))
```

### API

```{raw-cell}
:raw_mimetype: text/restructuredtext

.. autoclass:: pymoo.algorithms.soo.nonconvex.mde.MDE
    :noindex:
    :no-undoc-members:
```

+++

### References

MDE was proposed in <cite data-cite="Karkinli2023"></cite>. The reference implementation (MATLAB and Python) is available at [github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm](https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm).

+++

### Acknowledgement

The MDE implementation in pymoo was contributed by the author of the algorithm, [Ahmet Emin Karkinli](https://github.com/Aekarkinli).
