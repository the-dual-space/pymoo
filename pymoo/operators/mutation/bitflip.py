"""Bitflip mutation operator for binary variables."""

import numpy as np

from pymoo.core.mutation import Mutation


class BitflipMutation(Mutation):
    r"""Bitflip mutation operator for binary chromosomes.

    Flips bits in binary genotypes with an independent probability per locus.

    In evolutionary computation literature, the mutation probability (:math:`p_m`)
    almost universally refers to the per-variable (per-bit) flip rate (typically
    :math:`1 / n_{\text{var}}`). In pymoo, mutation operates on two levels:

    1. **Chromosome level** (``prob``): The probability that an offspring individual
       is mutated at all. Defaults to ``1.0`` (every offspring is mutated).
    2. **Gene level** (``prob_var``): The probability that each individual bit is flipped.
       Defaults to ``min(0.5, 1 / n_var)``.

    .. warning::
        **Compound probability trap:** Do not pass your per-locus mutation rate
        (e.g. :math:`1/n`) as the first positional argument ``prob`` without specifying
        ``prob_var``:

        .. code-block:: python

            # WRONG: The effective per-bit rate degrades to
            #   prob * min(0.5, 1/n_var) = (1/n_var) * min(0.5, 1/n_var)
            # For n_var >= 2 this equals 1/n_var^2, which is near zero.
            mutation = BitflipMutation(1.0 / n_var)

            # CORRECT: Every offspring is mutated with an independent 1/n per-bit rate:
            mutation = BitflipMutation(prob=1.0, prob_var=1.0 / n_var)

            # ALSO CORRECT (relies on the default prob_var = min(0.5, 1/n_var)):
            mutation = BitflipMutation(prob=1.0)

    Args:
        prob: Probability that an individual undergoes the bitflip mutation. In canonical
            genetic algorithms this is ``1.0`` so that every offspring is subject to
            per-locus flipping.
        prob_var: Probability of flipping each individual bit. If ``None``, it defaults to
            ``min(0.5, 1.0 / problem.n_var)``. For ``n_var >= 2`` this equals ``1 / n_var``,
            i.e. one flipped bit per individual in expectation.
        **kwargs: Additional keyword arguments passed to the parent class.

    Examples:
        >>> from pymoo.operators.mutation.bitflip import BitflipMutation
        >>> # Standard GA configuration (one flipped bit per chromosome in expectation):
        >>> mutation = BitflipMutation(prob=1.0)
        >>>
        >>> # Explicit per-locus rate (e.g. 5% flip probability per bit):
        >>> mutation = BitflipMutation(prob=1.0, prob_var=0.05)
    """

    def __init__(self, prob=1.0, prob_var=None, **kwargs):
        super().__init__(prob=prob, prob_var=prob_var, **kwargs)

    def _do(self, problem, X, random_state=None, **kwargs):
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        Xp = np.copy(X)
        flip = random_state.random(X.shape) < prob_var
        Xp[flip] = ~X[flip]
        return Xp


class BFM(BitflipMutation):
    pass
