from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp

from jaxmc.lattice import LatticeState

@jax.jit
def e_local_nn_1d(i:int, spins: jnp.array):
    L = jnp.size(spins)
    return - 1.0 * spins[i] * (spins[(i+L-1)%L] + spins[(i+1)%L])

@jax.jit
def energy(s: LatticeState, e_local: Callable) -> float:
    def body_fn(i:int, vals: tuple[jnp.ndarray, LatticeState], e_local: Callable):
        e, s = vals
        return (e + e_local(i, s.spins), s)

    init_vals = (jnp.zeros(1, dtype=jnp.float64), s)
    vals = jax.lax.fori_loop(0, s.L, partial(body_fn, e_local=e_local), init_vals)
    return 0.5 * vals[0]