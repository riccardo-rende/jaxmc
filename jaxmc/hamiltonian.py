from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp

from jaxmc.lattice import LatticeState

@jax.jit
def e_local_nn_1d(i:int, spins: jnp.array):
    L = jnp.size(spins)
    el = - 1.0 * spins[i] * (spins[(i+L-1)%L] + spins[(i+1)%L])
    return el

@jax.jit
def energy(spins: jnp.array, e_local: Callable) -> float:
    L = jnp.size(spins)

    e = 0.0

    for i in range(L):
        e = e + e_local(i, spins)
    e = 0.5 * e
    return e