from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp

from jaxmc.lattice import LatticeState

@jax.jit
def e_local_nn_1d(i:int, spins: jnp.array, neighbours: jnp.array):
    L = jnp.size(spins)
    el = 0.0
    for j in neighbours[i]:
        el = el + spins[j]
    
    el = - el * spins[i]
    return el

@partial(jax.jit, static_argnums=(2,))
def energy(spins: jnp.array, neighbours: jnp.array, e_local: Callable) -> float:
    def body_fn(i:int, vals: tuple[jnp.ndarray, jnp.ndarray], e_local: Callable):
        e, spins = vals
        return [e + e_local(i, spins, neighbours), spins]

    L = jnp.size(spins)

    init_vals = [0.0, spins]
    vals = jax.lax.fori_loop(0, L, partial(body_fn, e_local=e_local), init_vals)
    return 0.5 * vals[0]