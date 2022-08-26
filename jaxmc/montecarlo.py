from typing import Any
from flax import struct

import jax
import jax.numpy as jnp

from jaxmc.lattice import LatticeState
from jaxmc.hamiltonian import e_local_nn_1d

@struct.dataclass
class ActionState():
    acc_call: int
    tot_call: int

@struct.dataclass
class SingleSpinFlipState(ActionState):
    pass

@struct.dataclass
class MonteCarloState():
    key: jnp.array

@jax.jit
def single_spin_flip(i: int, spins: jnp.ndarray):
    return spins.at[i].set(-spins[i])

@jax.jit
def single_spin_flip_step(i: int, vals: tuple[jnp.array, MonteCarloState, ActionState]):
    spins, mc, a = vals
    L = jnp.size(spins)

    a = a.replace(tot_call=a.tot_call+1)

    new_key, subkey = jax.random.split(mc.key)
    index = jax.random.randint(subkey, (1,), 0, L)[0]
    mc = mc.replace(key = new_key)

    de = -2.0 * e_local_nn_1d(index, spins)

    new_key, subkey = jax.random.split(mc.key)
    rand = jax.random.uniform(subkey, (1,))[0]
    mc = mc.replace(key = new_key)

    def accepted_func(vals):
        index, spins, a = vals
        spins = single_spin_flip(index, spins)
        a = a.replace(acc_call=a.acc_call+1)
        return spins, a

    def refused_func(vals):
        index, spins, a = vals
        return spins, a

    spins, a = jax.lax.cond(rand < jnp.exp(-de), accepted_func, refused_func, (index, spins, a))

    return [spins, mc, a]

@jax.jit
def mc_sweeps(nsweeps: int, s: LatticeState, mc: MonteCarloState, a: ActionState):
    init_vals = [s.spins, mc, a]

    spins, mc, a = jax.lax.fori_loop(0, s.L*nsweeps, single_spin_flip_step, init_vals)
    
    s = s.replace(spins=spins)

    return s, mc, a

