from typing import Any
from flax import struct

import jax
import jax.numpy as jnp

from jaxmc.lattice import LatticeState
from jaxmc.hamiltonian import e_local_nn_1d, energy

@struct.dataclass
class SingleSpinFlipState():
    acc_call: int
    tot_call: int

@struct.dataclass
class MonteCarloState():
    key: jnp.array

@jax.jit
def single_spin_flip_step(i: int, vals: tuple[jnp.array, MonteCarloState, SingleSpinFlipState, jnp.ndarray]):
    spins, mc, a, rands = vals
    L = jnp.size(spins)

    a = a.replace(tot_call=a.tot_call+1)

    index = (rands[i, 0] * L).astype(int)
    de = -2.0 * e_local_nn_1d(index, spins)
    rand = rands[i, 1]

    def accepted_func(vals):
        index, spins, a = vals
        spins = spins.at[index].set(-spins[index]) # FLIP
        a = a.replace(acc_call=a.acc_call+1)
        return spins, a

    def refused_func(vals):
        index, spins, a = vals
        return spins, a

    spins, a = jax.lax.cond(rand < jnp.exp(-0.5*de), accepted_func, refused_func, (index, spins, a))

    return [spins, mc, a, rands]

@jax.jit
def sweep(i: int, vals: tuple[jnp.array, MonteCarloState, SingleSpinFlipState, jnp.ndarray, dict]):
    spins, mc, a, rands, mean_values = vals

    L = jnp.size(spins)

    for k in range(L):
        spins, mc, a, rands = single_spin_flip_step(i*L+k, [spins, mc, a, rands])
    
    mean_values["e"] = mean_values["e"] + energy(spins, e_local_nn_1d)

    return [spins, mc, a, rands, mean_values]

def mc_sweeps(nsweeps: int, s: LatticeState, mc: MonteCarloState, a: SingleSpinFlipState):
    tot_steps = s.L*nsweeps

    # Generate all the random numbers
    new_key, subkey = jax.random.split(mc.key, num=2)
    rand = jax.random.uniform(subkey, (tot_steps, 2))
    mc = mc.replace(key=new_key)

    mean_values = {"e": 0.0}
    
    init_vals = [s.spins, mc, a, rand, mean_values]
    spins, mc, a, _, mean_values = jax.lax.fori_loop(0, nsweeps, sweep, init_vals)

    mean_values = jax.tree_util.tree_map(lambda x: x/nsweeps/s.L, mean_values)

    s = s.replace(spins=spins)

    return s, mc, a, mean_values

