import jax; jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

import time

# again, this only works on startup!
from jax.config import config

from jaxmc.montecarlo import MonteCarloState
config.update("jax_enable_x64", True)

from jaxmc import LatticeState, energy, e_local_nn_1d, SingleSpinFlipState, single_spin_flip
from jaxmc import * 

seed = 0
L = 30

s = LatticeState(jnp.ones(L, dtype=jnp.float64), L)
a = SingleSpinFlipState(0, 0)
mc = MonteCarloState(jax.random.PRNGKey(seed))

#spins, mc, a = single_spin_flip_step(1, (lattice.spins, mc, a))

s, mc, a = mc_sweeps(1000, s, mc, a)

start = time.time()
s, mc, a = mc_sweeps(10**6, s, mc, a)
print(f"time = {time.time()-start}")

print("acceptance=", a.acc_call / a.tot_call)
#print(energy(lattice, e_local_nn_1d)[0])