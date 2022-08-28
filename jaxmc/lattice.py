from flax import struct

import jax
import jax.numpy as jnp

@struct.dataclass
class LatticeState():
    spins: jnp.ndarray
    neighbours: jnp.ndarray
    L: int