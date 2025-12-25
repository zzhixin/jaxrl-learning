#%%
import jax
import numpy as np
import jax.numpy as jnp
from jax import random, jit
import time
jax.config.update("jax_platform_name", "cpu")
import gymnax
from dataclasses import dataclass
from flax import struct
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from typing import Callable
from functools import partial

d = FrozenDict({"a": 1, "b": 2})
d2 = d.copy({"a": -1})
print(d2)