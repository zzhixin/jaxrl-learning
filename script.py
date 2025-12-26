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

from jaxrl_learning.algos.ddpg import ActorNet, QNet, prepare, config, make_policy


env, test_env, buffer, train_state = prepare(random.key(0), config)
(
    env_params, env_states, test_env_params, buffer_state,
    actor_train_state, critic_train_state,
    eval_eps_ret_mean, best_eps_ret,
    key
) = train_state

policy = make_policy(env, env_params, actor_train_state, 
                     use_eps_greedy=False, eps_cur=0.1, std=0.1)