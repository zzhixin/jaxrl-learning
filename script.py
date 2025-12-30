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
from jaxrl_learning.utils.rollout import batch_rollout
from jaxrl_learning.algos.ddpg import ActorNet, QNet, prepare, config, make_policy

NUM_ENVS = 16

env, test_env, buffer, train_state = prepare(random.key(0), config)
(
    env_params, env_states, test_env_params, buffer_state,
    actor_train_state, critic_train_state,
    eval_eps_ret_mean, best_eps_ret,
    key
) = train_state

print(f"action shape: {env.action_space(env_params).shape}")
policy = make_policy(env, env_params, actor_train_state.apply_fn, actor_train_state.params, 
                     use_eps_greedy=True, eps_cur=0.9, std=0.)

key = random.key(0)
rollout_keys = random.split(key, NUM_ENVS)

env_state, (obses, actions, rewards, next_obses, ters, trus, infos) = \
    batch_rollout(rollout_keys, env, env_states, env_params, policy, rollout_num_steps=100)

flat_actions = jnp.reshape(actions, (-1,) + env.action_space(env_params).shape)[...,0]

import matplotlib.pyplot as plt

plt.hist(flat_actions)
plt.show()


#%%
import jax 
from jax import numpy as jnp, random
from flax.core.frozen_dict import freeze

def foo(data):
    return data['x'] + data['y']

data = {
    'x': jnp.ones((10, 2)),
    'y': jnp.ones((2,)),
}
foo_vjit = jax.jit(jax.vmap(foo, in_axes=({'x': 0, 'y': None},)))
foo_vjit(data)


#%%
import flax.linen as nn
import jax.numpy as jnp
import jax.random as random
import jax
from jaxrl_learning.algos.ddpg import ActorNet, QNet, prepare, config, make_policy
jax.config.update("jax_platform_name", "cpu")
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
import pprint


def prepare_pure(key, config):
    env, test_env, buffer, train_state = prepare(key, config)
    return train_state

train_state = prepare_pure(random.key(0), config)
(
    env_params, env_states, test_env_params, buffer_state,
    actor_train_state, critic_train_state,
    metrics, key
) = train_state


config = freeze(config)
train_state_jit = jax.jit(prepare_pure, static_argnums=1)(random.key(0), config)
(
    env_params, env_states, test_env_params, buffer_state,
    actor_train_state, critic_train_state,
    metrics, key
) = train_state_jit
env_states = jax.block_until_ready(env_states)


def strip_apply_fn(ts):
    return ts.replace(apply_fn=None, tx=None)


def strip_state(state):
    (
        env_params, env_states, test_env_params, buffer_state,
        actor_ts, critic_ts,
        metrics, key
    ) = state
    return (
        env_params, env_states, test_env_params, buffer_state,
        strip_apply_fn(actor_ts), strip_apply_fn(critic_ts),
        metrics, key
    )


state_a = strip_state(train_state)
state_b = strip_state(train_state_jit)


def diff_leaf(x1, x2):
    x1 = jnp.asarray(x1)
    x2 = jnp.asarray(x2)
    if str(x1.dtype).startswith("key<"):
        x1 = jax.random.key_data(x1)
        x2 = jax.random.key_data(x2)
    if x1.dtype == jnp.bool_:
        return float(jnp.mean(x1 != x2))
    return float(jnp.mean(x1 - x2))


diff = jax.tree.map(diff_leaf, state_a, state_b)


def colorize(value):
    if value == 0.0:
        return f"\033[32m{value:.6g}\033[0m"
    return f"\033[31m{value:.6g}\033[0m"


leaves, _ = jax.tree_util.tree_flatten(diff)
print("diff leaves:")
for v in leaves:
    print(colorize(float(v)))
diff = None
pprint.pp(diff)
