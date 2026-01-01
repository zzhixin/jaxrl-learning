# #%%
# import jax
# import numpy as np
# import jax.numpy as jnp
# from jax import random, jit
# import time
# jax.config.update("jax_platform_name", "cpu")
# import gymnax
# from dataclasses import dataclass
# from flax import struct
# from flax.training.train_state import TrainState
# from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
# from typing import Callable
# from functools import partial
# from jaxrl_learning.utils.rollout import batch_rollout
# from jaxrl_learning.algos.ddpg import ActorNet, QNet, prepare, config, make_policy

# NUM_ENVS = 16

# env, test_env, buffer, train_state = prepare(random.key(0), config)
# (
#     env_params, env_states, test_env_params, buffer_state,
#     actor_train_state, critic_train_state,
#     eval_eps_ret_mean, best_eps_ret,
#     key
# ) = train_state

# print(f"action shape: {env.action_space(env_params).shape}")
# policy = make_policy(env, env_params, actor_train_state.apply_fn, actor_train_state.params, 
#                      use_eps_greedy=True, eps_cur=0.9, std=0.)

# key = random.key(0)
# rollout_keys = random.split(key, NUM_ENVS)

# env_state, (obses, actions, rewards, next_obses, ters, trus, infos) = \
#     batch_rollout(rollout_keys, env, env_states, env_params, policy, rollout_num_steps=100)

# flat_actions = jnp.reshape(actions, (-1,) + env.action_space(env_params).shape)[...,0]

# import matplotlib.pyplot as plt

# plt.hist(flat_actions)
# plt.show()


# #%%
# import jax 
# from jax import numpy as jnp, random
# from flax.core.frozen_dict import freeze

# def foo(data):
#     return data['x'] + data['y']

# data = {
#     'x': jnp.ones((10, 2)),
#     'y': jnp.ones((2,)),
# }
# foo_vjit = jax.jit(jax.vmap(foo, in_axes=({'x': 0, 'y': None},)))
# foo_vjit(data)
