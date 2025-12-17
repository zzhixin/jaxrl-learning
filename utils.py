#%% Epsilon-greedy policy
from functools import partial
import jax
from jax import numpy as jnp, random
import gymnax


def eps_greedy_policy(key, obs, env, env_params, qnet, qnet_params, eps):
    key, key1, key2 = random.split(key, 3)
    cond = random.uniform(key1) < eps
    rand_action = env.action_space(env_params).sample(key2)
    # return rand_action
    q_action = qnet.apply(qnet_params, obs).argmax()
    return (rand_action * cond + q_action * (1-cond)).astype(jnp.int32)


def eps_greedy_policy_continuous(key, obs, env, env_params, actor, actor_params, eps):
    key, key1, key2 = random.split(key, 3)
    cond = random.uniform(key1) < eps
    rand_action = env.action_space(env_params).sample(key2)
    # return rand_action
    action = actor.apply(actor_params, obs)
    return (rand_action * cond + action * (1-cond))

