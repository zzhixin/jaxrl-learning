#%%
import gymnax
import jax
from jax import random, numpy as jnp
from flax import struct, linen as nn
import numpy as np
from typing import Sequence
from jaxrl_learning.utils.wrapper import LogWrapper, TerminationTruncationWrapper
from jaxrl_learning.utils.replay_buffer_old import ReplayBuffer, ReplayBufferState, Experience
from jaxrl_learning.utils.utils import eps_greedy_policy, eps_greedy_policy_continuous
from jaxrl_learning.utils.rollout import rollout, batch_rollout
from functools import partial
from jax.experimental import checkify
from jaxrl_learning.utils.utils import save_model


#%%
@partial(jax.jit, static_argnames=("num_env", "num_steps", "env", "qnet"))
def evaluate(key, 
        qnet: nn.Module, qnet_params,
        env, env_params: gymnax.EnvParams,
        num_env, num_steps,):
    # checkify.check(isinstance(env, TerminationTruncationWrapper), "env should be TerminationTruncationWrapper")

    key, reset_key, rollout_key = random.split(key, 3)
    reset_keys = random.split(reset_key, num_env)
    rollout_keys = random.split(rollout_key, num_env)
    obses, env_states = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
    policy = partial(eps_greedy_policy,
                        env=env, 
                        env_params=env_params, 
                        qnet=qnet, 
                        qnet_params=qnet_params,
                        eps=0.)
    
    env_states, sampled_experiences = batch_rollout(rollout_keys, env, env_states, env_params, policy, num_steps)
    *_, infos = sampled_experiences
    episode_return_mean = (infos["returned_episode_returns"] * infos["returned_episode"]).sum() \
        / infos["returned_episode"].sum()

    return episode_return_mean



#%%
# @partial(jax.jit, static_argnames=("num_env", "num_steps", "env", "actor"))
def evaluate_continuous_action(key, 
        policy,
        env, env_params: gymnax.EnvParams,
        num_env, num_steps, global_steps):
    # checkify.check(isinstance(env, TerminationTruncationWrapper), "env should be TerminationTruncationWrapper")
    # checkify.check(env_params.max_steps_in_episode > num_steps, "max_steps_in_episode should be larger than num_steps")

    key, reset_key, rollout_key = random.split(key, 3)
    reset_keys = random.split(reset_key, num_env)
    rollout_keys = random.split(rollout_key, num_env)
    obses, env_states = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
    
    env_states, sampled_experiences = batch_rollout(rollout_keys, env, env_states, env_params, policy, num_steps)
    infos = sampled_experiences['info']
    episode_return_mean = (infos["returned_episode_returns"] * infos["returned_episode"]).sum() \
        / infos["returned_episode"].sum()
    episode_length_mean = (infos["returned_episode_lengths"] * infos["returned_episode"]).sum() \
        / infos["returned_episode"].sum()
    
    # jax.debug.print("global_steps: {},  episode_return_mean: {:.2f}", global_steps, episode_return_mean)

    return episode_return_mean, episode_length_mean


def make_eval_continuous(
        metrics,
        policy, env, env_params, num_env, num_steps, 
        global_steps):
    def eval_(key):
        nonlocal metrics
        best_eps_ret = metrics["eval/best_episodic_return"]
        eps_ret_mean, eps_len_mean = evaluate_continuous_action(key, policy, env, env_params, num_env, num_steps, global_steps)

        is_best_model = eps_ret_mean > best_eps_ret
        best_eps_ret = jnp.maximum(best_eps_ret, eps_ret_mean)
        metrics = metrics.copy({
            "eval/episodic_return": eps_ret_mean,
            "eval/episodic_length": eps_len_mean,
            "eval/best_episodic_return": best_eps_ret})
        return is_best_model, metrics

    return eval_