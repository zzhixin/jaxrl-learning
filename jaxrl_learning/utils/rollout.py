#%% Rollout function
from functools import partial
import jax
from jax import numpy as jnp


def rollout(key, env, env_state, env_params, policy, rollout_num_steps=100,
            collect_state=False, return_dict=True, policy_state=jnp.zeros(())):
    # checkify.check(isinstance(env, TerminationTruncationWrapper), "env should be TerminationTruncationWrapper")

    def policy_step(carry, _):
        state, key, policy_state = carry
        obs = env.get_obs(state)
        key, key_act, key_step = jax.random.split(key, 3)
        # action = env.action_space(env_params).sample(key_act_cur)
        if len(jnp.shape(policy_state)) > 0:
            action, next_policy_state = policy(key_act, obs, policy_state)
        else:
            action = policy(key_act, obs)
            next_policy_state = policy_state
        next_obs, next_state, reward, ter, tru, info = env.step(key_step, state, action, env_params)
        if collect_state:
            if return_dict:
                transition = {'state': state, 'obs': obs, 'action': action, 'rew': reward, 'next_obs': next_obs, 'ter': ter, 'tru': tru, 'info': info}
            else:
                transition = (state, obs, action, reward, next_obs, ter, tru, info)
        else:
            if return_dict:
                transition = {'obs': obs, 'action': action, 'rew': reward, 'next_obs': next_obs, 'ter': ter, 'tru': tru, 'info': info}
            else:
                transition = (obs, action, reward, next_obs, ter, tru, info)
        obs, state = next_obs, next_state
        carry = (state, key, next_policy_state)
        return carry, transition
    
    (env_state, key, _), exprs = \
        jax.lax.scan(policy_step, (env_state, key, policy_state), None, rollout_num_steps)
    
    return env_state, exprs


def batch_rollout(keys, env, env_states, env_params, policy, rollout_num_steps=100,
                  policy_state=jnp.zeros(())):
    if len(jnp.shape(policy_state)) == 0:
        b_rollout = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None, None, None, None))
        return b_rollout(keys, env, env_states, env_params, policy, rollout_num_steps,
                         False, True, policy_state)
    b_rollout = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None, None, None, 0))
    return b_rollout(keys, env, env_states, env_params, policy, rollout_num_steps,
                     False, True, policy_state)