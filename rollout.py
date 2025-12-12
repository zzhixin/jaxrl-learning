#%% Rollout function
from functools import partial
import jax
from jax import numpy as jnp
import gymnax
from gymnax.wrappers import LogWrapper


def rollout(key, env, env_state, env_params, policy, trajectory_len=100, include_state_seq=False):
    # checkify.check(isinstance(env, TerminationTruncationWrapper), "env should be TerminationTruncationWrapper")


    def policy_step(env_state_and_key, _):
        state, key = env_state_and_key
        if 'env_state' in state.__dir__():
            obs = env.get_obs(state.env_state, env_params)
        else:
            obs = env.get_obs(state, env_params)

        key, key_act, key_step = jax.random.split(key, 3)
        
        # action = env.action_space(env_params).sample(key_act_cur)
        action = policy(key_act, obs)

        next_obs, next_state, reward, ter, tru, info = env.step(key_step, state, action, env_params)
        # jax.debug.print("{done},{reward}\n{info}", done=done, reward=reward, info=info)
        if include_state_seq:
            transition = (state, obs, action, reward, next_obs, ter, tru, info)
        else:
            transition = (obs, action, reward, next_obs, ter, tru, info)

        obs, state = next_obs, next_state

        env_state_and_key = (state, key)

        return env_state_and_key, transition
    
    step_count = jnp.arange(trajectory_len)

    if include_state_seq:
        (env_state, key), (states, obses, actions, rewards, next_obses, ters, trus, infos) = \
            jax.lax.scan(policy_step, (env_state, key), step_count)
        
        return env_state, (states, obses, actions, rewards, next_obses, ters, trus, infos)
    else:
        (env_state, key), (obses, actions, rewards, next_obses, ters, trus, infos) = \
            jax.lax.scan(policy_step, (env_state, key), step_count)
        
        return env_state, (obses, actions, rewards, next_obses, ters, trus, infos)

#%%

# rollout_jit = jax.jit(rollout, static_argnums=[1, 4])
# batch_rollout = jax.vmap(rollout, in_axes=(0, None, None, None, None))
# batch_rollout_jit = jax.jit(batch_rollout, static_argnums=[1, 4])

# @partial(jax.jit, static_argnames=["env", "trajectory_len", "policy"])
def batch_rollout(keys, env, env_states, env_params, policy, trajectory_len=100):
    b_rollout = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None))
    return b_rollout(keys, env, env_states, env_params, policy, trajectory_len)


