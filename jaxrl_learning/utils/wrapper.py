#%%
import jax
from jax import numpy as jnp, random
from gymnax.wrappers import LogWrapper
from gymnax.wrappers.purerl import GymnaxWrapper, LogEnvState
from gymnax.environments import environment
from jax.experimental import checkify
from functools import partial


class TerminationTruncationWrapper(GymnaxWrapper):

    def __init__(self, env):
        # checkify.check(isinstance(env, LogWrapper), "Wrapped env should be LogWrapper")
        self._env = env

    def __getattr__(self, name):
        return self._env.__getattr__(name)

    @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> tuple[jax.Array, LogEnvState]:
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: LogEnvState,
        action: int | float,
        params: environment.EnvParams | None = None,
    ) -> tuple[jax.Array, LogEnvState, jax.Array, bool, bool, dict]:
        """Step the environment.


        Args:
          key: Pkey key.
          state: The current state of the environment.
          action: The action to take.
          params: The parameters of the environment.


        Returns:
          A tuple of (observation, state, reward, done, info).
        """
        episode_lengths = state.episode_lengths
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        truncation = episode_lengths == (params.max_steps_in_episode - 1)
        termination = done * jnp.logical_not(truncation) > 0
        return obs, state, reward, termination, truncation, info


# key = jax.random.key(0)
# key, reset_key, rollout_key = jax.random.split(key, 3)

# env, env_params = gymnax.make("CartPole-v1")
# env = TerminationTruncationWrapper(LogWrapper(env))
# print(env._env)

# env_params = env_params.replace(max_steps_in_episode=5)

# obs, state = env.reset(reset_key, env_params)
# # print(env.get_obs(state.env_state))
# pprint.pp(state)

# for t in range(40):
#     rollout_key, act_key, step_key = random.split(rollout_key, 3)
#     action = env.action_space(env_params).sample(act_key)
#     next_obs, next_state, reward, ter, tru, info = env.step(step_key, state, action, env_params)
#     # if done:
#         # print(f"t: {t}\ndone: {done}\ninfo: {pprint.pformat(info)}")
#     print(f"t: {t}\ntermination: {ter}\ntruncation: {tru}\ninfo: {pprint.pformat(info)}")
#     # pprint.pp(state)  

#     obs, state = next_obs, next_state

