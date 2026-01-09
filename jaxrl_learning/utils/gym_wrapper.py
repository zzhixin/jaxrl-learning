
# too much effort! not worth it.


from gymnax.wrappers import LogWrapper
from gymnax.wrappers.purerl import GymnaxWrapper, LogEnvState
import gymnasium as gym
from gymnax.environments import spaces as gymnax_spaces
from gymnax.environments.environment import EnvParams
import numpy as np


class GymToGymnaxWrapper(GymnaxWrapper):
    def reset(self, key, env_params):
        seed = self.jaxKeyToNumpySeed(key)
        obs, info = self._env.reset(seed=seed)
        return jnp.asarray(obs), jnp.asarray(obs)

    def step(self, key, state, action, env_params):
        action = np.asarray(action)
        obs, reward, ter, tru, info = self._env.step(action)
        done = jnp.logical_or(ter, tru)
        return jnp.asarray(obs), jnp.asarray(obs), jnp.asarray(reward), jnp.asarray(done), {}

    @staticmethod
    def jaxKeyToNumpySeed(jax_key):
        k = np.asarray(jax.random.key_data(jax_key))  # uint32[2]
        seed = (int(k[0]) << 32) | int(k[1])  # 64-bit seed
        return seed


class BipedalWalkerGymnaxWrapper(GymToGymnaxWrapper):
    def observation_space(self, env_params):
        low = jnp.asarray(self._env.observation_space.low)
        high = jnp.asarray(self._env.observation_space.high)
        shape = jnp.asarray(self._env.observation_space.shape)
        return gymnax_spaces.Box(low=low, high=high, shape=shape)

    def action_space(self, env_params):
        low = jnp.asarray(self._env.action_space.low)
        high = jnp.asarray(self._env.action_space.high)
        shape = jnp.asarray(self._env.action_space.shape)
        return gymnax_spaces.Box(low=low, high=high, shape=shape)

    def get_params(self):
        return EnvParams(max_steps_in_episode=self.spec.max_episode_steps)


def make_gymnax_env_from_gym(env_name):
    registered_wrapper = {
        "BipedalWalker-v3": BipedalWalkerGymnaxWrapper
    }
    if env_name not in registered_wrapper:
        raise Exception(f"Env {env_name} not implemented")
    gymnax_wrapper = registered_wrapper[env_name]
    from gymnasium.wrappers import Autoreset
    env = gymnax_wrapper(Autoreset(gym.make(env_name)))
    param = env.get_params() #TODO
    return env, param

    
#%%
import jax
from wrapper import TerminationTruncationWrapper
import pprint

key = jax.random.key(0)
key, key_reset, key_act, key_step = jax.random.split(key, 4)

# Instantiate the environment & its settings.
# env, env_params = gymnax.make("Pendulum-v1")
# pprint.pp(env_params)

env, env_params = make_gymnax_env_from_gym("BipedalWalker-v3")
env = TerminationTruncationWrapper(LogWrapper(env))


# Reset the environment.
obs, state = env.reset(key_reset, env_params)

for i in range(env_params.max_steps_in_episode - 2):
    action = env.action_space(env_params).sample(key_act)
    n_obs, n_state, reward, ter, tru, info = env.step(key_step, state, action, env_params)
    obs, state = n_obs, n_state
    if ter or tru:
        break
print(ter, tru, info)

for i in range(3):
    action = env.action_space(env_params).sample(key_act)
    n_obs, n_state, reward, ter, tru, info = env.step(key_step, state, action, env_params)
    obs, state = n_obs, n_state
    print(ter, tru, info)

#%%
import gymnasium as gym

# Initialise the environment
env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
pprint.pp(env.__dict__)

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(10):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()