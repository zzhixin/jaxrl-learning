#%%
import jax
from jax import numpy as jnp, random
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
import pprint
from typing import ClassVar, Optional
from brax.envs.base import PipelineEnv
from gymnax.environments import spaces
from gymnax.environments.environment import Environment as GymnaxEnv, EnvParams
import jax
import numpy as np


#%%
class Brax2GymWrapper(GymnaxEnv):
    """A wrapper that converts Brax Env to one that follows Gymnax API."""

    def __init__(
        self, env: PipelineEnv, seed: int = 0, backend: Optional[str] = None
    ):
        self._env = AutoResetWrapper(EpisodeWrapper(env, episode_length=1000, action_repeat=1))
        self.backend = backend

    def observation_space(self, params):
        obs = jnp.inf * jnp.ones(self._env.observation_size, dtype='float32')
        return spaces.Box(-obs, obs, (self._env.observation_size,), dtype='float32')
    
    def action_space(self, params):
        action = jax.tree.map(jnp.array, self._env.sys.actuator.ctrl_range)
        return spaces.Box(action[:, 0], action[:, 1], action.shape[:-1], dtype='float32')
    
    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Pendulum-v0."""
        return EnvParams(max_steps_in_episode=self._env.episode_length)

    def reset(self, key, params):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, state, action, params):
        state = self._env.step(state, action)
        # info = {**state.metrics, **state.info}
        info = {}
        return state.obs, state, state.reward, state.done, info


#%%
env_name = 'ant'  
backend = 'positional'  

env = envs.get_environment(env_name=env_name,
                           backend=backend)
env = AutoResetWrapper(EpisodeWrapper(env, episode_length=1500, action_repeat=1))
# env = EpisodeWrapper(env, episode_length=1500, action_repeat=1)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

state = jit_reset(rng=jax.random.PRNGKey(seed=0))

key = random.key(0)
for t in range(2000):
    key, act_key = random.split(key)
    ctrl_range = env.sys.actuator.ctrl_range  # shape (nu, 2)
    low, high = ctrl_range[:, 0], ctrl_range[:, 1]
    action = jax.random.uniform(act_key, (env.action_size,), minval=low, maxval=high)

    n_state = jit_step(state, action)
    if jnp.astype(state.done, jnp.float32) == 1.:
        print(t)
        pprint.pp(state.info)
        # pprint.pp(jax.tree.map(lambda x, y: jnp.allclose(x, y), state, n_state))
    
    state = n_state


#%%
env_name = 'ant'  
backend = 'positional'  

env = Brax2GymWrapper(envs.get_environment(env_name=env_name, backend=backend))

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

env_params = env.default_params
obs, state = jit_reset(random.key(0), None)

print(type(state))

#%%
key = random.key(0)
rollout = []
obses = []
for t in range(1000):
    rollout.append(state.pipeline_state)
    obses.append(obs)

    key, act_key = random.split(key)
    action = env.action_space(env_params).sample(act_key)
    n_obs, n_state, reward, done, _ = jit_step(state, action, env_params)
    state = n_state

obses = jnp.stack(obses)
print(jnp.min(obses, axis=0), jnp.max(obses, axis=0), jnp.mean(obses, axis=0), jnp.sqrt(jnp.var(obses, axis=0)), sep='\n')

from brax.io import html
from pathlib import Path

html_str = html.render(env._env.sys.tree_replace({'opt.timestep': env._env.dt}), rollout)
Path("html/brax_render.html").write_text(html_str, encoding="utf-8")

#%%
from dataclasses import fields, is_dataclass
from flax import struct

def extend_flax_struct(obj, new_field_name, new_field_type, new_value):
    if not is_dataclass(obj):
        raise TypeError("obj is not a dataclass instance")
    Base = obj.__class__

    namespace = {"__annotations__": {new_field_name: new_field_type}}
    NewCls = type(f"{Base.__name__}Ext", (Base,), namespace)
    NewCls = struct.dataclass(NewCls)  # this creates the frozen dataclass + pytree

    data = {f.name: getattr(obj, f.name) for f in fields(Base)}
    data[new_field_name] = new_value
    return NewCls(**data)

@struct.dataclass
class Old:
    a: float
    b: float

old = Old(a=1., b=2.)
new = extend_flax_struct(old, "c", float, 3.)
print(new)


#%%
import jax
from jax import numpy as jnp, random
import gymnax
import pprint
from jaxrl_learning.utils.env_factory import make_env
from jaxrl_learning.utils.rollout import rollout as rollout_fn, batch_rollout

key = random.key(0)
key, key_reset, key_act, key_step = jax.random.split(key, 4)

# Instantiate the environment & its settings.
env, env_params = make_env("Ant-brax", norm_obs=True)
# env, env_params = make_env("Pendulum-v1", norm_obs=False)

print(env.observation_space(env_params).shape)

# Reset the environment.
obs, env_state = env.reset(key_reset, env_params)
print(env_params)
print(env_state)

policy=lambda key, obs: env.action_space(env_params).sample(key) 
# env_state, exprs = rollout_fn(random.key(0), 
#                               env, env_state, env_params,
#                               policy=lambda key, obs: env.action_space(env_params).sample(key), 
#                               rollout_num_steps=50000)

# obses = exprs['obs']
# print(jnp.mean(obses, axis=0), jnp.var(obses, axis=0))

keys = random.split(random.key(0), 4)
obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(keys, env_params)
env_state, exprs = batch_rollout(keys, env, env_state, env_params, 
                                 policy, rollout_num_steps=10000)

obses = exprs['obs']
print(jnp.mean(obses, axis=1), jnp.var(obses, axis=1), sep="\n")
pprint.pp(env_state["obs_rms_state"])




#%%
import gymnasium as gym
from gymnasium.wrappers.stateful_observation import NormalizeObservation
from gymnasium.wrappers.common import Autoreset
import numpy as np

env = gym.make("Pendulum-v1")
env = NormalizeObservation(Autoreset(env))
obs, _ = env.reset(seed=17)
obses = []
for _ in range(50000):
    obses.append(obs)
    action = env.action_space.sample()
    n_obs, rew, ter, tru, _ = env.step(action)
    obs = n_obs

obses = np.stack(obses)
print(np.mean(obses, axis=0), np.var(obses, axis=0))

