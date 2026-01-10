import jax
from jax import numpy as jnp, random
from jax.experimental import checkify
from jaxrl_learning.utils.running_mean import RunningMeanStd, RunningMeanStdState
from flax.core.frozen_dict import freeze
from gymnax.wrappers.purerl import GymnaxWrapper, LogEnvState
from gymnax.environments import environment
from gymnax.environments import spaces
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from brax.envs.base import PipelineEnv
from typing import ClassVar, Optional


class TerminationTruncationWrapper(GymnaxWrapper):

    # @partial(jax.jit, static_argnames=("self",))
    def reset(
        self, key: jax.Array, params: environment.EnvParams | None = None
    ) -> tuple[jax.Array, LogEnvState]:
        return self._env.reset(key, params)

    # @partial(jax.jit, static_argnames=("self",))
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


class NormalizeObservation(GymnaxWrapper):
    """Normalizes observations to be centered at the mean with unit variance.\n
    WARN: unlike gymnasium, each reset will reset the running mean std of the observation.\n
    This wrapper alters the env_params and env_state's structure, which will be 
    ```
        FrozenDict({"env_params": original_env_params, "obs_rms_state": obs_rms_state})
        FrozenDict({"env_state": original_env_state, "obs_rms_state": obs_rms_state})
    ```

    The `self.default_params` will contain a initial obs_rms_state. You should retain the env_params after `gymnax.make`.
    Or you can manually pass the env_params with your obs_rms_state to `self.reset`, then it will return a env_state with 
    the denoted obs_rms_state. After that, the obs_rms_state will be transformed along with env_state until next reset.
    The `self.step` will not respect the env_params.\n
    Changing the env_state's obs_rms_state is not recommanded just like don't change env_state except for `env.step`. You should only 
    change the obs_rms_state using reset. 

    ### training env proceduce
    ```
    env, _ = gymnax.make(env_name)
    env = NormalizeObservation(env)
    env_params = env.default_params
    obs, env_state = env.reset(key, env_params)
    ...
    ```

    ### evaluation env proceduce
    ```
    obs_rms_state = get_rms_state()
    env, env_params = gymnax.make(env_name)
    env = NormalizeObservation(env)
    env_params = freeze({"env_params": env_params, "obs_rms_state": obs_rms_state})
    obs, env_state = env.reset(key, env_params)
    ...
    ```
    """

    def __init__(self, env, epsilon: float = 1e-8, eval=False):
        """This wrapper will normalize observations such that each observation is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)

        self.obs_rms = RunningMeanStd(epsilon=epsilon)
        self.epsilon = epsilon
        self._eval = eval
        self._update_running_mean = not eval
        
    def observation_space(self, env_params):
        """Observation space of the environment."""
        env_params = env_params["env_params"]
        obs_shape = self._env.observation_space(env_params).shape
        return spaces.Box(-jnp.inf, jnp.inf, obs_shape, jnp.float32)   

    def action_space(self, env_params):
        """Observation space of the environment."""
        env_params = env_params["env_params"]
        return self._env.action_space(env_params)

    @property
    def default_params(self):
        obs_shape = self._env.observation_space(self._env.default_params).shape
        obs_rms_state = RunningMeanStdState(
            mean=jnp.zeros(obs_shape),
            var=jnp.ones(obs_shape),
            count=self.epsilon,
        )
        return freeze({"env_params": self._env.default_params,
                       "obs_rms_state": obs_rms_state})
    
    def get_obs(self, env_state, env_params=None, key=None):
        if env_params:
            env_params = env_params["env_params"]
        obs_rms_state = env_state["obs_rms_state"]
        raw_obs = self._env.get_obs(env_state["env_state"], env_params, key)
        obs = (raw_obs - obs_rms_state.mean) / jnp.sqrt(obs_rms_state.var + self.epsilon)
        return obs

    def _observation(self, raw_obs, obs_rms_state):
        """Normalises the observation using the running mean and variance of the observations."""
        # raw_obs = super().get_obs(state)
        if self._update_running_mean:
            obs_rms_state = self.obs_rms.update(obs_rms_state, jnp.array([raw_obs]))
        obs = (raw_obs - obs_rms_state.mean) / jnp.sqrt(obs_rms_state.var + self.epsilon)
        return obs, obs_rms_state 

    def reset(self, key, env_params):
        obs_rms_state = env_params["obs_rms_state"]
        env_params = env_params["env_params"]
        obs, state = self._env.reset(key, env_params)
        obs, obs_rms_state = self._observation(obs, obs_rms_state)
        state = freeze({"env_state": state, "obs_rms_state": obs_rms_state})
        return obs, state

    def step(self, key, state, action, env_params):
        env_params = env_params["env_params"]
        obs_rms_state = state["obs_rms_state"]
        obs, state, *others = self._env.step(key, state["env_state"], action, env_params)
        obs, obs_rms_state = self._observation(obs, obs_rms_state)
        state = freeze({"env_state": state, "obs_rms_state": obs_rms_state})
        return obs, state, *others


class Brax2GymWrapper(environment.Environment):
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
    def default_params(self) -> environment.EnvParams:
        """Default environment parameters for Pendulum-v0."""
        return environment.EnvParams(max_steps_in_episode=self._env.episode_length)

    def reset(self, key, params):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params):
        state = self._env.step(state, action)
        # info = {**state.metrics, **state.info}
        info = {}
        return state.obs, state, state.reward, state.done, info
    
    def get_obs(self, state, params=None, key=None):
        return state.obs
