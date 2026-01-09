from dataclasses import dataclass
import gymnax
import jax
from jax import numpy as jnp
from jaxrl_learning.utils.wrapper import (
Brax2GymWrapper, TerminationTruncationWrapper, LogWrapper, NormalizeObservation
)
from gymnax.wrappers.purerl import FlattenObservationWrapper
import brax


def make_env(env_name: str, norm_obs=False, eval=False, **env_kwargs):
    is_brax = "brax" in env_name
    if is_brax:
        env_name = env_name[:-len("brax")]
        env = Brax2GymWrapper(brax.envs.get_environment(env_name="ant", **env_kwargs))
    else:
        env, _ = gymnax.make(env_name, **env_kwargs)

    env = TerminationTruncationWrapper(LogWrapper(FlattenObservationWrapper(env)))
    if norm_obs:
        env = NormalizeObservation(env, eval=eval)

    env_params = env.default_params
    return env, env_params