"""
A circular buffer implementation.
"""
#%% Replay buffer
import jax
from jax import random
from jax import numpy as jnp
import chex
from collections import deque
from functools import partial
from collections import namedtuple
from flax import struct
from jax import Array
from dataclasses import astuple, dataclass
from jax.lax import dynamic_update_slice


@struct.dataclass
class Experience:
    obs: Array
    action: Array
    reward: Array
    next_obs: Array
    termination: Array
    truncation: Array


@struct.dataclass
class ReplayBufferState:
    experience: Experience
    current_index: Array
    is_full: Array


@struct.dataclass
class ReplayBuffer:
    buffer_size: int
    rollout_batch: int
    time_axis_limit: int
    sample_batch: int
    discrete_action: bool

    def init(self, obs_shape, action_shape, batch_rollout=True):
        if batch_rollout:
            action_dtype = jnp.int32 if self.discrete_action else jnp.float32
            experience = Experience(
                obs=jnp.zeros((self.time_axis_limit, self.rollout_batch, *obs_shape)),
                action=jnp.zeros((self.time_axis_limit, self.rollout_batch, *action_shape), 
                                 dtype=action_dtype),
                reward=jnp.zeros((self.time_axis_limit, self.rollout_batch)),
                next_obs=jnp.zeros((self.time_axis_limit, self.rollout_batch, *obs_shape)),
                termination=jnp.zeros((self.time_axis_limit, self.rollout_batch)),
                truncation=jnp.zeros((self.time_axis_limit, self.rollout_batch)),
            )
            state = ReplayBufferState(
                experience=experience,
                current_index=jnp.array(0).astype(jnp.int32),
                is_full=jnp.array(False).astype(jnp.bool),
            )
            return state
        else:
            raise NotImplementedError("single rollout is not implemented yet")
    
    # NOTE: donate buffer state to allow "in-place" operation underhood to speed up.
    # Otherwise big buffer push can be slow.
    # @partial(jax.jit, static_argnames=['self'], donate_argnames=['state'])  
    @partial(jax.jit, static_argnames=['self'])  
    def push(self, state: ReplayBufferState, transition: tuple):
        obs_shape = state.experience.obs.shape[2:]
        action_shape = state.experience.action.shape[2:]
        transition = jax.tree.map(lambda data: jnp.expand_dims(data, 0), transition, )
        obs, action, reward, next_obs, termination, truncation = transition
        termination, truncation = termination.astype(jnp.float32), truncation.astype(jnp.float32)
        experience = state.experience
        cur_index = state.current_index
        # new_experience = Experience(
        #     obs= experience.obs.at[cur_index].set(obs),
        #     action = experience.action.at[cur_index].set(action),
        #     reward = experience.reward.at[cur_index].set(reward),
        #     next_obs= experience.next_obs.at[cur_index].set(next_obs),
        #     termination = experience.termination.at[cur_index].set(termination),
        #     truncation = experience.truncation.at[cur_index].set(truncation),
        # )
        new_experience = Experience(
            obs = dynamic_update_slice(experience.obs, obs, (cur_index, 0) + (0,)*len(obs_shape)),
            action = dynamic_update_slice(experience.action, action, (cur_index, 0) + (0,)*len(action_shape)),
            reward = dynamic_update_slice(experience.reward, reward, (cur_index, 0)),
            next_obs = dynamic_update_slice(experience.next_obs, next_obs, (cur_index, 0) + (0,)*len(obs_shape)),
            termination = dynamic_update_slice(experience.termination, termination, (cur_index, 0)),
            truncation = dynamic_update_slice(experience.truncation, truncation, (cur_index, 0)),
        )
        new_cur_index = (cur_index + 1) % self.time_axis_limit
        new_state = ReplayBufferState(
            experience=new_experience,
            current_index=new_cur_index,
            is_full=state.is_full | new_cur_index==0
        )
        return new_state

    @partial(jax.jit, static_argnames=['self', 'as_tuple'])
    def sample(self, key: random.PRNGKey, state: ReplayBufferState, as_tuple=True):
        key1, key2 = random.split(key)
        useable_time_len = self.time_axis_limit * state.is_full + (~state.is_full) * state.current_index
        sample_time_axis_indice = random.randint(key1, (self.sample_batch,), 0, useable_time_len)
        sample_batch_axis_indice = random.randint(key2, (self.sample_batch,), 0, self.rollout_batch)
        sampled_experience: Experience = jax.tree.map(
            lambda data: data[sample_time_axis_indice, sample_batch_axis_indice],
            state.experience
        )
        if as_tuple:
            return astuple(sampled_experience)
        else:
            return sampled_experience
        

    @classmethod
    def create(
        cls,
        buffer_size = 64,
        rollout_batch = 4,
        sample_batch = 8,
        discrete_action = True,
    ):
        time_axis_limit = buffer_size // rollout_batch + 1 * (buffer_size % rollout_batch > 0)
        return cls(
            buffer_size = buffer_size,
            rollout_batch = rollout_batch,
            time_axis_limit = int(time_axis_limit),
            sample_batch = sample_batch,
            discrete_action = discrete_action
        )

