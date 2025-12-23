from functools import partial
from flax import struct
import jax
from jax import Array, random, numpy as jnp
from typing import Callable, DefaultDict


@struct.dataclass
class ReplayBufferState:
    experience: DefaultDict
    current_index: Array
    is_full: Array
    

@struct.dataclass
class ReplayBuffer:
    init: Callable
    add: Callable
    sample: Callable
    

def rb_init(transition: DefaultDict, time_axis_limit):
    experience = jax.tree.map(
        lambda data: jnp.zeros((time_axis_limit,) + data.shape).astype(jnp.float32), 
        transition)
    state = ReplayBufferState(
        experience=experience,
        current_index=jnp.array(0).astype(jnp.int32),
        is_full=jnp.array(False).astype(jnp.bool),
    )
    return state
    

def rb_add(state: ReplayBufferState, transition: tuple, time_axis_limit):
    experience = state.experience
    cur_index = state.current_index
    new_experience = jax.tree.map(
        lambda buffer, new_data: buffer.at[cur_index].set(new_data),
        experience,
        transition
    )
    new_cur_index = (cur_index + 1) % time_axis_limit
    new_state = ReplayBufferState(
        experience=new_experience,
        current_index=new_cur_index,
        is_full=state.is_full | new_cur_index==0
    )
    return new_state
    

def rb_sample(key: random.PRNGKey, state: ReplayBufferState, time_axis_limit, sample_batch, rollout_batch):
    key1, key2 = random.split(key)
    useable_time_len = time_axis_limit * state.is_full + (~state.is_full) * state.current_index
    sample_time_axis_indice = random.randint(key1, (sample_batch,), 0, useable_time_len)
    sample_batch_axis_indice = random.randint(key2, (sample_batch,), 0, rollout_batch)
    sampled_experience = jax.tree.map(
        lambda data: data[sample_time_axis_indice, sample_batch_axis_indice],
        state.experience
    )
    return sampled_experience
    
        
def make_replay_buffer(buffer_size=64, rollout_batch=4, sample_batch=8):
    time_axis_limit = int(buffer_size // rollout_batch + 1 * (buffer_size % rollout_batch > 0))
    
    add = partial(rb_add, time_axis_limit=time_axis_limit)
    init = partial(rb_init, time_axis_limit=time_axis_limit)
    sample = partial(rb_sample, 
                     time_axis_limit=time_axis_limit, 
                     sample_batch=sample_batch, 
                     rollout_batch=rollout_batch)

    return ReplayBuffer(
        init = init,
        add = add,
        sample = sample
    )