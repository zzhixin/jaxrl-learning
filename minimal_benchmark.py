#%%
import jax
jax.config.update("jax_platform_name", "cpu")
from jax import numpy as jnp, random
import time
from functools import partial
import humanize
from flax import struct
from jax import Array
from dataclasses import astuple, dataclass
from typing import Callable, DefaultDict


seed = 0
lr = 2.5e-4
gamma = 0.99

obs_shape = (3,)
features = (128, 64)

buffer_size = int(1e6)
sample_batch_size = 256 
add_batch_size = 6

N_ITERS = 100


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

def rb_init(timestep: DefaultDict, time_axis_limit):
    experience = jax.tree.map(
        lambda data: jnp.zeros((time_axis_limit,) + data.shape).astype(jnp.float32), 
        timestep)
    state = ReplayBufferState(
        experience=experience,
        current_index=jnp.array(0).astype(jnp.int32),
        is_full=jnp.array(False).astype(jnp.bool),
    )
    return state

def rb_add(state: ReplayBufferState, timestep: tuple, time_axis_limit):
    experience = state.experience
    cur_index = state.current_index
    new_experience = jax.tree.map(
        lambda buffer, new_data: buffer.at[cur_index].set(new_data),
        experience,
        timestep
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
        
def make_replay_buffer( buffer_size = 64, rollout_batch = 4, sample_batch = 8):
    time_axis_limit = buffer_size // rollout_batch + 1 * (buffer_size % rollout_batch > 0)
    
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

def create_fake_transition(key):
    key0, key1 = random.split(key)
    obs = random.normal(key0, obs_shape)
    rew = random.normal(key1, ())
    return {
        "obs": obs, 
        "rew": rew,
        }

def train_one_step_return_later(key, model_params, buffer, buffer_state):
    # collect data
    key, key_gen_data, key_sample = random.split(key, 3)
    keys = random.split(key_gen_data, add_batch_size)
    transition = jax.vmap(create_fake_transition)(keys)
    buffer_state = buffer.add(buffer_state, transition)

    # update model
    sampled = buffer.sample(key_sample, buffer_state)
    def loss_fn(model_params, experiences):
        obs, reward = experiences['obs'], experiences['rew']
        return ((obs@model_params - reward)**2).sum()
    
    grads = jax.grad(loss_fn)(model_params, sampled)
    # grads = jnp.ones_like(model_params)
    model_params = jax.tree.map(lambda param, grad: param - grad*lr, model_params, grads)
    
    return model_params, buffer_state, key

def train_one_step_return_early(key, model_params, buffer, buffer_state):
    model_params, buffer_state, key = train_one_step_return_later(
        key, model_params, buffer, buffer_state)
    return buffer_state, model_params, key

def benchmark_return_later():
    # prepare components
    buffer = make_replay_buffer(
            buffer_size=buffer_size, 
            rollout_batch=add_batch_size, 
            sample_batch=sample_batch_size,
        )
    keys = random.split(random.key(0), add_batch_size)
    timestep = jax.vmap(create_fake_transition)(keys)
    buffer_state = buffer.init(timestep)

    key = random.key(seed)
    model_params = jnp.ones(obs_shape)
    
    train_one_step_jit = jax.jit(train_one_step_return_later, 
                                static_argnames=["buffer"], 
                                donate_argnames=['buffer_state'])
    # warm up
    model_params, buffer_state, key = jax.block_until_ready(
        train_one_step_jit(key, model_params, buffer, buffer_state))  

    # training loop
    start_time = time.time()
    for iter in range(N_ITERS):
        model_params, buffer_state, key = jax.block_until_ready(
            train_one_step_jit(key, model_params, buffer, buffer_state))  
    ave_timedelta = (time.time() - start_time)/N_ITERS
    print(f"Average time: {humanize.precisedelta(ave_timedelta, minimum_unit='microseconds', format='%0.3f')}")

def benchmark_return_early():
    # prepare components
    buffer = make_replay_buffer(
            buffer_size=buffer_size, 
            rollout_batch=add_batch_size, 
            sample_batch=sample_batch_size,
        )
    keys = random.split(random.key(0), add_batch_size)
    timestep = jax.vmap(create_fake_transition)(keys)
    buffer_state = buffer.init(timestep)

    key = random.key(seed)
    model_params = jnp.ones(obs_shape)

    train_one_step_jit = jax.jit(train_one_step_return_early, 
                                static_argnames=["buffer"], 
                                donate_argnames=['buffer_state'])

    # warm up
    buffer_state, model_params, key = jax.block_until_ready(
        train_one_step_jit(key, model_params, buffer, buffer_state))  

    # training loop
    start_time = time.time()
    for iter in range(N_ITERS):
        buffer_state, model_params, key = jax.block_until_ready(
            train_one_step_jit(key, model_params, buffer, buffer_state))  
    ave_timedelta = (time.time() - start_time)/N_ITERS
    print(f"Average time: {humanize.precisedelta(ave_timedelta, minimum_unit='microseconds', format='%0.3f')}")


if __name__ == "__main__":
    print("-------- return later ---------")
    benchmark_return_later()
    print("\n-------- return early ---------")
    benchmark_return_early()