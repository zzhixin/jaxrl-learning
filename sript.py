import jax
import numpy as np
import jax.numpy as jnp
from jax import random, jit
import time

jax.config.update("jax_platform_name", "cpu")

small_arr_size = (10,)
large_arr_size = (1_000_000,)

N_ITERS = 100

def complex_update_small_arr(small, large):
    x, y = large['x'], large['y']
    sampled_x = x[0:np.prod(small_arr_size)].reshape(small_arr_size)
    sampled_y = y[0:np.prod(small_arr_size)].reshape(small_arr_size)
    def loss(small_arr, sampled_x, sampled_y):
        return (small_arr * sampled_x - sampled_y).sum()
    grads = jax.grad(loss)(small, sampled_x, sampled_y)
    small = jax.tree.map(lambda param, grad: param - grad, small, grads)
    return small

def step_donate_early(key, small, large):
    key_, key = random.split(key)
    large['x'] = large['x'].at[0].set(0.)
    large['y'] = large['y'].at[0].set(0.)
    small['data'] = complex_update_small_arr(small['data'], large)
    # small['data'] = small['data'] + large['data'][0:small_arr_size[0]]
    # small['data'] = small['data'] + random.uniform(key, small_arr_size)
    return large, key_, small  # large early → in-place reuse

def step_donate_later(key, small, large):
    key_, key = random.split(key)
    large['x'] = large['x'].at[0].set(0.)
    large['y'] = large['y'].at[0].set(0.)
    small['data'] = complex_update_small_arr(small['data'], large)
    # small['data'] = small['data'] + large['data'][0:small_arr_size[0]]
    # small['data'] = small['data'] + random.uniform(key, small_arr_size)
    return key_, small, large  # large late → copy + deletion

step_donate_early = jit(step_donate_early, donate_argnames=["large"])
step_donate_later = jit(step_donate_later, donate_argnames=["large"])

def benchmark_early(jitted_fn, name):
    key = random.key(0)
    small = {'data': jnp.zeros(small_arr_size, dtype=jnp.float32)}
    large = {'x': jnp.zeros(large_arr_size, dtype=jnp.float32),
                 'y': jnp.zeros(large_arr_size, dtype=jnp.float32)}  
    # Warmup
    for _ in range(5):
        large, key, small = jitted_fn(key, small, large)
        jax.block_until_ready((large, key, small))
    # Time
    print(f"=== {name} ===")
    start = time.time()
    for _ in range(N_ITERS):
        large, key, small = jitted_fn(key, small, large)
        jax.block_until_ready((large, key, small))
    print(f"Average time: {(time.time() - start)/N_ITERS * 1e6:.1f} µs\n")

def benchmark_later(jitted_fn, name):
    key = random.key(0)
    small = {'data': jnp.zeros(small_arr_size, dtype=jnp.float32)}
    large = {'x': jnp.zeros(large_arr_size, dtype=jnp.float32),
                 'y': jnp.zeros(large_arr_size, dtype=jnp.float32)}  
    # Warmup
    for _ in range(5):
        key, small, large = jitted_fn(key, small, large)
        jax.block_until_ready((key, large, small))
    # Time
    print(f"=== {name} ===")
    start = time.time()
    for _ in range(N_ITERS):
        key, small, large = jitted_fn(key, small, large)
        jax.block_until_ready((key, large, small))
    print(f"Average time: {(time.time() - start)/N_ITERS * 1e6:.1f} µs\n")

benchmark_early(step_donate_early, "large returned early")
benchmark_later(step_donate_later, "large returned later")