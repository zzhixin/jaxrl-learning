import jax
jax.config.update("jax_platform_name", "cpu")
from jax import numpy as jnp, random
import time
from functools import partial
from replay_buffer import ReplayBuffer, ReplayBufferState


buffer_size = int(1e6)

@partial(jax.jit, static_argnames=["buffer",], donate_argnames=['buffer_state'])
# @partial(jax.jit, static_argnames=["buffer", ])
def train_one_step(buffer, buffer_state, model_params):
    obses = jnp.ones((10, 3))
    actions = jnp.ones((10, 1))
    rews = jnp.ones((10,))
    next_obses = jnp.ones((10, 3))
    ters = jnp.ones((10,))
    trus = jnp.ones((10,))
    transition = (obses, actions, rews, next_obses, ters, trus)
    buffer_state = buffer.push(buffer_state, transition)
    sampled = buffer.sample(random.key(0), buffer_state)

    def loss_fn(model_params, experience):
        obs, action, reward, next_obs, termination, truncation = experience
        return ((obs@model_params - reward)**2).mean()
    grads = jax.grad(loss_fn)(model_params, sampled)
    model_params = model_params + grads * 0.001
    return buffer_state, model_params

# buffer_state = jnp.ones((N, N))
buffer = ReplayBuffer.create(
        buffer_size=buffer_size, 
        rollout_batch=10, 
        sample_batch=256,
        discrete_action=False
    )
buffer_state = buffer.init((3,), (1,))

model_params = jnp.ones((3,))
buffer_state, model_params = train_one_step(buffer, buffer_state, model_params)  

start_time = time.time()
for i in range(100):
    sample = i%2
    buffer_state, model_params = train_one_step(buffer, buffer_state, model_params)  
print(f"Elapsed time: {time.time() - start_time:.5f}s")
