import time
import warnings
warnings.filterwarnings("ignore")
import jax
jax.config.update("jax_platform_name", "cpu")
from jax import random, numpy as jnp
from flax import linen as nn
import time
from replay_buffer import ReplayBuffer, ReplayBufferState
from colorama import Fore, Style, init
init(autoreset=True)
from flax import linen as nn
import time
from functools import partial


seed = 0
lr = 2.5e-4
gamma = 0.99

obs_shape = (3,)
action_shape = (2,)   
features = (128, 64)

buffer_size = int(1e6)
sample_batch_size = 256 
add_batch_size = 6 

N_ITERS = 100


class CriticNet(nn.Module):
    features: tuple

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def create_fake_transition(key):
    key0, key1, key2, key3, key4, key5 = random.split(key, 6)
    obs = random.normal(key0, obs_shape)
    action = random.normal(key1, action_shape)
    rew = random.normal(key2, ())
    next_obs = random.normal(key3, obs_shape)
    ter = random.normal(key4, ())
    tru = random.normal(key5, ())
    return obs, action, rew, next_obs, ter, tru


@partial(jax.jit, static_argnames=["buffer"], donate_argnames=["buffer_state"])
def collect(key, buffer: ReplayBuffer, buffer_state: ReplayBufferState):
    key, sample_key = random.split(key)
    rollout_keys = jax.random.split(key, add_batch_size)
    experiences = jax.vmap(create_fake_transition)(rollout_keys)
    buffer_state = buffer.push(buffer_state, experiences)
    sampled = buffer.sample(sample_key, buffer_state)
    return buffer_state, sampled


@partial(jax.jit, static_argnames=["model"])
def update_model(model: nn.Module, model_params, sampled):
    def batch_model_loss_fn(model_params, experiences):
        def model_loss_fn(experience):
            obs, action, reward, next_obs, termination, truncation = experience
            v_next = model.apply(model_params, next_obs)
            target = reward + gamma * v_next * (1 - termination)
            target = jax.lax.stop_gradient(target)
            v_pred = model.apply(model_params, obs)
            return (v_pred - target)**2
        return jax.vmap(model_loss_fn)(experiences).mean()

    grads = jax.grad(batch_model_loss_fn)(model_params, sampled)
    model_params = jax.tree.map(lambda param, grad: param - grad*lr, model_params, grads)
    return  model_params 


def train_one_step(key, model: nn.Module, model_params, 
                   buffer: ReplayBuffer, buffer_state: ReplayBufferState):
    key, rollout_key = jax.random.split(key)
    buffer_state, sampled = collect(rollout_key, buffer, buffer_state)
    model_params = update_model(model, model_params, sampled)
    return  key, model_params, buffer_state


def benchmark(train_fn):
    # prepare components
    buffer = ReplayBuffer.create(
        buffer_size=buffer_size, 
        rollout_batch=add_batch_size, 
        sample_batch=sample_batch_size,
        discrete_action=False
    )
    buffer_state = buffer.init(obs_shape, action_shape)
    key, model_key = random.split(random.key(seed))
    model = CriticNet(features=features)
    model_params = model.init(model_key, jnp.ones(obs_shape))
    
    # warm up
    key, model_params, buffer_state = jax.block_until_ready(
            train_fn(key, model, model_params, buffer, buffer_state))

    # training loop
    print("start timing...")
    start_time = time.time()
    for iter in range(N_ITERS):
        key, model_params, buffer_state = jax.block_until_ready(
            train_fn(key, model, model_params, buffer, buffer_state))
    ave_timedelta = (time.time() - start_time)/N_ITERS
    print(f"{Fore.BLUE}Average time {ave_timedelta*1e3:.2f} ms")


if __name__ == "__main__":
    # Only jit inner function
    print("--------- jit inner ----------")
    benchmark(train_one_step)
    print(f"{Fore.GREEN}rollout_and_push compile times: {collect._cache_size()}")
    print(f"{Fore.GREEN}update_model compile times: {update_model._cache_size()}")
        

    # Jit the outmost function train_one_step
    print("\n-------- jit outer ---------")
    train_one_step_jit = jax.jit(train_one_step,
                                 static_argnames=["model", "buffer"],
                                 donate_argnames=["buffer_state"]
                                 )
    benchmark(train_one_step_jit)
    print(f"{Fore.GREEN}train_one_step compile times: {train_one_step_jit._cache_size()}")