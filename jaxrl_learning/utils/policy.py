from functools import partial
import jax
from jax import numpy as jnp, random


def eps_greedy_policy(key, obs, env, env_params, qnet, qnet_params, eps):
    key, key1, key2 = random.split(key, 3)
    cond = random.uniform(key1) < eps
    rand_action = env.action_space(env_params).sample(key2)
    q_action = qnet.apply(qnet_params, obs).argmax()
    return (rand_action * cond + q_action * (1-cond)).astype(jnp.int32)


def eps_greedy_policy_continuous(key, obs, env, env_params, actor, actor_params, eps):
    key, key1, key2 = random.split(key, 3)
    cond = random.uniform(key1) < eps
    rand_action = env.action_space(env_params).sample(key2)
    action = actor.apply(actor_params, obs)
    return (rand_action * cond + action * (1-cond))


class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma, theta, dt, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0

    def sample(self, key, x, shape):
        # Based on https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        # dXt = theta * (mu - Xt) * dt + sigma * sqrt(dt) * dWt
        # Approximate with Euler-Maruyama method.
        # Here, x is the previous state.
        noise = self.sigma * jnp.sqrt(self.dt) * random.normal(key, shape=shape)
        x_new = x + self.theta * (self.mu - x) * self.dt + noise
        return x_new, x_new # Return noise and new state


def make_policy(env, env_params, 
                actor_apply_fn: callable, actor_params,
                exploration_type: str,
                eps_cur=None,
                std=None,
                ou_theta=None,
                dt=1.0):
    # Base policy (no exploration)
    base_policy = lambda key, obs: actor_apply_fn(actor_params, obs)

    def normal_noisy_policy_fn(key, obs, std_dev, dt):
        mean = actor_apply_fn(actor_params, obs)
        lo = env.action_space(env_params).low
        hi = env.action_space(env_params).high
        action_scale = (hi - lo)/2.
        noise = random.normal(key, mean.shape) * std_dev * jnp.sqrt(dt) * action_scale
        return jnp.clip(mean + noise, lo, hi)

    def ou_noise_policy_fn(key, obs, ou_state, theta, sigma, dt):
        mean = actor_apply_fn(actor_params, obs)
        lo = env.action_space(env_params).low
        hi = env.action_space(env_params).high
        scale = (hi - lo)/2.
        # Discrete-time OU with dt=1 per env step (same convention as normal noise).
        noise = ou_state + theta * (0. - ou_state) * dt \
            + sigma * jnp.sqrt(dt) * random.normal(key, mean.shape) * scale
        return jnp.clip(mean + noise, lo, hi), noise

    def eps_greedy_policy_fn(key, obs, sub_policy_inner, epsilon):
        key, key1, key2, key3 = random.split(key, 4)
        action = sub_policy_inner(key3, obs)
        cond = random.uniform(key1) < epsilon
        rand_action = env.action_space(env_params).sample(key2)
        return (rand_action * cond + action * (1-cond))

    if exploration_type == "none":
        policy = base_policy
    elif exploration_type == "normal_noise":
        policy = partial(normal_noisy_policy_fn, std_dev=std, dt=dt)
    elif exploration_type == "epsilon_greedy":
        policy = partial(eps_greedy_policy_fn, sub_policy_inner=base_policy, epsilon=eps_cur)
    elif exploration_type == "ou_noise":
        policy = partial(ou_noise_policy_fn, theta=ou_theta, sigma=std, dt=dt)
    else:
        raise ValueError(f"Unknown exploration_type: {exploration_type}")

    return policy

