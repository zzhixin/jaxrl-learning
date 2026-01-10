import jax
from jax import numpy as jnp, random


# epsilon Schedule
def epsilon_schedule(global_steps, config):
    steps_fraction = global_steps / config["total_timesteps"]
    eps = jnp.interp(steps_fraction,
                    jnp.array([0., 1.]),
                    jnp.array([config["epsilon_start"], config["epsilon_end"]]))
    return eps

# noise Schedule (for normal_noise and ou_noise), decays over full training
def noise_std_schedule(global_steps, cfg):
    if not cfg["exploration_noise_decay"]:
        return cfg["exploration_noise"]
    steps_fraction = global_steps / cfg["total_timesteps"]
    std = jnp.interp(steps_fraction,
                    jnp.array([0., 1.]),
                    jnp.array([cfg["exploration_noise"], cfg["exploration_noise_end"]]))
    return std