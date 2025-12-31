import jax
from jax import numpy as jnp, random


# epsilon Schedule
def epsilon_schedule(i_step, config):
    num_updates = config["total_timesteps"] // config["train_interval"]
    steps_fraction = i_step / (num_updates * config["exploration_fraction"])
    eps = jnp.interp(steps_fraction,
                    jnp.array([0., 1.]),
                    jnp.array([config["epsilon_start"], config["epsilon_end"]]))
    return eps

# noise Schedule (for normal_noise and ou_noise), decays over full training
def noise_std_schedule(i_step, config):
    num_updates = config["total_timesteps"] // config["train_interval"]
    if not config["exploration_noise_decay"]:
        return config["exploration_noise"]
    steps_fraction = i_step / num_updates
    std = jnp.interp(steps_fraction,
                    jnp.array([0., 1.]),
                    jnp.array([config["exploration_noise"], config["exploration_noise_end"]]))
    return std