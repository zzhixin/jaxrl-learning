#%%
import os
from pathlib import Path
import sys
ws_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ws_root))

import jax
jax.config.update("jax_platform_name", "cuda")
from jaxrl_learning.algos.ddpg import DDPGConfig, make_train
from jax import numpy as jnp, random
from dataclasses import replace


BASE_CONFIG = replace(
    DDPGConfig(),
    env_name="MountainCarContinuous-v0",
    total_timesteps=300_000,
    lr_critic=1e-3,
    lr_actor=1e-3,
    gamma=0.99,
    tau=0.001,
    target_update_interval=1,
    exploration_type="ou_noise",
    exploration_noise=0.5,
    exploration_noise_end=0.0,
    exploration_noise_decay=False,
    ou_theta=0.15,
    epsilon_start=1.0,
    epsilon_end=0.05,
    exploration_fraction=0.9,
    num_env=1,
    update_interval=4,
    train_batch_size=64,
    buffer_size=1e5,
    learning_start=1e4,
    eval_interval=8192*4,
    eval_num_steps=2000,
    eval_num_env=16,
    log_interval=8192*4,
    wandb=True,
    save_model=True,
    run_name=None,
    ckpt_path="/home/zhixin/jaxrl-learning/ckpts/",
    silent=False,
    vmap_run=True,
)

def make_tune(batch_config, base_config):
    base_config = replace(base_config, vmap_run=True)

    def _train(batch_config, key):
        config = replace(base_config, **batch_config)
        train = make_train()
        return train(config, key)
    
    v_train = jax.vmap(_train, in_axes=(None, 0))
    for param in reversed(list(batch_config.keys())):
        in_axes = jax.tree.map(lambda k: None, batch_config)
        in_axes.update({param: 0})
        v_train = jax.vmap(v_train, in_axes=(in_axes, None))
    return v_train

exploration_noise = jnp.linspace(0.0, 1.0, 11)
ou_theta = jnp.linspace(0.0, 0.5, 11)
batch_config = {"exploration_noise": exploration_noise,
                "ou_theta": ou_theta}
keys = random.split(random.key(0), 5)
tune = make_tune(batch_config, BASE_CONFIG)
metrics, *_ = tune(batch_config, keys)

#%%
episodic_return = metrics["eval/episodic_return"]
print(episodic_return.shape)
evals_per_logging = BASE_CONFIG.eval_interval // BASE_CONFIG.update_interval
# episodic_return = episodic_return[...,::evals_per_logging]

average_episodic_return = jnp.nanmean(episodic_return, axis=(-1, -2))
print(average_episodic_return)

#%%
import matplotlib.pyplot as plt
import numpy as np

assert len(average_episodic_return.shape) == 2
k1, k2 = list(batch_config.keys())[0], list(batch_config.keys())[1]
x_vals = np.array(batch_config[k1])
y_vals = np.array(batch_config[k2])
z = np.array(average_episodic_return)
expected = (len(x_vals), len(y_vals))
if z.shape != expected:
    raise ValueError(f"Unexpected z shape {z.shape}, expected {expected} for (x,y)")
z_plot = z.T  # vmap order is (x,y), but pcolormesh expects (y,x)
z_max = np.nanmax(np.abs(z_plot))
z_min = -z_max
X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")

fig, ax = plt.subplots()

c = ax.pcolormesh(X, Y, z_plot, shading="auto", cmap="RdBu", vmin=z_min, vmax=z_max)
ax.set_title('average episodic return')
ax.set_xlabel(f"{k1}")
ax.set_ylabel(f"{k2}")
ax.set_xticks(x_vals)
ax.set_yticks(y_vals)
ax.axis([x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()])
fig.colorbar(c, ax=ax)

plt.show()
