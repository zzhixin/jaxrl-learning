#%%
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu' 

import gymnax
import numpy as np
import jax
from jax import random, numpy as jnp
from flax import linen as nn
import optax
import time
import wandb
from functools import partial

from wrapper import LogWrapper, TerminationTruncationWrapper
from replay_buffer import ReplayBuffer, ReplayBufferState
from rollout import batch_rollout
from utils import eps_greedy_policy
from evals import evaluate
import pprint
from datetime import datetime


# %% Config Dictionary
# 直接定义为 Python 字典
config = {
    "project_name": "jaxrl",
    # "env_name": "CartPole-v1",
    # "env_name": "Acrobot-v1",
    "env_name": "MountainCar-v0",
    # "env_name": "FourRooms-misc",
    "total_timesteps":  100_000,
    "lr": 2.5e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "target_update_freq": 1,
    # "tau": 1.,
    # "target_update_freq": 500,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "exploration_fraction": 0.5,
    "num_env": 1,
    "train_freq": 10,
    "train_batch_size": 128,
    "buffer_size": 1e6,
    "learning_start": 1e5,
    "test_freq": 10_000,       
    "test_num_steps": 500,
    "test_num_env": 16,
    "features": (128, 64),    # <--- 直接写成 Tuple，WandB 不会改动本地这个变量
    "seed": 0,
    "log_freq": 1000,          # <--- 控制 WandB 写入频率，避免拖慢速度
    "wandb": False,
    "ckpt_path": '/home/zhixin/jaxrl-learning/ckpts/'
}

# %% Q net
class QNet(nn.Module):
    features: tuple  # 明确标记为 tuple
    num_actions: int

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return x

# %% Training function (保持不变)
@partial(jax.jit, 
         static_argnames=["model", "opt", "env", "buffer", 
                          "num_steps", "gamma", "tau",
                          "update_target_net", "update_qnet"])
def train_one_step(
        key, 
        model: nn.Module, model_params, target_model_params,
        opt: optax.GradientTransformation, opt_state: optax.OptState,
        env, env_params: gymnax.EnvParams, env_state: gymnax.EnvState, 
        eps_cur: float,
        gamma: float, tau: float, 
        update_qnet: bool, update_target_net: bool,
        buffer: ReplayBuffer, buffer_state: ReplayBufferState,
        num_steps: int):
    
    policy = partial(eps_greedy_policy,
                    env=env, env_params=env_params, qnet=model, qnet_params=model_params, eps=eps_cur)
    
    key, rollout_key, sample_key = jax.random.split(key, 3)
    rollout_keys = jax.random.split(rollout_key, env.num_env)

    env_state, experiences = batch_rollout(rollout_keys, env, env_state, env_params, policy, num_steps)
    experiences = experiences[:-1]
    flat_experience = jax.tree.map(lambda x: jnp.reshape(x, shape=(-1, *x.shape[2:])), experiences)
    buffer_state = buffer.push(buffer_state, flat_experience)
    sampled_experiences = buffer.sample(sample_key, buffer_state)

    def batch_loss_fn(params, experiences):
        def loss_fn(experience):
            obs, action, reward, next_obs, termination, truncation = experience
            q_next = jnp.max(model.apply(target_model_params, next_obs))
            target = reward + gamma * q_next * (1 - termination)
            target = jax.lax.stop_gradient(target)
            q_pred = model.apply(params, obs)[action]
            return (q_pred - target)**2
        return jax.vmap(loss_fn)(experiences).mean()
    
    loss = jnp.array(0.0)
    if update_qnet:
        loss, grads = jax.value_and_grad(batch_loss_fn)(model_params, sampled_experiences)
        updates, opt_state = opt.update(grads, opt_state)
        model_params = optax.apply_updates(model_params, updates)

    if update_target_net:
        target_model_params = jax.tree.map(lambda q, t: q * tau + t * (1 - tau), model_params, target_model_params)

    return  key, loss, model_params, target_model_params, opt_state, env_state, buffer_state


# %% Main Loop
def run_training(config):
    run_name = config["env_name"] + "__dqn__" + datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. 初始化 WandB (仅用于记录，不依赖其 config 对象)
    if config["wandb"]:
        wandb.init(
            project=config["project_name"],
            name=run_name,
            config=config, # 这里只是把字典传上去做展示
            monitor_gym=False
        )
    
    # 2. 衍生参数
    rollout_batch_size = config["num_env"] * config["train_freq"]
    num_steps_per_rollout = config["train_freq"] // config["num_env"]
    num_train_steps = config["total_timesteps"] // rollout_batch_size
    
    print(f"config:\n{pprint.pformat(config)}")

    # 3. 环境与随机数
    key = jax.random.key(config["seed"])
    key, init_reset_key, train_key, test_key, init_act_key = jax.random.split(key, 5)
    
    env, env_params = gymnax.make(config["env_name"])
    env = TerminationTruncationWrapper(LogWrapper(env))

    env.num_env = config["num_env"]
    
    key_resets = random.split(init_reset_key, config["num_env"])
    obses, env_states = jax.vmap(env.reset, in_axes=(0, None))(key_resets, env_params)

    # 4. 初始化组件
    buffer = ReplayBuffer.create(
        buffer_size=config["buffer_size"], 
        rollout_batch=rollout_batch_size, 
        sample_batch=config["train_batch_size"]
    )
    buffer_state = buffer.init(obses.shape[1:], ())

    model = QNet(features=config["features"], num_actions=env.num_actions)
    model_params = model.init(init_act_key, obses)
    target_model_params = model_params.copy()

    opt = optax.adam(learning_rate=config["lr"])
    opt_state = opt.init(model_params)

    test_env, test_env_params = gymnax.make(config["env_name"])
    test_env = TerminationTruncationWrapper(LogWrapper(test_env))

    # 5. Epsilon Schedule
    @jax.jit
    def get_epsilon(i_step):
        steps_fraction = i_step / (num_train_steps * config["exploration_fraction"])
        eps = jnp.interp(steps_fraction, jnp.array([0., 1.]), jnp.array([config["epsilon_start"], config["epsilon_end"]]))
        return eps

    # 6. 训练循环
    start_time = time.time()
    global_steps = 0
    
    # 提取常用参数到局部变量 (Micro-optimization)
    learning_start = config["learning_start"]
    target_update_freq = config["target_update_freq"]
    gamma = config["gamma"]
    tau = config["tau"]
    log_freq = config["log_freq"] # 使用 config 里定义的频率

    for i_step in range(num_train_steps):
        eps_cur = get_epsilon(i_step)
        
        update_qnet = global_steps > learning_start
        update_target_net = update_qnet and (i_step % target_update_freq == 0)

        train_key, loss, *train_state = train_one_step(
            train_key, 
            model, model_params, target_model_params,
            opt, opt_state, 
            env, env_params, env_states, 
            eps_cur, 
            gamma, tau,  # 直接传值
            update_qnet, update_target_net,
            buffer, buffer_state,
            num_steps_per_rollout,
        )
        model_params, target_model_params, opt_state, env_states, buffer_state = train_state

        global_steps += rollout_batch_size

        # Logging
        if global_steps % config["test_freq"] == 0:
            test_key, test_key_cur = random.split(test_key)
            test_rew_mean = evaluate(
                test_key_cur, model, model_params, 
                test_env, test_env_params, 
                config["test_num_env"], config["test_num_steps"]
            )

            print(f"Step: {global_steps}, Eps: {eps_cur:.3f}, Loss: {loss:.4f}, Test Rew: {test_rew_mean:.2f}")
            
            if config["wandb"]:
                wandb.log({
                    "train/loss": loss,
                    "train/epsilon": eps_cur,
                    "test/reward_mean": test_rew_mean,
                    "global_steps": global_steps,
                    "time_elapsed": time.time() - start_time
                })
            
        elif update_qnet and i_step % log_freq == 0:
            if config["wandb"]:
                wandb.log({"train/loss": loss, "global_steps": global_steps})

    # wandb.finish()
    print(f"Training finished in {time.time() - start_time:.2f}s")

    return run_name, jax.block_until_ready(model_params)


#%%
run_name, model_params = run_training(config)


#%%
import numpy as np
import orbax.checkpoint as ocp
import jax
from pathlib import Path

# path = ocp.test_utils.erase_and_create_empty(config["ckpt_path"])
path = Path(config["ckpt_path"])

checkpointer = ocp.StandardCheckpointer()
# 'checkpoint_name' must not already exist.
checkpointer.save(path / run_name, model_params)


#%%
from gymnax.visualize import Visualizer
import numpy as np
import orbax.checkpoint as ocp
import jax
from pathlib import Path
import gymnasium as gym


key = jax.random.key(config["seed"])
key, init_reset_key, rollout_key = jax.random.split(key, 3)

env, env_params = gymnax.make(config["env_name"])
obs, state = env.reset(init_reset_key, env_params)

model = QNet(features=config["features"], num_actions=env.num_actions)
model_params = model.init(random.key(0), obs)
abstract_model_params = jax.tree_util.tree_map(
    ocp.utils.to_shape_dtype_struct, model_params)
checkpointer = ocp.StandardCheckpointer()
model_params = checkpointer.restore(
    Path(config["ckpt_path"]) / run_name,
    abstract_model_params
)

policy = partial(eps_greedy_policy, 
                 env=env, env_params=env_params, 
                 qnet=model, qnet_params=model_params, 
                 eps=0.)



gym_env = gym.make(config["env_name"], render_mode="human")
obs, _ = gym_env.reset(seed=0)

while True:
    key, key_act = jax.random.split(key)
    action = np.array(policy(key_act, obs))
    next_obs, reward, ter, tru, info = gym_env.step(action)

    done = ter or tru

    if done:
        time.sleep(1)
        gym_env.close()
        break
    else:
      obs = next_obs
      gym_env.render()
    #   time.sleep(0.05)


# # use gymnax's visualizer
# state_seq, reward_seq = [], []
# while True:
#     state_seq.append(state)
#     key, key_act, key_step = jax.random.split(key, 3)
#     action = policy(key_act, obs)
#     next_obs, next_state, reward, done, info = env.step(
#         key_step, state, action, env_params
#     )
#     reward_seq.append(reward)

#     if done:
#         break
#     else:
#       obs = next_obs
#       state = next_state

# cum_rewards = jnp.cumsum(jnp.array(reward_seq))
# print(len(state_seq), len(cum_rewards))

# print("generating gif...")
# vis = Visualizer(env, env_params, state_seq, cum_rewards)
# vis.animate(f"gif/anim.gif")



# %%
