# Compared with dqn, ddpg needs much larger buffer size and batch size 
# because it tries to learn a more complex obs-action mapping. And the actor learning
# is more noisy.

# The performance tips: which function to be jitted should be decided carefully.
# tldr: don't jit the outmost `train_one_step` function, split it into 
# `rollout_and_push` and `update_model` and jit them respectively. Specifically, 
# donate the large `buffer_state` argument of `rollout_and_push` function, which 
# is the critic point to ensure underhood in-place update of large buffer.

# So why jitting together degrades (donating the large `buffer_state` argument
# doesn't make difference)? GPT says :
# From XLA’s perspective:
# > “This buffer participates in a complex graph; proving safe in-place reuse is hard.
# > I’ll materialize a new value to be safe.”


import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, 'vendor')
import gymnax
import numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_log_compiles", "1")
# jax.config.update("jax_debug_nans", True)
# jax.disable_jit(disable=True)
from jax import random, numpy as jnp
from flax import linen as nn
import optax
import time
import wandb
from functools import partial

from wrapper import LogWrapper, TerminationTruncationWrapper
from replay_buffer import ReplayBuffer, ReplayBufferState
from rollout import batch_rollout
from utils import eps_greedy_policy_continuous
from evals import evaluate_continuous_action
import pprint
from datetime import datetime
from colorama import Fore, Style, init
init(autoreset=True)
from monitor_recompile import monitor_recompiles


#  Config Dictionary
# 直接定义为 Python 字典
config = {
    "project_name": "jaxrl",
    # "env_name": "MountainCarContinuous-v0",
    "env_name": "Pendulum-v1",
    "total_timesteps": 100_000,
    "lr": 2.5e-4,
    "gamma": 0.99,
    "tau": 0.001,
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
    "learning_start": 1e4,
    "test_freq": 10_000,       
    "test_num_steps": 2000,
    "test_num_env": 16,
    "features": (128, 64),    # <--- 直接写成 Tuple，WandB 不会改动本地这个变量
    "seed": 0,
    "log_freq": 1000,          # <--- 控制 WandB 写入频率，避免拖慢速度
    "wandb": False,
    "ckpt_path": '/home/zhixin/jaxrl-learning/ckpts/'
}

#  Q net
class QNet(nn.Module):
    features: tuple

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
    
class ActorNet(nn.Module):
    features: tuple
    action_dim: int
    action_scale: float
    action_bias: float

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x) * self.action_scale + self.action_bias
        return x


@partial(
        jax.jit, 
        static_argnames=["actor", "env", "buffer", "num_steps"],
        donate_argnames=["buffer_state"]
)
def rollout_and_push(key, 
                     actor: nn.Module, actor_params,
                     env, env_params: gymnax.EnvParams, env_state: gymnax.EnvState, 
                     eps_cur: float,
                     buffer: ReplayBuffer, buffer_state: ReplayBufferState,
                     num_steps: int):
    rollout_keys = jax.random.split(key, env.num_env)

    policy = partial(eps_greedy_policy_continuous,
                    env=env, env_params=env_params, actor=actor, actor_params=actor_params, eps=eps_cur)

    env_state, experiences = batch_rollout(rollout_keys, env, env_state, env_params, policy, num_steps)
    experiences = experiences[:-1]
    flat_experience = jax.tree.map(lambda x: jnp.reshape(x, shape=(-1, *x.shape[2:])), experiences)
    buffer_state = buffer.push(buffer_state, flat_experience)

    return  env_state, buffer_state


@partial(
        jax.jit, 
        static_argnames=["qnet", "actor", "opt"],
)
def update_model(sampled_experiences,
                 qnet: nn.Module, actor: nn.Module, opt: optax.GradientTransformation, 
                 qnet_params, actor_params, target_qnet_params, target_actor_params,
                 qnet_opt_state: optax.OptState, actor_opt_state: optax.OptState, 
                 gamma: float
                 ):
    def batch_critic_loss_fn(qnet_params, target_actor_params, experiences):
        def critic_loss_fn(experience):
            obs, action, reward, next_obs, termination, truncation = experience
            next_max_action = actor.apply(target_actor_params, next_obs)
            q_next = qnet.apply(target_qnet_params, jnp.concat((next_obs, next_max_action), axis=-1))
            target = reward + gamma * q_next * (1 - termination)
            target = jax.lax.stop_gradient(target)
            q_pred = qnet.apply(qnet_params, jnp.concat((obs, action), axis=-1))
            return (q_pred - target)**2
        
        return jax.vmap(critic_loss_fn)(experiences).mean()
    
    def batch_actor_loss_fn(actor_params, qnet_params, experiences):
        def actor_loss_fn(experience):
            obs, *_ = experience
            action = actor.apply(actor_params, obs)
            q_pred = qnet.apply(qnet_params, jnp.concat((obs, action), axis=-1))
            return -q_pred
        
        return jax.vmap(actor_loss_fn)(experiences).mean()

    # update qnet
    loss, grads = jax.value_and_grad(batch_critic_loss_fn)(qnet_params, target_actor_params, sampled_experiences)
    updates, qnet_opt_state = opt.update(grads, qnet_opt_state)
    qnet_params = optax.apply_updates(qnet_params, updates)

    # update actor
    loss, grads = jax.value_and_grad(batch_actor_loss_fn)(actor_params, qnet_params, sampled_experiences)
    updates, actor_opt_state = opt.update(grads, actor_opt_state)
    actor_params = optax.apply_updates(actor_params, updates)

    return  loss, qnet_params, target_qnet_params, qnet_opt_state, actor_params, target_actor_params, actor_opt_state


# @partial(
#         jax.jit, 
#         static_argnames=["qnet", "actor", "opt", "env", 
#                          "buffer", "num_steps",
#                          "is_update_target_model", "is_update_model"],
#         donate_argnames=["buffer_state"]
# )
def train_one_step(
        key, 
        qnet: nn.Module, actor: nn.Module, opt: optax.GradientTransformation, 
        qnet_params, actor_params, target_qnet_params, target_actor_params,
        qnet_opt_state: optax.OptState, actor_opt_state: optax.OptState, 
        env, env_params: gymnax.EnvParams, env_state: gymnax.EnvState, 
        eps_cur: jax.Array,
        gamma: float, tau: float, 
        is_update_model: bool, is_update_target_model: bool,
        buffer: ReplayBuffer, buffer_state: ReplayBufferState,
        num_steps: int):
    
    key, rollout_key, sample_key = jax.random.split(key, 3)

    env_state, buffer_state = rollout_and_push(rollout_key, 
                     actor, actor_params,
                     env, env_params, env_state, 
                     eps_cur,
                     buffer, buffer_state,
                     num_steps)

    loss = jnp.array(0.0)
    if is_update_model:
        sampled_experiences = buffer.sample(sample_key, buffer_state)
        loss, qnet_params, target_qnet_params, qnet_opt_state, actor_params, target_actor_params, actor_opt_state = \
            update_model(sampled_experiences,
                        qnet, actor, opt, 
                        qnet_params, actor_params, target_qnet_params, target_actor_params,
                        qnet_opt_state, actor_opt_state, 
                        gamma
                        )
    if is_update_target_model:
        target_qnet_params = jax.tree.map(lambda q, t: q * tau + t * (1 - tau), qnet_params, target_qnet_params)
        target_actor_params = jax.tree.map(lambda q, t: q * tau + t * (1 - tau), actor_params, target_actor_params)
    
    return  key, loss, qnet_params, target_qnet_params, qnet_opt_state, actor_params, target_actor_params, actor_opt_state, env_state, buffer_state


def prepare(key, config):
    key = jax.random.key(config["seed"])
    key, init_reset_key, qnet_init_key, actor_init_key = jax.random.split(key, 4)
    
    env, env_params = gymnax.make(config["env_name"])
    env = TerminationTruncationWrapper(LogWrapper(env))

    env.num_env = config["num_env"]
    
    key_resets = random.split(init_reset_key, config["num_env"])
    obses, env_states = jax.vmap(env.reset, in_axes=(0, None))(key_resets, env_params)
    dummy_act_keys = random.split(random.key(0), config["num_env"])          
    dummy_actions = jax.vmap(env.action_space(env_params).sample)(dummy_act_keys)

    rollout_batch_size = config["num_env"] * config["train_freq"]
    buffer = ReplayBuffer.create(
        buffer_size=config["buffer_size"], 
        rollout_batch=rollout_batch_size, 
        sample_batch=config["train_batch_size"],
        discrete_action=False
    )
    buffer_state = buffer.init(obses.shape[1:], dummy_actions.shape[1:])

    qnet = QNet(features=config["features"])
    qnet_params = qnet.init(qnet_init_key, jnp.concat((obses, dummy_actions), axis=-1))
    target_qnet_params = qnet_params.copy()

    action_lo = env.action_space(env_params).low
    action_hi = env.action_space(env_params).high
    actor = ActorNet(features=config["features"], 
                    action_dim=np.prod(env.action_space(env_params).shape),
                    action_scale=(action_hi - action_lo)/2, 
                    action_bias=(action_lo + action_hi)/2)
    actor_params = actor.init(actor_init_key, obses)
    target_actor_params = actor_params.copy()

    opt = optax.adam(learning_rate=config["lr"])
    qnet_opt_state = opt.init(qnet_params)
    actor_opt_state = opt.init(actor_params)

    test_env, test_env_params = gymnax.make(config["env_name"])
    test_env = TerminationTruncationWrapper(LogWrapper(test_env))

    return (
        env, env_params, env_states, 
        test_env, test_env_params, 
        buffer, buffer_state, 
        qnet, actor, opt, 
        qnet_params, actor_params, target_qnet_params, target_actor_params,
        qnet_opt_state, actor_opt_state,  
    ) 

def run_training(config, warmup=None, silent=False):
    run_name = config["env_name"] + "__ddpg__" + datetime.now().strftime('%Y%m%d_%H%M%S')

    if config["wandb"]:
        wandb.init(
            project=config["project_name"],
            name=run_name,
            config=config, 
            monitor_gym=False
        )
    
    # extrac configurations
    rollout_batch_size = config["num_env"] * config["train_freq"]
    num_steps_per_rollout = config["train_freq"] // config["num_env"]
    num_train_steps = config["total_timesteps"] // rollout_batch_size
    learning_start = config["learning_start"]
    target_update_freq = config["target_update_freq"]
    gamma = config["gamma"]
    tau = config["tau"]
    log_freq = config["log_freq"] 
    if not silent:
        print(f"config:\n{pprint.pformat(config)}")

    key = random.key(config["seed"])
    key, init_key, train_key, test_key = random.split(key, 4)
    (
        env, env_params, env_states, 
        test_env, test_env_params, 
        buffer, buffer_state, 
        qnet, actor, opt, 
        qnet_params, actor_params, target_qnet_params, target_actor_params,
        qnet_opt_state, actor_opt_state,  
    )  =  prepare(init_key, config)

    # epsilon Schedule
    @jax.jit
    def get_epsilon(i_step):
        steps_fraction = i_step / (num_train_steps * config["exploration_fraction"])
        eps = jnp.interp(steps_fraction, jnp.array([0., 1.]), jnp.array([config["epsilon_start"], config["epsilon_end"]]))
        return eps
    
    # optional: warm up for benchmarking
    if warmup:
        (
            qnet_params, target_qnet_params, qnet_opt_state, 
            actor_params, target_actor_params, actor_opt_state, 
            env_states, buffer_state
        ) = warmup(train_one_step, rollout_and_push, update_model,
                   train_key, gamma, tau, 
                   qnet, actor, opt, 
                   qnet_params, actor_params, target_qnet_params, target_actor_params,
                   qnet_opt_state, actor_opt_state, 
                   env, env_params, env_states, 
                   buffer, buffer_state,
                   num_steps_per_rollout)
        
    # training loop
    print("start timing...")
    start_time = time.time()
    global_steps = 0

    for i_step in range(num_train_steps):
        eps_cur = get_epsilon(i_step)
        
        is_update_model = global_steps > learning_start
        is_update_target_model = is_update_model and (i_step % target_update_freq == 0)

        train_key, loss, *train_state = train_one_step(
            train_key, 
            qnet, actor, opt, 
            qnet_params, actor_params, target_qnet_params, target_actor_params,
            qnet_opt_state, actor_opt_state, 
            env, env_params, env_states, 
            eps_cur, 
            gamma, tau, 
            is_update_model, is_update_target_model,
            buffer, buffer_state,
            num_steps_per_rollout,
        )
        qnet_params, target_qnet_params, qnet_opt_state, actor_params, target_actor_params, actor_opt_state, env_states, buffer_state = train_state

        global_steps += rollout_batch_size

        # Logging
        assert config["test_num_steps"] > env_params.max_steps_in_episode
        if global_steps % config["test_freq"] == 0:
            test_key, test_key_cur = random.split(test_key)
            test_rew_mean = evaluate_continuous_action(
                test_key_cur, actor, actor_params, 
                test_env, test_env_params, 
                config["test_num_env"], config["test_num_steps"]
            )
            # print(f"evaluate_continuous_action compile times: {evaluate_continuous_action._cache_size()}")

            print(f"Step: {global_steps}, Eps: {eps_cur:.3f}, Loss: {loss:.4f}, Test Rew: {test_rew_mean:.2f}")
            
            if config["wandb"]:
                wandb.log({
                    "train/loss": loss,
                    "train/epsilon": eps_cur,
                    "test/reward_mean": test_rew_mean,
                    "global_steps": global_steps,
                    "time_elapsed": time.time() - start_time
                })
            
        elif is_update_model and i_step % log_freq == 0:
            if config["wandb"]:
                wandb.log({"train/loss": loss, "global_steps": global_steps})

    qnet_params = jax.block_until_ready(qnet_params)
    print(f"{Fore.BLUE}Training finished in {time.time() - start_time:.2f}s")

    return run_name, qnet_params, actor_params


if __name__ == "__main__":
    
    run_name, qnet_params, actor_params = run_training(config)

    # # Save latest model
    # import numpy as np
    # import orbax.checkpoint as ocp
    # import jax
    # from pathlib import Path

    # # path = ocp.test_utils.erase_and_create_empty(config["ckpt_path"])
    # path = Path(config["ckpt_path"])

    # model_params = {
    #     "qnet_params": qnet_params,
    #     "actor_params": actor_params
    # }

    # checkpointer = ocp.StandardCheckpointer()
    # # 'checkpoint_name' must not already exist.
    # checkpointer.save(path / run_name, model_params)


    # # Visualization
    # from gymnax.visualize import Visualizer
    # import numpy as np
    # import orbax.checkpoint as ocp
    # import jax
    # from pathlib import Path
    # import gymnasium as gym


    # key = jax.random.key(config["seed"])
    # key, init_reset_key, rollout_key = jax.random.split(key, 3)

    # env, env_params = gymnax.make(config["env_name"])
    # obs, state = env.reset(init_reset_key, env_params)
    # dummy_action = env.action_space(env_params).sample(random.key(0))


    # qnet = QNet(features=config["features"])
    # qnet_params = qnet.init(random.key(0), jnp.concat((obs, dummy_action), axis=-1))
    # action_lo = env.action_space(env_params).low
    # action_hi = env.action_space(env_params).high
    # actor = ActorNet(features=config["features"], 
    #                 action_dim=np.prod(env.action_space(env_params).shape),
    #                 action_scale=(action_hi - action_lo)/2,
    #                 action_bias=(action_lo + action_hi)/2)
    # actor_params = actor.init(random.key(0), obs)
    # model_params = {
    #     "qnet_params": qnet_params,
    #     "actor_params": actor_params
    # }

    # abstract_model_params = jax.tree_util.tree_map(
    #     ocp.utils.to_shape_dtype_struct, model_params)
    # checkpointer = ocp.StandardCheckpointer()
    # actor_params = checkpointer.restore(
    #     Path(config["ckpt_path"]) / run_name,
    #     abstract_model_params
    # )["actor_params"]

    # policy = partial(eps_greedy_policy_continuous, 
    #                  env=env, env_params=env_params, 
    #                  actor=actor, actor_params=actor_params, 
    #                  eps=0.)



    # gym_env = gym.make(config["env_name"], render_mode="human")
    # obs, _ = gym_env.reset(seed=0)

    # while True:
    #     key, key_act = jax.random.split(key)
    #     action = np.array(policy(key_act, obs))
    #     next_obs, reward, ter, tru, info = gym_env.step(action)

    #     done = ter or tru

    #     if done:
    #         time.sleep(1)
    #         gym_env.close()
    #         break
    #     else:
    #       obs = next_obs
    #       gym_env.render()
    #     #   time.sleep(0.05)


    # # # use gymnax's visualizer
    # # state_seq, reward_seq = [], []
    # # while True:
    # #     state_seq.append(state)
    # #     key, key_act, key_step = jax.random.split(key, 3)
    # #     action = policy(key_act, obs)
    # #     next_obs, next_state, reward, done, info = env.step(
    # #         key_step, state, action, env_params
    # #     )
    # #     reward_seq.append(reward)

    # #     if done:
    # #         break
    # #     else:
    # #       obs = next_obs
    # #       state = next_state

    # # cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    # # print(len(state_seq), len(cum_rewards))

    # # print("generating gif...")
    # # vis = Visualizer(env, env_params, state_seq, cum_rewards)
    # # vis.animate(f"gif/anim.gif")

