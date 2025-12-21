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
from replay_buffer import ReplayBuffer, ReplayBufferState, make_replay_buffer
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
    "env_name": "MountainCarContinuous-v0",
    # "env_name": "Pendulum-v1",
    "total_timesteps": 1_000_000,
    "lr": 2.5e-4,   
    "gamma": 0.99,
    "tau": 0.001,
    "target_update_interval": 1,
    # "tau": 1.,
    # "target_update_interval": 500,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "exploration_fraction": 0.5,
    "num_env": 1,
    "train_interval": 32,
    "train_batch_size": 128,
    "buffer_size": 1e6,
    "learning_start": 2e5,
    "eval_interval": 8192,       
    "eval_num_steps": 2000,
    "eval_num_env": 16,
    "features": (128, 64),
    "seed": 0,
    "log_freq": 1000,
    "wandb": False,
    "ckpt_path": '/home/zhixin/jaxrl-learning/ckpts/'
}


def check_config(config):
    assert config["train_interval"] >= config["num_env"]
    assert config["train_interval"] % config["num_env"] == 0
    assert config["total_timesteps"] > config["learning_start"]
    assert config["eval_interval"] > config["train_interval"]
    assert config["eval_interval"] % config["train_interval"] == 0
    _, env_params = gymnax.make(config["env_name"])
    assert config["eval_num_steps"] > env_params.max_steps_in_episode


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


def rollout_and_push(key, 
                     actor: nn.Module, actor_params,
                     env, env_params: gymnax.EnvParams, env_state: gymnax.EnvState, 
                     eps_cur: float,
                     buffer: ReplayBuffer, buffer_state: ReplayBufferState,
                     num_steps: int):
    rollout_keys = jax.random.split(key, env.num_env)

    policy = partial(eps_greedy_policy_continuous,
                    env=env, env_params=env_params, actor=actor, actor_params=actor_params, eps=eps_cur)

    env_state, exprs = batch_rollout(rollout_keys, env, env_state, env_params, policy, num_steps)
    exprs = exprs[:-1]   # exclude the info
    exprs_dict = {'obs': exprs[0], 'action': exprs[1], 'rew': exprs[2], 'next_obs': exprs[3], 'ter': exprs[4], 'tru': exprs[5]}
    flat_exprs_dict = jax.tree.map(lambda x: jnp.reshape(x, shape=(-1, *x.shape[2:])), exprs_dict)
    buffer_state = buffer.add(buffer_state, flat_exprs_dict)

    return  env_state, buffer_state


def update_model(gamma: float, buffer: ReplayBuffer,
                 qnet: nn.Module, actor: nn.Module, opt: optax.GradientTransformation, 
                 update_state, buffer_state, target_qnet_params, target_actor_params, 
                 ):
    (
        qnet_params, actor_params, 
        qnet_opt_state, actor_opt_state, key
    ) = update_state

    def batch_critic_loss_fn(qnet_params, target_actor_params, experiences):
        def critic_loss_fn(expr):
            obs, action, reward, next_obs, termination = \
                expr['obs'], expr['action'], expr['rew'], expr['next_obs'], expr['ter']
            next_max_action = actor.apply(target_actor_params, next_obs)
            q_next = qnet.apply(target_qnet_params, jnp.concat((next_obs, next_max_action), axis=-1))
            target = reward + gamma * q_next * (1 - termination)
            target = jax.lax.stop_gradient(target)
            q_pred = qnet.apply(qnet_params, jnp.concat((obs, action), axis=-1))
            return (q_pred - target)**2
        
        return jax.vmap(critic_loss_fn)(experiences).mean()
    
    def batch_actor_loss_fn(actor_params, qnet_params, experiences):
        def actor_loss_fn(expr):
            # obs, *_ = experience
            obs = expr['obs']
            action = actor.apply(actor_params, obs)
            q_pred = qnet.apply(qnet_params, jnp.concat((obs, action), axis=-1))
            return -q_pred
        
        return jax.vmap(actor_loss_fn)(experiences).mean()
    
    key, sample_key = random.split(key)
    sampled_experiences = buffer.sample(sample_key, buffer_state)

    # update qnet
    critic_loss, grads = jax.value_and_grad(batch_critic_loss_fn)(qnet_params, target_actor_params, sampled_experiences)
    updates, qnet_opt_state = opt.update(grads, qnet_opt_state)
    qnet_params = optax.apply_updates(qnet_params, updates)

    # update actor
    actor_loss, grads = jax.value_and_grad(batch_actor_loss_fn)(actor_params, qnet_params, sampled_experiences)
    updates, actor_opt_state = opt.update(grads, actor_opt_state)
    actor_params = optax.apply_updates(actor_params, updates)

    update_state = (
        qnet_params, actor_params, 
        qnet_opt_state, actor_opt_state, key
    )
    return  update_state, {'critic_loss': critic_loss, 'actor_loss': actor_loss}


def make_update_model(gamma: float, buffer: ReplayBuffer,
                      qnet: nn.Module, actor: nn.Module, opt: optax.GradientTransformation):
    def update_model_(update_state, buffer_state, target_qnet_params, target_actor_params):
        return update_model(gamma, buffer,
                 qnet, actor, opt, 
                 update_state, buffer_state, target_qnet_params, target_actor_params, )

    def no_update_model_(update_state, buffer_state, target_qnet_params, target_actor_params):
        return  update_state, {'critic_loss': jnp.float32(0.), 'actor_loss': jnp.float32(0.)}
    
    return update_model_, no_update_model_


def make_eval(actor, test_env, num_envs, num_steps, global_steps):
    def eval_(test_key, actor_params, test_env_params):
        return evaluate_continuous_action(
            test_key, actor, actor_params, 
            test_env, test_env_params, 
            num_envs, num_steps, 
            global_steps
        )
    
    def no_eval_(test_key, actor_params, test_env_params):
        return jnp.float32(0.)
    return eval_, no_eval_


def train_one_step(
        config,
        env, test_env, buffer: ReplayBuffer, 
        qnet: nn.Module, actor: nn.Module, opt: optax.GradientTransformation, 
        train_state,
        i_train_step: int):
    
    (
        env_params, env_state, test_env_params,
        buffer_state,
        qnet_params, actor_params, 
        target_qnet_params, target_actor_params,
        qnet_opt_state, actor_opt_state, 
        key
    ) = train_state
    
    rollout_batch_size = config["train_interval"]
    rollout_num_steps = rollout_batch_size // config["num_env"]
    num_updates = config["total_timesteps"] // rollout_batch_size
    learning_start = config["learning_start"]
    target_update_interval = config["target_update_interval"]
    gamma = config["gamma"]
    tau = config["tau"]

    global_steps = rollout_batch_size * i_train_step
    
    # epsilon Schedule
    def get_epsilon(i_step):
        steps_fraction = i_step / (num_updates * config["exploration_fraction"])
        eps = jnp.interp(steps_fraction, 
                         jnp.array([0., 1.]), 
                         jnp.array([config["epsilon_start"], config["epsilon_end"]]))
        return eps
    
    eps_cur = get_epsilon(i_train_step)
        
    is_update_model = global_steps > learning_start
    is_update_target_model = is_update_model & (i_train_step % target_update_interval == 0)
    
    key, rollout_key, test_key = jax.random.split(key, 3)

    env_state, buffer_state = rollout_and_push(rollout_key, 
                     actor, actor_params,
                     env, env_params, env_state, 
                     eps_cur,
                     buffer, buffer_state,
                     rollout_num_steps)

    update_model, no_update_model = make_update_model(gamma, buffer, qnet, actor, opt)

    update_state = (
        qnet_params, actor_params, 
        qnet_opt_state, actor_opt_state, key
    )
    update_state, loss = \
        jax.lax.cond(is_update_model, 
                     update_model, 
                     lambda *_: (update_state, {'critic_loss': jnp.float32(0.), 'actor_loss': jnp.float32(0.)}), 
                     update_state, buffer_state, target_qnet_params, target_actor_params)
    (
        qnet_params, actor_params, 
        qnet_opt_state, actor_opt_state, key
    ) = update_state

    effective_tau = jnp.where(is_update_target_model, tau, 0.0)
    target_qnet_params = jax.tree.map(lambda q, t: q * effective_tau + t * (1 - effective_tau), 
                                      qnet_params, target_qnet_params)
    target_actor_params = jax.tree.map(lambda q, t: q * effective_tau + t * (1 - effective_tau), 
                                       actor_params, target_actor_params)

    # Logging
    # assert config["eval_num_steps"] > env_params.max_steps_in_episode
    global_steps = (i_train_step + 1) * rollout_batch_size
    eval, no_eval = make_eval(actor, test_env, 
                              config["eval_num_env"], config["eval_num_steps"], 
                              global_steps)
    eval_eps_ret_mean = jax.lax.cond(
        global_steps % (config["eval_interval"]) == 0,
        eval, no_eval,
        test_key, actor_params, test_env_params)
    
    # if global_steps % config["eval_interval"] == 0:
    #     if config["wandb"]:
    #         wandb.log({
    #             "train/loss": loss,
    #             "test/reward_mean": test_rew_mean,
    #             "global_steps": global_steps,
    #         })
        
    # elif is_update_model and i_train_step % log_freq == 0:
    #     if config["wandb"]:
    #         wandb.log({"train/loss": loss, "global_steps": global_steps})
    
    train_state = (
        env_params, env_state, test_env_params,
        buffer_state,
        qnet_params, actor_params, 
        target_qnet_params, target_actor_params,
        qnet_opt_state, actor_opt_state, 
        key
    ) 

    return  train_state, eval_eps_ret_mean


def make_train_one_step(config, env, test_env, buffer, qnet, actor, opt):
    def train_one_step_(train_state, i_update_step):
        return train_one_step(config, env, test_env, buffer, qnet, actor, opt, 
                              train_state, i_update_step)
    return train_one_step_


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

    rollout_batch_size = config["train_interval"]
    buffer = make_replay_buffer(
        buffer_size=config["buffer_size"], 
        rollout_batch=rollout_batch_size, 
        sample_batch=config["train_batch_size"],
    )
    dummy_transitions = {
        'obs': jnp.zeros((rollout_batch_size,) + env.observation_space(env_params).shape), 
        'action': jnp.zeros((rollout_batch_size,) + env.action_space(env_params).shape), 
        'rew': jnp.zeros((rollout_batch_size,)),
        'next_obs': jnp.zeros((rollout_batch_size,) + env.observation_space(env_params).shape), 
        'ter':  jnp.zeros((rollout_batch_size,)),
        'tru': jnp.zeros((rollout_batch_size,))
    }
    buffer_state = buffer.init(dummy_transitions)

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


def run_training(config, silent=False):
    run_name = config["env_name"] + "__ddpg__" + datetime.now().strftime('%Y%m%d_%H%M%S')

    if config["wandb"]:
        wandb.init(
            project=config["project_name"],
            name=run_name,
            config=config, 
            monitor_gym=False
        )
    
    # extrac configurations
    rollout_batch_size = config["train_interval"]
    num_train_steps = config["total_timesteps"] // rollout_batch_size

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

    train_state = (
        env_params, env_states, test_env_params,
        buffer_state,
        qnet_params, actor_params, 
        target_qnet_params, target_actor_params,
        qnet_opt_state, actor_opt_state, 
        key
    )

    train_one_step = make_train_one_step(config, env, test_env, buffer, qnet, actor, opt)

    # training loop
    print("start training...")
    start_time = time.time()
    train_state, losses = jax.lax.scan(train_one_step, train_state, jnp.arange(num_train_steps))
    # train_state, eval_eps_ret_means = jax.jit(lambda: jax.lax.scan(train_one_step, train_state, jnp.arange(num_train_steps)))()
    train_state = jax.block_until_ready(train_state)
    print(f"{Fore.BLUE}Training finished in {time.time() - start_time:.2f}s")

    (
        env_params, env_states, test_env_params,
        buffer_state,
        qnet_params, actor_params, 
        target_qnet_params, target_actor_params,
        qnet_opt_state, actor_opt_state, 
        key
    ) = train_state

    return run_name, qnet_params, actor_params


if __name__ == "__main__":
    check_config(config)
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
