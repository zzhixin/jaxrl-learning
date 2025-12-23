"""
Compared with dqn, ddpg needs much larger buffer size and batch size
because it tries to learn a more complex obs-action mapping. And the actor learning
is more noisy.
"""
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

from jaxrl_learning.utils.wrapper import LogWrapper, TerminationTruncationWrapper
from jaxrl_learning.utils.replay_buffer import ReplayBuffer, ReplayBufferState, make_replay_buffer
from jaxrl_learning.utils.rollout import batch_rollout
from jaxrl_learning.utils.evals import evaluate_continuous_action, make_eval_and_logging_continuous
import pprint
from datetime import datetime
from colorama import Fore, Style, init
init(autoreset=True)
from jaxrl_learning.benchmark.monitor_recompile import monitor_recompiles
import orbax.checkpoint as ocp
import jax
from pathlib import Path


#  Config Dictionary
config = {
    "project_name": "jaxrl",
    # "env_name": "MountainCarContinuous-v0",
    "env_name": "Pendulum-v1",
    "total_timesteps": 100_000,
    "lr": 5e-3,
    "gamma": 0.99,
    "tau": 0.001,
    "target_update_interval": 1,
    # "tau": 1.,
    # "target_update_interval": 500,
    "exploration_noise": 0.1,
    "use_eps_gready": False,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "exploration_fraction": 0.5,
    "num_env": 1,
    "train_interval": 4,
    "train_batch_size": 64,
    "buffer_size": 1e6,
    "learning_start": 1e4,
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


def make_policy(use_eps_greedy, 
                env, env_params, 
                actor, actor_params, 
                eps_cur=None,
                std=None):
    def noisy_policy(key, obs):
        mean = actor.apply(actor_params, obs)
        lo = env.action_space(env_params).low
        hi = env.action_space(env_params).high
        scale = hi - lo
        noise = random.truncated_normal(key, lo, hi, mean.shape)*std*scale
        return mean + noise
    if std == 0.:
        sub_policy = lambda key, obs: actor.apply(actor_params, obs)
    else:
        sub_policy = noisy_policy

    def eps_greedy_policy(key, obs, sub_policy):
        key, key1, key2, key3 = random.split(key, 4)
        action = sub_policy(key3, obs)
        cond = random.uniform(key1) < eps_cur
        rand_action = env.action_space(env_params).sample(key2)
        return (rand_action * cond + action * (1-cond))

    if use_eps_greedy:
        policy = partial(eps_greedy_policy, sub_policy=noisy_policy)
    else:
        policy = sub_policy

    return policy


def collect(key,
            actor: nn.Module, actor_params,
            env, env_params: gymnax.EnvParams, env_state: gymnax.EnvState,
            eps_cur: float,
            buffer: ReplayBuffer, buffer_state: ReplayBufferState,
            num_steps: int):
    rollout_keys = jax.random.split(key, env.num_env)

    policy = make_policy(config["use_eps_gready"], 
                         env, env_params, 
                         actor, actor_params, eps_cur, 
                         config["exploration_noise"])

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

    return update_model_ 


def train_one_step(
        config, run_name,
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
        eval_eps_ret_mean, best_eps_ret,
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

    # collect data
    env_state, buffer_state = collect(rollout_key,
                     actor, actor_params,
                     env, env_params, env_state,
                     eps_cur,
                     buffer, buffer_state,
                     rollout_num_steps)

    # update model
    update_model = make_update_model(gamma, buffer, qnet, actor, opt)
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

    # update target model
    effective_tau = jnp.where(is_update_target_model, tau, 0.0)
    target_qnet_params = jax.tree.map(lambda q, t: q * effective_tau + t * (1 - effective_tau),
                                      qnet_params, target_qnet_params)
    target_actor_params = jax.tree.map(lambda q, t: q * effective_tau + t * (1 - effective_tau),
                                       actor_params, target_actor_params)

    # evaluation
    global_steps = (i_train_step + 1) * rollout_batch_size
    eval_policy = make_policy(config["use_eps_gready"], 
                              env, env_params, 
                              actor, actor_params, eps_cur, 
                              config["exploration_noise"])
    model_params = {
        "qnet_params": qnet_params,
        "actor_params": actor_params
    }
    eval_and_logging = make_eval_and_logging_continuous(
        config, run_name, best_eps_ret, model_params, eval_policy, 
        test_env, test_env_params, 
        config["eval_num_env"], config["eval_num_steps"], global_steps
    )
    eval_eps_ret_mean, best_eps_ret = jax.lax.cond(
        global_steps % (config["eval_interval"]) == 0,
        eval_and_logging,
        lambda *_: (eval_eps_ret_mean, best_eps_ret),
        test_key)


    train_state = (
        env_params, env_state, test_env_params,
        buffer_state,
        qnet_params, actor_params,
        target_qnet_params, target_actor_params,
        qnet_opt_state, actor_opt_state,
        eval_eps_ret_mean, best_eps_ret,
        key
    )

    return  train_state, eval_eps_ret_mean


def make_train_one_step(config, run_name, env, test_env, buffer, qnet, actor, opt):
    def train_one_step_(train_state, i_update_step):
        return train_one_step(config, run_name, env, test_env, buffer, qnet, actor, opt,
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

    eval_eps_ret_mean = jnp.nan; best_eps_ret = -jnp.inf
    train_state = (
        env_params, env_states, test_env_params,
        buffer_state,
        qnet_params, actor_params,
        target_qnet_params, target_actor_params,
        qnet_opt_state, actor_opt_state,
        eval_eps_ret_mean, best_eps_ret,
        key
    )
    return env, test_env, buffer, qnet, actor, opt, train_state


def run_training(config, silent=False):
    # extrac configurations
    rollout_batch_size = config["train_interval"]
    num_train_steps = config["total_timesteps"] // rollout_batch_size
    run_name = config["env_name"] + "__ddpg__" + datetime.now().strftime('%Y%m%d_%H%M%S')
    if not silent:
        print(f"config:\n{pprint.pformat(config)}")

    # wandb
    if config["wandb"]:
        wandb.init(
            project=config["project_name"],
            name=run_name,
            config=config,
            monitor_gym=False
        )

    # prepare components
    key = random.key(config["seed"])
    key, init_key = random.split(key)
    env, test_env, buffer, qnet, actor, opt, train_state = prepare(init_key, config)

    train_one_step = make_train_one_step(config, run_name, env, test_env, buffer, qnet, actor, opt)

    # training loop
    print("start training...")
    start_time = time.time()
    train_state, eps_ret_means = jax.lax.scan(train_one_step, train_state, jnp.arange(num_train_steps))
    train_state = jax.block_until_ready(train_state)
    print(f"{Fore.BLUE}Training finished in {time.time() - start_time:.2f}s")

    # parse train_state
    (
        env_params, env_states, test_env_params,
        buffer_state,
        qnet_params, actor_params,
        target_qnet_params, target_actor_params,
        qnet_opt_state, actor_opt_state,
        eval_eps_ret_mean, best_eps_ret,
        key
    ) = train_state

    return run_name, qnet_params, actor_params


if __name__ == "__main__":
    check_config(config)
    run_name, qnet_params, actor_params = run_training(config)

