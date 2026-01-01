# DQN refactor aligned with DDPG structure and new replay buffer API.
import warnings
warnings.filterwarnings("ignore")

import gymnax
import jax
jax.config.update("jax_platform_name", "cpu")
from jax import random, numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
from flax import struct
import optax
import time
import wandb
from functools import partial
import pprint
from datetime import datetime
from dataclasses import dataclass, replace
from typing import Tuple, Union

from jaxrl_learning.utils.wrapper import LogWrapper, TerminationTruncationWrapper
from jaxrl_learning.utils.replay_buffer import ReplayBuffer, ReplayBufferState, make_replay_buffer
from jaxrl_learning.utils.rollout import batch_rollout, rollout
from jaxrl_learning.utils.schedule import epsilon_schedule
from jaxrl_learning.utils.evals import make_eval_discrete
from jaxrl_learning.utils.track import make_track_and_save_callback, save_model, upload_best_model_artifact
from jaxrl_learning.utils.config import BaseConfig


@dataclass(frozen=True)
class DQNConfig(BaseConfig):
    project_name: str = "jaxrl"
    exp_name: str = "dqn"
    env_name: str = "CartPole-v1"
    total_timesteps: int = 200_000
    lr: float = 2.5e-4
    gamma: float = 0.99
    tau: float = 0.005
    target_update_freq: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    exploration_fraction: float = 0.5
    num_env: int = 1
    train_interval: int = 4
    train_batch_size: int = 128
    buffer_size: float = 1e4
    learning_start: float = 1e4
    eval_interval: int = 8192
    eval_num_steps: int = 1000
    eval_num_env: int = 16
    log_interval: int = 8192
    features: Tuple[int, ...] = (128, 64)
    seed: Union[int, Tuple[int, ...]] = 0
    wandb: bool = True
    save_model: bool = True
    run_name: str | None = None
    ckpt_path: str = "/home/zhixin/jaxrl-learning/ckpts/"
    silent: bool = False
    vmap_run: bool = False

config = DQNConfig()


def check_config(cfg: DQNConfig):
    assert cfg.train_interval >= cfg.num_env
    assert cfg.train_interval % cfg.num_env == 0
    assert cfg.total_timesteps > cfg.learning_start
    assert cfg.eval_interval > cfg.train_interval
    assert cfg.eval_interval % cfg.log_interval == 0
    assert cfg.log_interval % cfg.train_interval == 0
    _, env_params = gymnax.make(cfg.env_name)
    assert cfg.eval_num_steps >= env_params.max_steps_in_episode


class QNet(nn.Module):
    features: tuple
    num_actions: int

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return x


class CustomTrainState(TrainState):
    target_params: FrozenDict = struct.field(pytree_node=True)


def make_policy(env, env_params, qnet_apply_fn, qnet_params, eps_cur):
    def eps_greedy_policy_fn(key, obs, epsilon):
        key, key1, key2 = random.split(key, 3)
        cond = random.uniform(key1) < epsilon
        rand_action = env.action_space(env_params).sample(key2)
        q_action = qnet_apply_fn(qnet_params, obs).argmax()
        return (rand_action * cond + q_action * (1 - cond)).astype(jnp.int32)

    return partial(eps_greedy_policy_fn, epsilon=eps_cur)


def collect(cfg: DQNConfig, key,
            q_train_state: CustomTrainState,
            env, env_params: gymnax.EnvParams, env_state: gymnax.EnvState,
            eps_cur: float,
            buffer: ReplayBuffer, buffer_state: ReplayBufferState,
            metrics, num_steps: int):
    rollout_keys = jax.random.split(key, env.num_env)
    policy = make_policy(env, env_params, q_train_state.apply_fn, q_train_state.params, eps_cur)
    env_state, exprs = batch_rollout(rollout_keys, env, env_state, env_params, policy, num_steps)
    flat_exprs = jax.tree.map(lambda x: jnp.reshape(x, shape=(-1, *x.shape[2:])), exprs)
    buffer_state = buffer.add(buffer_state, flat_exprs)

    infos = flat_exprs["info"]
    metrics = metrics.copy({"charts/episodic_return": jnp.mean(infos["returned_episode_returns"]),
                            "charts/episodic_length": jnp.mean(infos["returned_episode_lengths"])})
    return env_state, buffer_state, metrics


def update_model(gamma: float, buffer: ReplayBuffer, buffer_state,
                 q_train_state: CustomTrainState, key: random.PRNGKey):
    q_params = q_train_state.params
    target_q_params = q_train_state.target_params

    sampled_experiences = buffer.sample(key, buffer_state)

    def batch_loss_fn(params, experiences):
        def loss_fn(expr):
            obs, action, reward, next_obs, termination = \
                expr["obs"], expr["action"], expr["rew"], expr["next_obs"], expr["ter"]
            action = action.astype(jnp.int32)
            q_next = jnp.max(q_train_state.apply_fn(target_q_params, next_obs), axis=-1)
            target = reward + gamma * q_next * (1 - termination)
            target = jax.lax.stop_gradient(target)
            q_pred_all = q_train_state.apply_fn(params, obs)
            q_pred = jnp.take_along_axis(q_pred_all, action[..., None], axis=-1)[..., 0]
            return (q_pred - target) ** 2

        return jax.vmap(loss_fn)(experiences).mean()

    q_loss, q_grads = jax.value_and_grad(batch_loss_fn)(q_params, sampled_experiences)
    q_train_state = q_train_state.apply_gradients(grads=q_grads)
    return q_train_state, {"losses/q_loss": q_loss}


def make_update_model(gamma: float, buffer: ReplayBuffer, buffer_state: ReplayBufferState):
    def update_model_(q_train_state, key):
        return update_model(gamma, buffer, buffer_state, q_train_state, key)
    return update_model_


def train_one_step(
        cfg: DQNConfig, run_name,
        env, test_env, buffer: ReplayBuffer,
        train_state,
        i_train_step: int):
    (
        env_params, env_states, test_env_params, buffer_state,
        q_train_state,
        best_model_params,
        metrics, key
    ) = train_state

    rollout_batch_size = cfg.train_interval
    rollout_num_steps = rollout_batch_size // cfg.num_env

    global_steps = rollout_batch_size * i_train_step
    metrics = metrics.copy({"charts/global_steps": global_steps})

    eps_cur = epsilon_schedule(i_train_step, cfg)

    is_update_model = global_steps > cfg.learning_start
    is_update_target = is_update_model & (i_train_step % cfg.target_update_freq == 0)

    key, rollout_key, sample_key, test_key = jax.random.split(key, 4)

    env_states, buffer_state, metrics = collect(
        cfg, rollout_key, q_train_state,
        env, env_params, env_states,
        eps_cur, buffer, buffer_state,
        metrics, rollout_num_steps
    )

    update_model_fn = make_update_model(cfg.gamma, buffer, buffer_state)
    q_train_state, loss = jax.lax.cond(
        is_update_model,
        update_model_fn,
        lambda *_: (q_train_state, {"losses/q_loss": jnp.float32(0.)}),
        q_train_state, sample_key
    )
    metrics = metrics.copy(loss)

    effective_tau = jnp.where(is_update_target, cfg.tau, 0.0)
    q_train_state = q_train_state.replace(
        target_params=jax.tree.map(
            lambda q, t: q * effective_tau + t * (1 - effective_tau),
            q_train_state.params, q_train_state.target_params
        )
    )

    global_steps = (i_train_step + 1) * rollout_batch_size
    eval_policy = make_policy(env, env_params, q_train_state.apply_fn, q_train_state.params, 0.0)
    eval_fn = make_eval_discrete(
        metrics, eval_policy,
        test_env, test_env_params,
        cfg.eval_num_env, cfg.eval_num_steps,
        global_steps
    )
    is_best_model, metrics = jax.lax.cond(
        global_steps % cfg.eval_interval == 0,
        eval_fn,
        lambda *_: (False, metrics),
        test_key
    )

    model_params_to_save = {"q": q_train_state.params}
    best_model_params = jax.lax.cond(
        (global_steps % cfg.eval_interval == 0) & is_best_model,
        lambda: model_params_to_save,
        lambda: best_model_params,
    )

    track_and_save_callback = make_track_and_save_callback(cfg, run_name)
    jax.lax.cond(
        global_steps % cfg.log_interval == 0,
        lambda: jax.debug.callback(
            track_and_save_callback,
            global_steps,
            metrics,
            model_params_to_save,
            is_best_model,
        ),
        lambda: None
    )

    train_state = (
        env_params, env_states, test_env_params, buffer_state,
        q_train_state,
        best_model_params,
        metrics, key
    )
    return train_state, metrics


def make_train_one_step(cfg, run_name, env, test_env, buffer):
    def train_one_step_(train_state, i_update_step):
        return train_one_step(cfg, run_name, env, test_env, buffer, train_state, i_update_step)
    return train_one_step_


def prepare(key, cfg: DQNConfig):
    key, init_reset_key, q_init_key = jax.random.split(key, 3)

    env, env_params = gymnax.make(cfg.env_name)
    env = TerminationTruncationWrapper(LogWrapper(env))
    env.num_env = cfg.num_env

    key_resets = random.split(init_reset_key, cfg.num_env)
    obses, env_states = jax.vmap(env.reset, in_axes=(0, None))(key_resets, env_params)

    rollout_batch_size = cfg.train_interval
    buffer = make_replay_buffer(
        buffer_size=cfg.buffer_size,
        rollout_batch=rollout_batch_size,
        sample_batch=cfg.train_batch_size,
    )
    _, dummy_state = env.reset(random.key(0), env_params)
    _, dummy_transitions = rollout(
        random.key(0), env, dummy_state, env_params,
        lambda key, obs: env.action_space(env_params).sample(key),
        rollout_num_steps=rollout_batch_size
    )
    buffer_state = buffer.init(dummy_transitions)

    qnet = QNet(features=cfg.features, num_actions=env.num_actions)
    q_params = qnet.init(q_init_key, obses)
    q_tx = optax.adam(cfg.lr)
    q_train_state = CustomTrainState.create(
        apply_fn=qnet.apply,
        params=q_params,
        target_params=q_params.copy(),
        tx=q_tx,
    )

    test_env, test_env_params = gymnax.make(cfg.env_name)
    test_env = TerminationTruncationWrapper(LogWrapper(test_env))

    metrics = FrozenDict({
        "charts/global_steps": 0,
        "charts/episodic_return": jnp.nan,
        "charts/episodic_length": jnp.nan,
        "losses/q_loss": jnp.nan,
        "eval/episodic_return": jnp.nan,
        "eval/episodic_length": jnp.nan,
        "eval/best_episodic_return": -jnp.inf
    })
    best_model_params = {"q": q_train_state.params}
    train_state = (
        env_params, env_states, test_env_params, buffer_state,
        q_train_state,
        best_model_params,
        metrics, key
    )
    return env, test_env, buffer, train_state


def train(config: DQNConfig, key):
    rollout_batch_size = config.train_interval
    num_train_steps = config.total_timesteps // rollout_batch_size
    run_name = config.run_name

    key, init_key = random.split(key)
    env, test_env, buffer, train_state = prepare(init_key, config)
    train_one_step = make_train_one_step(config, run_name, env, test_env, buffer)
    train_state, metrics = jax.lax.scan(train_one_step, train_state, jnp.arange(num_train_steps))
    (
        env_params, env_states, test_env_params, buffer_state,
        q_train_state,
        best_model_params,
        metrics_last, key
    ) = train_state
    return metrics, q_train_state, best_model_params


def make_train():
    return train


def main(config: DQNConfig):
    run_name = config.env_name + "__dqn__" + datetime.now().strftime('%Y%m%d_%H%M%S')
    config = replace(config, run_name=run_name)

    if not config.vmap_run and config.wandb:
        wandb.init(
            project=config.project_name,
            name=run_name,
            config=config.to_dict(),
        )
    if not config.silent:
        print(f"config:\n{pprint.pformat(config)}")

    seed = config.seed
    if isinstance(seed, (list, tuple)):
        seed = seed[0]
    key = random.key(seed)
    if not config.silent:
        print("start training...")
    start_time = time.time()
    train_fn = jax.jit(make_train(), static_argnames=("config",))
    metrics, q_train_state, best_model_params = train_fn(config, key)
    metrics = jax.block_until_ready(metrics)
    if not config.silent:
        print(f"Training finished in {time.time() - start_time:.2f}s")

    if not config.vmap_run and config.save_model:
        model_path = save_model(config.to_dict(), best_model_params, run_name, "best_model")
        if config.wandb:
            upload_best_model_artifact(model_path, run_name, "best_model")
    if not config.vmap_run and config.wandb:
        wandb.finish()

    return metrics, q_train_state


if __name__ == "__main__":
    check_config(config)
    main(config)
