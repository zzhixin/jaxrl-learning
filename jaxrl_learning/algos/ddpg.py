# Compared with dqn, ddpg needs much larger buffer size and batch size
# because it tries to learn a more complex obs-action mapping. And the actor learning
# is more noisy.

import warnings
warnings.filterwarnings("ignore")
import gymnax
import numpy as np
import jax
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_log_compiles", "1")
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)
from jax import random, numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
from flax import struct
import optax
import time
import wandb
from dataclasses import dataclass, replace
from typing import Tuple, Union

from jaxrl_learning.utils.replay_buffer import ReplayBuffer, ReplayBufferState, make_replay_buffer
from jaxrl_learning.utils.rollout import batch_rollout, rollout
from jaxrl_learning.utils.evals import make_eval_continuous
from jaxrl_learning.utils.policy import make_policy_continuous
from jaxrl_learning.utils.schedule import epsilon_schedule, noise_std_schedule
from jaxrl_learning.utils.track import make_track_and_save_callback, save_model, upload_best_model_artifact
from jaxrl_learning.utils.config import BaseConfig
from jaxrl_learning.utils.env_factory import make_env
from jaxrl_learning.utils.running_mean import RunningMeanStd, RunningMeanStdState

import pprint
from datetime import datetime
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)


@dataclass(frozen=True)
class DDPGConfig(BaseConfig):
    seed: Union[int, Tuple[int, ...]] = 0
    exp_name: str = "ddpg"
    project_name: str = "jaxrl"
    # env_name: str = "MountainCarContinuous-v0"
    env_name: str = "Ant-brax"
    norm_obs: bool = False
    total_timesteps: int = 50_000_000
    features: Tuple[int, ...] = (256, 256)
    lr_critic: float = 3e-4
    lr_actor: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.001
    target_update_interval: int = 1
    exploration_type: str = "normal_noise"
    exploration_noise: float = 0.2
    exploration_noise_end: float = 0.0
    exploration_noise_decay: bool = False
    ou_theta: float = 0.15
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    exploration_fraction: float = 0.9
    num_env: int = 1024
    update_interval: int = 256
    train_batch_size: int = 1024
    buffer_size: float = 1e6
    learning_start: float = 1e5
    eval_interval: int = 1024*4096
    eval_num_steps: int = 2000
    eval_num_env: int = 16
    log_interval: int = 1024*4096
    wandb: bool = False
    save_model: bool = True
    run_name: str | None = None
    ckpt_path: str = "/home/zhixin/jaxrl-learning/ckpts/"
    silent: bool = False
    vmap_run: bool = False


config = DDPGConfig()


def check_config(config: DDPGConfig):
    assert (config.update_interval % config.num_env == 0) | (config.num_env % config.num_env == 0)
    assert config.total_timesteps > config.learning_start
    assert config.eval_interval > config.update_interval
    assert config.eval_interval % config.log_interval == 0
    assert config.log_interval % config.update_interval == 0
    _, env_params = make_env(config.env_name)
    assert config.eval_num_steps > env_params.max_steps_in_episode


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


class CustomTrainState(TrainState):
    target_params: FrozenDict = struct.field(pytree_node=True)


def make_collect(cfg: DDPGConfig, env, buffer: ReplayBuffer, obs_rms: RunningMeanStd, rollout_num_steps: int | jax.Array):
    def collect(key,
                actor_train_state: CustomTrainState,
                env_params: gymnax.EnvParams, env_state: gymnax.EnvState,
                obs_rms_state: RunningMeanStdState,
                eps_cur: float,
                noise_std: float,
                buffer_state: ReplayBufferState,
                metrics):
        rollout_keys = jax.random.split(key, env.num_env)

        policy = make_policy_continuous(env, env_params, 
                                        actor_train_state.apply_fn, actor_train_state.params,
                                        cfg.norm_obs, obs_rms, obs_rms_state,
                                        cfg.exploration_type,
                                        eps_cur, 
                                        noise_std,
                                        cfg.ou_theta,
                                        1.0)  # dt=1 per env step; keep OU dt out of config for consistency
        policy_state = jnp.zeros(())
        if cfg.exploration_type == "ou_noise":
            policy_state = jnp.zeros((env.num_env, *env.action_space(env_params).shape))

        env_state, exprs = batch_rollout(rollout_keys, env, env_state, env_params, policy, rollout_num_steps,
                                         policy_state=policy_state)
        flat_exprs = jax.tree.map(lambda x: jnp.reshape(x, shape=(-1, *x.shape[2:])), exprs)
        if cfg.norm_obs:
            obs_rms_state = obs_rms.update(obs_rms_state, flat_exprs["obs"])
        buffer_state = buffer.add(buffer_state, flat_exprs)

        infos = flat_exprs['info']
        metrics = metrics.copy({"charts/episodic_return": jnp.mean(infos['returned_episode_returns']),
                                "charts/episodic_length": jnp.mean(infos['returned_episode_lengths'])})

        return  env_state, obs_rms_state, buffer_state, metrics

    return collect


def make_update_model(
    cfg: DDPGConfig,
    buffer: ReplayBuffer,
    buffer_state: ReplayBufferState,
    num_updates_per_train_step: int,
    global_train_steps: int,
    obs_rms: RunningMeanStd,
):
    def update_model(actor_train_state: CustomTrainState, 
                     critic_train_state: CustomTrainState, 
                     key: random.PRNGKey,
                     obs_rms_state: RunningMeanStdState):
        def update_step(carry, local_updates):
            actor_train_state, critic_train_state, key = carry
            key, sample_key = random.split(key)
            sampled_experiences = buffer.sample(sample_key, buffer_state)

            # update model
            critic_params = critic_train_state.params
            actor_params = actor_train_state.params
            target_critic_params = critic_train_state.target_params
            target_actor_params = actor_train_state.target_params

            def batch_critic_loss_fn(critc_params, experiences):
                def critic_loss_fn(expr):
                    obs, action, reward, next_obs, termination = \
                        expr['obs'], expr['action'], expr['rew'], expr['next_obs'], expr['ter']
                    if cfg.norm_obs:
                        obs = obs_rms.normalize(obs, obs_rms_state)
                        next_obs = obs_rms.normalize(next_obs, obs_rms_state)
                    next_max_action = actor_train_state.apply_fn(target_actor_params, next_obs)
                    q_next = critic_train_state.apply_fn(target_critic_params, jnp.concat((next_obs, next_max_action), axis=-1))
                    target = reward + cfg.gamma * q_next * (1 - termination)
                    target = jax.lax.stop_gradient(target)
                    q_pred = critic_train_state.apply_fn(critc_params, jnp.concat((obs, action), axis=-1))
                    return (q_pred - target)**2

                return jax.vmap(critic_loss_fn)(experiences).mean()

            def batch_actor_loss_fn(actor_params, experiences):
                def actor_loss_fn(expr):
                    obs = expr['obs']
                    if cfg.norm_obs:
                        obs = obs_rms.normalize(obs, obs_rms_state)
                    action = actor_train_state.apply_fn(actor_params, obs)
                    q_pred = critic_train_state.apply_fn(critic_params, jnp.concat((obs, action), axis=-1))
                    return -q_pred

                return jax.vmap(actor_loss_fn)(experiences).mean()

            critic_loss, critic_grads = jax.value_and_grad(batch_critic_loss_fn)(critic_params, sampled_experiences)
            actor_loss, actor_grads = jax.value_and_grad(batch_actor_loss_fn)(actor_params, sampled_experiences)
            actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
            critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

            # update target model
            global_updates = global_train_steps*num_updates_per_train_step + local_updates
            is_update_target_model = (global_updates % cfg.target_update_interval == 0)
            effective_tau = jnp.where(is_update_target_model, cfg.tau, 0.0)
            actor_train_state = actor_train_state.replace(
                target_params = jax.tree.map(lambda q, t: q * effective_tau + t * (1 - effective_tau),
                                            actor_train_state.params, actor_train_state.target_params)
            )
            critic_train_state = critic_train_state.replace(
                target_params = jax.tree.map(lambda q, t: q * effective_tau + t * (1 - effective_tau),
                                            critic_train_state.params, critic_train_state.target_params)
            )
            return  (actor_train_state, critic_train_state, key), {'losses/critic_loss': critic_loss, 'losses/actor_loss': actor_loss}
        
        carry, loss = jax.lax.scan(
            update_step,
            (actor_train_state, critic_train_state, key),
            jnp.arange(num_updates_per_train_step),
        )
        actor_train_state, critic_train_state, key = carry
        loss = jax.tree.map(lambda m: jnp.mean(m), loss)
        return actor_train_state, critic_train_state, loss 

    return update_model


def make_train_one_step(cfg: DDPGConfig, run_name, env, test_env, buffer, obs_rms):
    def train_one_step(train_state, global_train_steps: int):
        (
            env_params, env_states, test_env_params, 
            obs_rms_state, buffer_state,
            actor_train_state, critic_train_state,
            best_model_params,
            metrics, key
        ) = train_state
        if cfg.update_interval >= cfg.num_env:
            num_updates_per_train_step = 1
            rollout_batch_size = cfg.update_interval
            rollout_num_steps = rollout_batch_size // cfg.num_env
        else:
            rollout_num_steps = 1
            rollout_batch_size = cfg.num_env
            num_updates_per_train_step = cfg.num_env // cfg.update_interval
        global_steps = rollout_batch_size * global_train_steps
        metrics = metrics.copy({"charts/global_steps": global_steps})
        key, rollout_key, sample_key, test_key = jax.random.split(key, 4)

        # exploration parameter schedule
        eps_cur = epsilon_schedule(global_steps, cfg)
        noise_std = noise_std_schedule(global_steps, cfg)

        # collect data
        collect = make_collect(cfg, env, buffer, obs_rms, rollout_num_steps)
        env_states, obs_rms_state, buffer_state, metrics = collect(rollout_key,
                                                                   actor_train_state,
                                                                   env_params, env_states,
                                                                   obs_rms_state,
                                                                   eps_cur,
                                                                   noise_std,
                                                                   buffer_state,
                                                                   metrics)

        # update model
        is_update_model = global_steps > cfg.learning_start
        update_model = make_update_model(cfg, buffer, buffer_state, num_updates_per_train_step, global_train_steps, obs_rms)
        actor_train_state, critic_train_state, loss = \
            jax.lax.cond(is_update_model,
                        update_model,
                        lambda *_: (actor_train_state, critic_train_state, 
                                    {'losses/critic_loss': jnp.float32(0.), 'losses/actor_loss': jnp.float32(0.)}),
                        actor_train_state, critic_train_state, sample_key, obs_rms_state)
        metrics = metrics.copy(loss)

        # evaluation
        global_steps = (global_train_steps + 1) * rollout_batch_size
        eval_policy = make_policy_continuous(env, env_params, 
                                             actor_train_state.apply_fn, actor_train_state.params, 
                                             cfg.norm_obs, obs_rms, obs_rms_state, "none")
        eval = make_eval_continuous(
            metrics, eval_policy, 
            test_env, test_env_params, 
            cfg.eval_num_env, cfg.eval_num_steps, global_steps
        )
        is_best_model, metrics = \
            jax.lax.cond(global_steps % (cfg.eval_interval) == 0,
                         eval,
                         lambda *_: (False, metrics),
                         test_key)
        
        # tracking / saving (disabled in vmap_run)
        model_params_to_save = {
            "actor": actor_train_state.params,
            "critic": critic_train_state.params,
        }
        best_model_params = jax.lax.cond(
            (global_steps % cfg.eval_interval == 0) & is_best_model,
            lambda: model_params_to_save,
            lambda: best_model_params,
        )
        track_and_save_callback = make_track_and_save_callback(cfg, run_name)
        jax.lax.cond(global_steps % cfg.log_interval == 0,
                        lambda: jax.debug.callback(track_and_save_callback, 
                                                global_steps,
                                                metrics, 
                                                model_params_to_save,
                                                is_best_model),
                        lambda: None)

        train_state = (
            env_params, env_states, test_env_params, 
            obs_rms_state, buffer_state,
            actor_train_state, critic_train_state,
            best_model_params,
            metrics, key
        )

        return  train_state, metrics

    return train_one_step


def prepare(key, cfg: DDPGConfig):
    key, init_reset_key, critic_init_key, actor_init_key = jax.random.split(key, 4)

    env, env_params = make_env(cfg.env_name)
    env.num_env = cfg.num_env

    key_resets = random.split(init_reset_key, cfg.num_env)
    obses, env_states = jax.vmap(env.reset, in_axes=(0, None))(key_resets, env_params)
    dummy_act_keys = random.split(random.key(0), cfg.num_env)
    dummy_actions = jax.vmap(env.action_space(env_params).sample)(dummy_act_keys)

    if cfg.update_interval >= cfg.num_env:
        rollout_batch_size = cfg.update_interval
    else:
        rollout_batch_size = cfg.num_env
    buffer = make_replay_buffer(
        buffer_size=cfg.buffer_size,
        rollout_batch=rollout_batch_size,
        sample_batch=cfg.train_batch_size,
    )
    dummy_obs, dummy_state = env.reset(random.key(0), env_params)
    _, dummy_transitions = rollout(random.key(0), env, dummy_state, env_params, 
                                lambda key, obs: env.action_space(env_params).sample(key),
                                rollout_num_steps=rollout_batch_size)
    buffer_state = buffer.init(dummy_transitions)

    obs_rms = RunningMeanStd()
    obs_rms_state = obs_rms.init(dummy_obs)

    critic = QNet(features=cfg.features)
    critic_params = critic.init(critic_init_key, jnp.concat((obses, dummy_actions), axis=-1))
    critic_tx = optax.adam(cfg.lr_critic)
    critic_train_state = CustomTrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        target_params=critic_params.copy(),
        tx=critic_tx,
    )

    action_lo = env.action_space(env_params).low
    action_hi = env.action_space(env_params).high
    actor = ActorNet(features=cfg.features,
                    action_dim=np.prod(env.action_space(env_params).shape),
                    action_scale=(action_hi - action_lo)/2,
                    action_bias=(action_lo + action_hi)/2)
    actor_params = actor.init(actor_init_key, obses)
    actor_tx = optax.adam(cfg.lr_actor)
    actor_train_state = CustomTrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        target_params=actor_params.copy(),
        tx=actor_tx,
    )

    test_env, test_env_params = make_env(cfg.env_name, eval=True)

    metrics = FrozenDict({
        "charts/global_steps": 0,
        "charts/episodic_return": jnp.nan,
        "charts/episodic_length": jnp.nan,
        "losses/actor_loss": jnp.nan,
        "losses/critic_loss": jnp.nan,
        "eval/episodic_return": jnp.nan,
        "eval/episodic_length": jnp.nan,
        "eval/best_episodic_return": -jnp.inf
    })
    best_model_params = {
        "actor": actor_train_state.params,
        "critic": critic_train_state.params,
    }

    train_state = (
        env_params, env_states, test_env_params, 
        obs_rms_state, buffer_state,
        actor_train_state, critic_train_state,
        best_model_params,
        metrics, key
    )
    return env, test_env, buffer, obs_rms, train_state


def make_train():
    def train(cfg: DDPGConfig, key):
        # extrac configurations
        if cfg.update_interval >= cfg.num_env:
            rollout_batch_size = cfg.update_interval
        else:
            rollout_batch_size = cfg.num_env
        num_train_steps = cfg.total_timesteps // rollout_batch_size
        run_name = cfg.run_name

        # prepare components
        key, init_key = random.split(key)
        env, test_env, buffer, obs_rms, train_state = prepare(init_key, cfg)

        train_one_step = make_train_one_step(cfg, run_name, env, test_env, buffer, obs_rms)

        # training loop
        train_state, metrics_log = jax.lax.scan(train_one_step, train_state, jnp.arange(num_train_steps))

        (
            env_params, env_states, test_env_params, 
            obs_rms_state, buffer_state,
            actor_train_state, critic_train_state,
            best_model_params,
            metrics, key
        ) = train_state
        return metrics_log, actor_train_state, critic_train_state, best_model_params

    return train


def main(cfg):
    """
    Handle random seed and wandb.
    """
    # run_name and wandb
    seed = cfg.seed
    if isinstance(seed, (list, tuple)):
        seed = seed[0]
    run_name = cfg.env_name + "__ddpg__" + f"{seed}__" + datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg = replace(cfg, run_name=run_name, seed=seed)
    if not cfg.vmap_run and cfg.wandb:
        wandb.init(
            project=cfg.project_name,
            name=run_name,
            config=cfg.to_dict(),
        )
    if not cfg.silent:
        print(f"config:\n{pprint.pformat(cfg)}")

    # training
    key = random.key(cfg.seed)
    if not cfg.silent:
        print("start training...")
    start_time = time.time()
    train = jax.jit(make_train(), static_argnames=("cfg",))
    metrics, actor_train_state, critic_train_state, best_model_params = train(cfg, key)
    metrics = jax.block_until_ready(metrics)
    if not cfg.silent:
        print(f"{Fore.BLUE}Training finished in {time.time() - start_time:.2f}s")

    if not cfg.vmap_run and cfg.save_model:
        model_path = save_model(cfg.to_dict(), best_model_params, run_name, "best_model")
        if cfg.wandb:
            upload_best_model_artifact(model_path, run_name, "best_model")
    if not cfg.vmap_run and cfg.wandb:
        wandb.finish()
    
    # summary
    best_episodic_return = metrics["eval/best_episodic_return"][-1]
    average_episodic_return = jnp.nanmean(metrics["eval/episodic_return"])
    latest_episodic_return = metrics["eval/episodic_return"][-1]
    pprint.pp({
        "best episodic return": best_episodic_return.item(),
        "average episodic return": average_episodic_return.item(),
        "latest episodic return": latest_episodic_return.item()
    })
    return metrics, actor_train_state, critic_train_state


if __name__ == "__main__":
    check_config(config)
    metrics, *_ = main(config)
