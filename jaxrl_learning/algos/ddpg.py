# Compared with dqn, ddpg needs much larger buffer size and batch size
# because it tries to learn a more complex obs-action mapping. And the actor learning
# is more noisy.

import warnings
warnings.filterwarnings("ignore")
import gymnax
import numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_log_compiles", "1")
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)
from jax import random, numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax import struct
import optax
import time
import wandb

from jaxrl_learning.utils.wrapper import LogWrapper, TerminationTruncationWrapper
from jaxrl_learning.utils.replay_buffer import ReplayBuffer, ReplayBufferState, make_replay_buffer
from jaxrl_learning.utils.rollout import batch_rollout, rollout
from jaxrl_learning.utils.evals import make_eval_continuous
from jaxrl_learning.utils.policy import make_policy
from jaxrl_learning.utils.schedule import epsilon_schedule, noise_std_schedule
from jaxrl_learning.utils.logging import make_log_metrics_callback, make_save_model, log_best_model_artifact
import pprint
from datetime import datetime
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)


#  Config Dictionary
config = {
    "seed": 0,
    "project_name": "jaxrl",
    "env_name": "MountainCarContinuous-v0",
    # "env_name": "Pendulum-v1",
    "total_timesteps": 300_000,
    "features": (128, 64),
    "lr_critic": 1e-3,
    "lr_actor": 1e-3,
    "gamma": 0.99,
    "tau": 0.001,
    "target_update_interval": 1,
    # "tau": 1.,
    # "target_update_interval": 500,
    # "exploration_type": "normal_noise", # "none", "normal_noise", "epsilon_greedy", "ou_noise"
    "exploration_type": "ou_noise", # "none", "normal_noise", "epsilon_greedy", "ou_noise"
    "exploration_noise": 0.5,
    "exploration_noise_end": 0.0,
    "exploration_noise_decay": False,
    "ou_theta": 0.15,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "exploration_fraction": 0.9,
    "num_env": 1,
    "train_interval": 4,
    "train_batch_size": 64,
    "buffer_size": 1e5 ,
    "learning_start": 1e4,
    "eval_interval": 8192,
    "eval_num_steps": 2000,
    "eval_num_env": 16,
    "log_interval": 8192,
    "logging": False,
    "wandb": True,
    "save_model": True,
    "ckpt_path": '/home/zhixin/jaxrl-learning/ckpts/',
    "silent": False
}


def check_config(config):
    assert config["train_interval"] >= config["num_env"]
    assert config["train_interval"] % config["num_env"] == 0
    assert config["total_timesteps"] > config["learning_start"]
    assert config["eval_interval"] > config["train_interval"]
    assert config["eval_interval"] % config["log_interval"] == 0
    assert config["log_interval"] % config["train_interval"] == 0
    _, env_params = gymnax.make(config["env_name"])
    assert config["eval_num_steps"] > env_params.max_steps_in_episode


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


def make_collect(config, env, buffer: ReplayBuffer):
    def collect(key,
                actor_train_state: CustomTrainState,
                env_params: gymnax.EnvParams, env_state: gymnax.EnvState,
                eps_cur: float,
                noise_std: float,
                buffer_state: ReplayBufferState,
                metrics, num_steps: int):
        rollout_keys = jax.random.split(key, env.num_env)

        policy = make_policy(env, env_params, 
                            actor_train_state.apply_fn, actor_train_state.params,
                            config["exploration_type"],
                            eps_cur, 
                            noise_std,
                            config["ou_theta"],
                            1.0)  # dt=1 per env step; keep OU dt out of config for consistency
        policy_has_state = config["exploration_type"] == "ou_noise"
        policy_state = None
        if policy_has_state:
            policy_state = jnp.zeros((env.num_env, *env.action_space(env_params).shape))

        env_state, exprs = batch_rollout(rollout_keys, env, env_state, env_params, policy, num_steps,
                                        policy_state=policy_state, policy_has_state=policy_has_state)
        flat_exprs = jax.tree.map(lambda x: jnp.reshape(x, shape=(-1, *x.shape[2:])), exprs)
        buffer_state = buffer.add(buffer_state, flat_exprs)

        infos = flat_exprs['info']
        metrics = metrics.copy({"charts/episodic_return": jnp.mean(infos['returned_episode_returns']),
                                "charts/episodic_length": jnp.mean(infos['returned_episode_lengths'])})

        return  env_state, buffer_state, metrics

    return collect


def make_update_model(gamma: float, buffer: ReplayBuffer, buffer_state: ReplayBufferState):
    def update_model(actor_train_state: CustomTrainState, 
                     critic_train_state: CustomTrainState, 
                     key: random.PRNGKey):
        critic_params = critic_train_state.params
        actor_params = actor_train_state.params
        target_critic_params = critic_train_state.target_params
        target_actor_params = actor_train_state.target_params

        def batch_critic_loss_fn(critc_params, experiences):
            def critic_loss_fn(expr):
                obs, action, reward, next_obs, termination = \
                    expr['obs'], expr['action'], expr['rew'], expr['next_obs'], expr['ter']
                next_max_action = actor_train_state.apply_fn(target_actor_params, next_obs)
                q_next = critic_train_state.apply_fn(target_critic_params, jnp.concat((next_obs, next_max_action), axis=-1))
                target = reward + gamma * q_next * (1 - termination)
                target = jax.lax.stop_gradient(target)
                q_pred = critic_train_state.apply_fn(critc_params, jnp.concat((obs, action), axis=-1))
                return (q_pred - target)**2

            return jax.vmap(critic_loss_fn)(experiences).mean()

        def batch_actor_loss_fn(actor_params, experiences):
            def actor_loss_fn(expr):
                obs = expr['obs']
                action = actor_train_state.apply_fn(actor_params, obs)
                q_pred = critic_train_state.apply_fn(critic_params, jnp.concat((obs, action), axis=-1))
                return -q_pred

            return jax.vmap(actor_loss_fn)(experiences).mean()

        sampled_experiences = buffer.sample(key, buffer_state)

        # update model
        critic_loss, critic_grads = jax.value_and_grad(batch_critic_loss_fn)(critic_params, sampled_experiences)
        actor_loss, actor_grads = jax.value_and_grad(batch_actor_loss_fn)(actor_params, sampled_experiences)
        actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
        critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
        return  actor_train_state, critic_train_state, {'losses/critic_loss': critic_loss, 'losses/actor_loss': actor_loss}

    return update_model


def make_train_one_step(config, run_name, env, test_env, buffer):
    def train_one_step(train_state, i_train_step: int):
        (
            env_params, env_states, test_env_params, buffer_state,
            actor_train_state, critic_train_state,
            metrics, key
        ) = train_state
        rollout_batch_size = config["train_interval"]
        rollout_num_steps = rollout_batch_size // config["num_env"]
        num_updates = config["total_timesteps"] // rollout_batch_size
        learning_start = config["learning_start"]
        target_update_interval = config["target_update_interval"]
        gamma = config["gamma"]
        tau = config["tau"]
        global_steps = rollout_batch_size * i_train_step
        metrics = metrics.copy({"charts/global_steps": global_steps})

        key, rollout_key, sample_key, test_key = jax.random.split(key, 4)

        # exploration parameter schedule
        eps_cur = epsilon_schedule(i_train_step, config)
        noise_std = noise_std_schedule(i_train_step, config)

        # collect data
        collect = make_collect(config, env, buffer)
        env_states, buffer_state, metrics = collect(rollout_key,
                                                    actor_train_state,
                                                    env_params, env_states,
                                                    eps_cur,
                                                    noise_std,
                                                    buffer_state,
                                                    metrics, rollout_num_steps)

        # update model
        is_update_model = global_steps > learning_start
        is_update_target_model = is_update_model & (i_train_step % target_update_interval == 0)

        update_model = make_update_model(gamma, buffer, buffer_state)
        actor_train_state, critic_train_state, loss = \
            jax.lax.cond(is_update_model,
                        update_model,
                        lambda *_: (actor_train_state, critic_train_state, 
                                    {'losses/critic_loss': jnp.float32(0.), 'losses/actor_loss': jnp.float32(0.)}),
                        actor_train_state, critic_train_state, sample_key)
        
        metrics = metrics.copy(loss)

        # update target model
        effective_tau = jnp.where(is_update_target_model, tau, 0.0)
        actor_train_state = actor_train_state.replace(
            target_params = jax.tree.map(lambda q, t: q * effective_tau + t * (1 - effective_tau),
                                        actor_train_state.params, actor_train_state.target_params)
        )
        critic_train_state = critic_train_state.replace(
            target_params = jax.tree.map(lambda q, t: q * effective_tau + t * (1 - effective_tau),
                                        critic_train_state.params, critic_train_state.target_params)
        )

        # evaluation
        global_steps = (i_train_step + 1) * rollout_batch_size
        eval_policy = make_policy(env, env_params, 
                                actor_train_state.apply_fn, actor_train_state.params,
                                "none")
        model_params_to_save = {
            "actor": actor_train_state.params,
            "critic": critic_train_state.params,
        }
        eval = make_eval_continuous(
            metrics, eval_policy, 
            test_env, test_env_params, 
            config["eval_num_env"], config["eval_num_steps"], global_steps
        )
        is_best_model, metrics = jax.lax.cond(
            global_steps % (config["eval_interval"]) == 0,
            eval,
            lambda *_: (False, metrics),
            test_key)
        
        # logging
        if config["logging"]:
            # save best model
            if config["save_model"]:
                save_model_fn = make_save_model(config, run_name, "best_model")
                jax.lax.cond((global_steps % config["eval_interval"] == 0) & is_best_model,
                            lambda: jax.debug.callback(save_model_fn, model_params_to_save),
                            lambda: None)

            # log metrics
            log_metrics_callback = make_log_metrics_callback(config)
            jax.lax.cond(global_steps % config["log_interval"] == 0,
                        lambda: jax.debug.callback(log_metrics_callback, metrics, global_steps),
                        lambda: None)

        train_state = (
            env_params, env_states, test_env_params, buffer_state,
            actor_train_state, critic_train_state,
            metrics,
            key
        )

        return  train_state, metrics

    return train_one_step


def prepare(key, config):
    key, init_reset_key, critic_init_key, actor_init_key = jax.random.split(key, 4)

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
    _, dummy_state = env.reset(random.key(0), env_params)
    _, dummy_transitions = rollout(random.key(0), env, dummy_state, env_params, 
                                lambda key, obs: env.action_space(env_params).sample(key),
                                rollout_num_steps=rollout_batch_size)
    buffer_state = buffer.init(dummy_transitions)

    critic = QNet(features=config["features"])
    critic_params = critic.init(critic_init_key, jnp.concat((obses, dummy_actions), axis=-1))
    critic_tx = optax.adam(config["lr_critic"])
    critic_train_state = CustomTrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        target_params=critic_params.copy(),
        tx=critic_tx,
    )

    action_lo = env.action_space(env_params).low
    action_hi = env.action_space(env_params).high
    actor = ActorNet(features=config["features"],
                    action_dim=np.prod(env.action_space(env_params).shape),
                    action_scale=(action_hi - action_lo)/2,
                    action_bias=(action_lo + action_hi)/2)
    actor_params = actor.init(actor_init_key, obses)
    actor_tx = optax.adam(config["lr_actor"])
    actor_train_state = CustomTrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        target_params=actor_params.copy(),
        tx=actor_tx,
    )

    test_env, test_env_params = gymnax.make(config["env_name"])
    test_env = TerminationTruncationWrapper(LogWrapper(test_env))

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
    train_state = (
        env_params, env_states, test_env_params, buffer_state,
        actor_train_state, critic_train_state,
        metrics, key
    )
    return env, test_env, buffer, train_state


def make_train():
    def train(config, key):
        # extrac configurations
        rollout_batch_size = config["train_interval"]
        num_train_steps = config["total_timesteps"] // rollout_batch_size
        run_name = config["run_name"]

        # prepare components
        key, init_key = random.split(key)
        env, test_env, buffer, train_state = prepare(init_key, config)

        train_one_step = make_train_one_step(config, run_name, env, test_env, buffer)

        # training loop
        train_state, metrics = jax.lax.scan(train_one_step, train_state, jnp.arange(num_train_steps))

        (
            env_params, env_states, test_env_params, buffer_state,
            actor_train_state, critic_train_state,
            metrics_last, key
        ) = train_state
        return metrics, actor_train_state, critic_train_state

    return train


def main(config):
    """
    Handle random seed and wandb.
    """
    # run_name and wandb
    run_name = config["env_name"] + "__ddpg__" + datetime.now().strftime('%Y%m%d_%H%M%S')
    config = freeze({**config, "run_name": run_name})
    if config["logging"] & config["wandb"]:
        wandb.init(
            project=config["project_name"],
            name=run_name,
            config=unfreeze(config),
        )
    if not config["silent"]:
        print(f"config:\n{pprint.pformat(config)}")

    # training
    key = random.key(config["seed"])
    if not config["silent"]:
        print("start training...")
    start_time = time.time()
    train = jax.jit(make_train(), static_argnames=("config",))
    metrics, actor_train_state, critic_train_state = train(config, key)
    metrics = jax.block_until_ready(metrics)
    if not config["silent"]:
        print(f"{Fore.BLUE}Training finished in {time.time() - start_time:.2f}s")

    if config["logging"] & config["wandb"]:
        log_best_model_artifact()
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
