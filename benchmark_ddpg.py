import time
import ddpg
import warnings
warnings.filterwarnings("ignore")
import gymnax
import jax
jax.config.update("jax_platform_name", "cpu")
from jax import random, numpy as jnp
from flax import linen as nn
import optax
import time
from replay_buffer import ReplayBuffer, ReplayBufferState
from colorama import Fore, Style, init
init(autoreset=True)


config = {
    "project_name": "jaxrl",
    "env_name": "Pendulum-v1",
    "total_timesteps": 100_000,
    "lr": 2.5e-4,
    "gamma": 0.99,
    "tau": 0.001,
    "target_update_freq": 1,
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
    "features": (128, 64),    
    "seed": 0,
    "log_freq": 1000,          
    "wandb": False,
}

def warmup_inner(train_one_step, rollout_and_push, update_model,
                 train_key, gamma, tau, 
                 qnet, actor, opt, 
                 qnet_params, actor_params, target_qnet_params, target_actor_params,
                 qnet_opt_state, actor_opt_state, 
                 env, env_params, env_states, 
                 buffer, buffer_state,
                 num_steps_per_rollout,):
    eps_cur = jnp.float32(0.5)

    train_key, loss, *train_state = train_one_step(
        train_key, 
        qnet, actor, opt, 
        qnet_params, actor_params, target_qnet_params, target_actor_params,
        qnet_opt_state, actor_opt_state, 
        env, env_params, env_states, 
        eps_cur, 
        gamma, tau, 
        buffer, buffer_state,
        num_steps_per_rollout,
    )
    qnet_params, target_qnet_params, qnet_opt_state, actor_params, target_actor_params, actor_opt_state, env_states, buffer_state = train_state
    print("warmup finished")
    print(f"{Fore.GREEN}rollout_and_push compile times: {rollout_and_push._cache_size()}")
    print(f"{Fore.GREEN}update_model compile times: {update_model._cache_size()}")
    return qnet_params, target_qnet_params, qnet_opt_state, actor_params, target_actor_params, actor_opt_state, env_states, buffer_state

def warmup_outer(train_one_step, rollout_and_push, update_model,
                 train_key, gamma, tau, 
                 qnet, actor, opt, 
                 qnet_params, actor_params, target_qnet_params, target_actor_params,
                 qnet_opt_state, actor_opt_state, 
                 env, env_params, env_states, 
                 buffer, buffer_state,
                 num_steps_per_rollout,):
    eps_cur = jnp.float32(0.5)
    train_key, loss, *train_state = train_one_step(
        train_key, 
        qnet, actor, opt, 
        qnet_params, actor_params, target_qnet_params, target_actor_params,
        qnet_opt_state, actor_opt_state, 
        env, env_params, env_states, 
        eps_cur, 
        gamma, tau, 
        buffer, buffer_state,
        num_steps_per_rollout,
    )
    qnet_params, target_qnet_params, qnet_opt_state, actor_params, target_actor_params, actor_opt_state, env_states, buffer_state = train_state
    print("warmup finished")
    print(f"{Fore.GREEN}train_one_step compile times: {train_one_step._cache_size()}")
    return qnet_params, target_qnet_params, qnet_opt_state, actor_params, target_actor_params, actor_opt_state, env_states, buffer_state


def train_one_step(
        key, 
        qnet: nn.Module, actor: nn.Module, opt: optax.GradientTransformation, 
        qnet_params, actor_params, target_qnet_params, target_actor_params,
        qnet_opt_state: optax.OptState, actor_opt_state: optax.OptState, 
        env, env_params: gymnax.EnvParams, env_state: gymnax.EnvState, 
        eps_cur: jax.Array,
        gamma: float, tau: float, 
        buffer: ReplayBuffer, buffer_state: ReplayBufferState,
        num_steps: int):
    
    key, rollout_key, sample_key = jax.random.split(key, 3)

    env_state, buffer_state = ddpg.rollout_and_push(rollout_key, 
                     actor, actor_params,
                     env, env_params, env_state, 
                     eps_cur,
                     buffer, buffer_state,
                     num_steps)

    loss = jnp.array(0.0)
    sampled_experiences = buffer.sample(sample_key, buffer_state)
    loss, qnet_params, target_qnet_params, qnet_opt_state, actor_params, target_actor_params, actor_opt_state = \
        ddpg.update_model(sampled_experiences,
                    qnet, actor, opt, 
                    qnet_params, actor_params, target_qnet_params, target_actor_params,
                    qnet_opt_state, actor_opt_state, 
                    gamma
                    )
    return  key, loss, qnet_params, target_qnet_params, qnet_opt_state, actor_params, target_actor_params, actor_opt_state, env_state, buffer_state


def benchmark(config, warmup, n_iters=1000):
    
    # extrac configurations
    num_steps_per_rollout = config["train_freq"] // config["num_env"]
    gamma = config["gamma"]
    tau = config["tau"]

    key = random.key(config["seed"])
    key, init_key, train_key, test_key = random.split(key, 4)
    (
        env, env_params, env_states, 
        test_env, test_env_params, 
        buffer, buffer_state, 
        qnet, actor, opt, 
        qnet_params, actor_params, target_qnet_params, target_actor_params,
        qnet_opt_state, actor_opt_state,  
    )  =  ddpg.prepare(init_key, config)
    
    (
        qnet_params, target_qnet_params, qnet_opt_state, 
        actor_params, target_actor_params, actor_opt_state, 
        env_states, buffer_state
    ) = warmup(train_one_step, ddpg.rollout_and_push, ddpg.update_model,
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

    for iter in range(n_iters):
        eps_cur = jnp.float32(0.5)
        train_key, loss, *train_state = train_one_step(
            train_key, 
            qnet, actor, opt, 
            qnet_params, actor_params, target_qnet_params, target_actor_params,
            qnet_opt_state, actor_opt_state, 
            env, env_params, env_states, 
            eps_cur, 
            gamma, tau, 
            buffer, buffer_state,
            num_steps_per_rollout,
        )
        (qnet_params, target_qnet_params, qnet_opt_state, 
         actor_params, target_actor_params, actor_opt_state, 
         env_states, buffer_state) = jax.block_until_ready(train_state)

    qnet_params = jax.block_until_ready(qnet_params)
    print(f"{Fore.BLUE}Average time {(time.time() - start_time)/n_iters*1e3:.2f}ms")



if __name__ == "__main__":
    # Only jit inner function
    print("--------------- Only jit inner function ----------------")
    benchmark(config, warmup=warmup_inner)
    print(f"{Fore.GREEN}rollout_and_push compile times: {ddpg.rollout_and_push._cache_size()}")
    print(f"{Fore.GREEN}update_model compile times: {ddpg.update_model._cache_size()}")
        

    # Jit the outmost function train_one_step
    print("\n-------- Jit the outmost function train_one_step ---------")
    train_one_step = jax.jit(
        train_one_step,
        static_argnames=["qnet", "actor", "opt", "env", 
                         "buffer", "num_steps"],
        donate_argnames=["buffer_state"])
    benchmark(config, warmup=warmup_outer)
    print(f"{Fore.GREEN}train_one_step compile times: {train_one_step._cache_size()}")


    # # Only jit inner function
    # print("--------------- Only jit inner function ----------------")
    # run_name, qnet_params, actor_params = ddpg.run_training(config, warmup=warmup_inner, silent=True)
    # print(f"{Fore.GREEN}rollout_and_push compile times: {ddpg.rollout_and_push._cache_size()}")
    # print(f"{Fore.GREEN}update_model compile times: {ddpg.update_model._cache_size()}")
        

    # # Jit the outmost function train_one_step
    # print("\n-------- Jit the outmost function train_one_step ---------")
    # ddpg.train_one_step = jax.jit(
    #     ddpg.train_one_step,
    #     static_argnames=["qnet", "actor", "opt", "env", 
    #                      "buffer", "num_steps",
    #                      "is_update_target_model", "is_update_model"],
    #     donate_argnames=["buffer_state"])
    # run_name, qnet_params, actor_params = ddpg.run_training(config, warmup=warmup_outer, silent=True)
    # print(f"{Fore.GREEN}train_one_step compile times: {ddpg.train_one_step._cache_size()}")


