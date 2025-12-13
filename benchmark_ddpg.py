import ddpg
import jax
from jax import random, numpy as jnp
from colorama import Fore, Style, init
init(autoreset=True)


def warmup_inner(train_one_step, rollout_and_push, update_model,
                 train_key, gamma, tau, 
                 qnet, actor, opt, 
                 qnet_params, actor_params, target_qnet_params, target_actor_params,
                 qnet_opt_state, actor_opt_state, 
                 env, env_params, env_states, 
                 buffer, buffer_state,
                 num_steps_per_rollout,):
    eps_cur = jnp.float32(0.5)
    combs = ((False, False), (False, True), (True, False), (True, True))
    for is_update_model, is_update_target_model in combs:

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
    combs = ((False, False), (False, True), (True, False), (True, True))
    for is_update_model, is_update_target_model in combs:

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
    print("warmup finished")
    print(f"{Fore.GREEN}train_one_step compile times: {train_one_step._cache_size()}")
    return qnet_params, target_qnet_params, qnet_opt_state, actor_params, target_actor_params, actor_opt_state, env_states, buffer_state



if __name__ == "__main__":
    # Only jit inner function
    print("--------------- Only jit inner function ----------------")
    print("warmup finished")
    run_name, qnet_params, actor_params = ddpg.run_training(ddpg.config, warmup=warmup_inner)
    print(f"{Fore.GREEN}rollout_and_push compile times: {ddpg.rollout_and_push._cache_size()}")
    print(f"{Fore.GREEN}update_model compile times: {ddpg.update_model._cache_size()}")
        

    # Jit the outmost function train_one_step
    print("\n-------- Jit the outmost function train_one_step ---------")
    ddpg.train_one_step = jax.jit(
        ddpg.train_one_step,
        static_argnames=["qnet", "actor", "opt", "env", 
                         "buffer", "num_steps",
                         "is_update_target_model", "is_update_model"],
        donate_argnames=["buffer_state"])
    print("warmup finished")
    run_name, qnet_params, actor_params = ddpg.run_training(ddpg.config, warmup=warmup_outer)
    print(f"{Fore.GREEN}train_one_step compile times: {ddpg.train_one_step._cache_size()}")
