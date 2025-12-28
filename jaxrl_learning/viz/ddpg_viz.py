import gymnasium as gym
import gymnax
import time


def gym_visualize(run_name, model_name="best_model"):
    # Visualization
    import numpy as np
    import orbax.checkpoint as ocp
    import jax
    from jax import random, numpy as jnp
    from pathlib import Path
    import gymnasium as gym
    from jaxrl_learning.algos.ddpg import QNet, ActorNet, make_policy
    from jaxrl_learning.benchmark.ddpg import Pendulum_v1_config as config


    key = jax.random.key(config["seed"])
    key, init_reset_key, rollout_key = jax.random.split(key, 3)

    # env
    env, env_params = gymnax.make(config["env_name"])
    obs, state = env.reset(init_reset_key, env_params)
    dummy_action = env.action_space(env_params).sample(random.key(0))


    # dummy model
    qnet = QNet(features=config["features"])
    critic_params = qnet.init(random.key(0), jnp.concat((obs, dummy_action), axis=-1))
    action_lo = env.action_space(env_params).low
    action_hi = env.action_space(env_params).high
    actor = ActorNet(features=config["features"],
                    action_dim=np.prod(env.action_space(env_params).shape),
                    action_scale=(action_hi - action_lo)/2,
                    action_bias=(action_lo + action_hi)/2)
    actor_params = actor.init(random.key(0), obs)
    model_params = {
        "critic": critic_params,
        "actor": actor_params
    }

    # make abstract_model_params and load model
    abstract_model_params = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, model_params)
    checkpointer = ocp.StandardCheckpointer()
    actor_params = checkpointer.restore(
        Path(config["ckpt_path"]) / run_name / model_name,
        abstract_model_params
    )["actor"]

    # make policy
    policy = make_policy(env, env_params, actor.apply, actor_params, use_eps_greedy=False)

    # render in gym environment 
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


if __name__ == "__main__":
    gym_visualize("Pendulum-v1__ddpg__20251226_121228")