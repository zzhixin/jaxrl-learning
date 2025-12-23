# import gymnasium as gym
# import gymnax
# import time


# def gym_visualize(config, policy)
#     # Visualization
#     from gymnax.visualize import Visualizer
#     import numpy as np
#     import orbax.checkpoint as ocp
#     import jax
#     from pathlib import Path
#     import gymnasium as gym


#     key = jax.random.key(config["seed"])
#     key, init_reset_key, rollout_key = jax.random.split(key, 3)

#     env, env_params = gymnax.make(config["env_name"])
#     obs, state = env.reset(init_reset_key, env_params)
#     dummy_action = env.action_space(env_params).sample(random.key(0))


#     qnet = QNet(features=config["features"])
#     qnet_params = qnet.init(random.key(0), jnp.concat((obs, dummy_action), axis=-1))
#     action_lo = env.action_space(env_params).low
#     action_hi = env.action_space(env_params).high
#     actor = ActorNet(features=config["features"],
#                     action_dim=np.prod(env.action_space(env_params).shape),
#                     action_scale=(action_hi - action_lo)/2,
#                     action_bias=(action_lo + action_hi)/2)
#     actor_params = actor.init(random.key(0), obs)
#     model_params = {
#         "qnet_params": qnet_params,
#         "actor_params": actor_params
#     }

#     abstract_model_params = jax.tree_util.tree_map(
#         ocp.utils.to_shape_dtype_struct, model_params)
#     checkpointer = ocp.StandardCheckpointer()
#     actor_params = checkpointer.restore(
#         Path(config["ckpt_path"]) / run_name,
#         abstract_model_params
#     )["actor_params"]

#     policy = partial(eps_greedy_policy_continuous,
#                      env=env, env_params=env_params,
#                      actor=actor, actor_params=actor_params,
#                      eps=0.)



#     gym_env = gym.make(config["env_name"], render_mode="human")
#     obs, _ = gym_env.reset(seed=0)

#     while True:
#         key, key_act = jax.random.split(key)
#         action = np.array(policy(key_act, obs))
#         next_obs, reward, ter, tru, info = gym_env.step(action)

#         done = ter or tru

#         if done:
#             time.sleep(1)
#             gym_env.close()
#             break
#         else:
#           obs = next_obs
#           gym_env.render()
#         #   time.sleep(0.05)


#     # # use gymnax's visualizer
#     # state_seq, reward_seq = [], []
#     # while True:
#     #     state_seq.append(state)
#     #     key, key_act, key_step = jax.random.split(key, 3)
#     #     action = policy(key_act, obs)
#     #     next_obs, next_state, reward, done, info = env.step(
#     #         key_step, state, action, env_params
#     #     )
#     #     reward_seq.append(reward)

#     #     if done:
#     #         break
#     #     else:
#     #       obs = next_obs
#     #       state = next_state

#     # cum_rewards = jnp.cumsum(jnp.array(reward_seq))
#     # print(len(state_seq), len(cum_rewards))

#     # print("generating gif...")
#     # vis = Visualizer(env, env_params, state_seq, cum_rewards)
#     # vis.animate(f"gif/anim.gif")