#%% Epsilon-greedy policy
from functools import partial
import jax
from jax import numpy as jnp, random
from pathlib import Path
import orbax.checkpoint as ocp


def eps_greedy_policy(key, obs, env, env_params, qnet, qnet_params, eps):
    key, key1, key2 = random.split(key, 3)
    cond = random.uniform(key1) < eps
    rand_action = env.action_space(env_params).sample(key2)
    q_action = qnet.apply(qnet_params, obs).argmax()
    return (rand_action * cond + q_action * (1-cond)).astype(jnp.int32)


def eps_greedy_policy_continuous(key, obs, env, env_params, actor, actor_params, eps):
    key, key1, key2 = random.split(key, 3)
    cond = random.uniform(key1) < eps
    rand_action = env.action_space(env_params).sample(key2)
    action = actor.apply(actor_params, obs)
    return (rand_action * cond + action * (1-cond))


def save_model(config, model_params, run_name, model_name):
    # path = ocp.test_utils.erase_and_create_empty(config["ckpt_path"])
    path = Path(config["ckpt_path"])
    model_path = path / run_name / model_name
    import shutil
    if model_path.is_dir():
        shutil.rmtree(model_path)
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(path / run_name / model_name, model_params)
    jax.debug.print(f"{model_name} saved.")
    
