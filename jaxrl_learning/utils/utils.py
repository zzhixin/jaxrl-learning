#%% Epsilon-greedy policy
from functools import partial
import jax
from jax import numpy as jnp, random
from pathlib import Path
import atexit
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


class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma, theta, dt, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0

    def sample(self, key, x, shape):
        # Based on https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        # dXt = theta * (mu - Xt) * dt + sigma * sqrt(dt) * dWt
        # Approximate with Euler-Maruyama method.
        # Here, x is the previous state.
        noise = self.sigma * jnp.sqrt(self.dt) * random.normal(key, shape=shape)
        x_new = x + self.theta * (self.mu - x) * self.dt + noise
        return x_new, x_new # Return noise and new state


_BEST_MODEL_INFO = {"path": None, "run_name": None, "model_name": None}
_FINALIZER_REGISTERED = False


def _finalize_best_model_artifact():
    try:
        import wandb
    except Exception:
        return
    if wandb.run is None:
        return
    info = _BEST_MODEL_INFO
    if not info["path"] or not Path(info["path"]).exists():
        return
    artifact_name = f"{info['run_name']}-best-model"
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_dir(str(info["path"]))
    wandb.run.log_artifact(artifact, aliases=["best"])


def save_model(config, model_params, run_name, model_name):
    # path = ocp.test_utils.erase_and_create_empty(config["ckpt_path"])
    use_wandb = False
    if config.get("wandb"):
        try:
            import wandb
            use_wandb = wandb.run is not None
        except Exception:
            use_wandb = False
    if use_wandb:
        base_path = Path(wandb.run.dir) / "ckpts"
        model_path = base_path / model_name
    else:
        base_path = Path(config["ckpt_path"]) / run_name
        model_path = base_path / model_name
    base_path.mkdir(parents=True, exist_ok=True)
    import shutil
    if model_path.is_dir():
        shutil.rmtree(model_path)
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(model_path, model_params)
    jax.debug.print(f"{model_name} saved.")
    if use_wandb and model_name == "best_model":
        _BEST_MODEL_INFO["path"] = model_path
        _BEST_MODEL_INFO["run_name"] = run_name
        _BEST_MODEL_INFO["model_name"] = model_name
        global _FINALIZER_REGISTERED
        if not _FINALIZER_REGISTERED:
            atexit.register(_finalize_best_model_artifact)
            _FINALIZER_REGISTERED = True


def log_best_model_artifact():
    _finalize_best_model_artifact()
    

def make_save_model(config, run_name, model_name):
    """
    Usage: 
    ```python
    save_model_fn = make_save_model(config, run_name, model_name)
    jax.debug.callback(save_model_fn, model_params)
    ```
    """
    def save_model_(model_params):
        return save_model(config, model_params, run_name, model_name)
    return save_model_
