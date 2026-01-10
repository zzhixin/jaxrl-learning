import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import gymnax
import yaml


def _read_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _as_config(data: dict):
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict) and "value" in value:
            flat[key] = value["value"]
        else:
            flat[key] = value
    if "features" in flat and isinstance(flat["features"], list):
        flat["features"] = tuple(flat["features"])
    return flat


def _select_env_config(configs: dict, run_name: str, source: str):
    if run_name:
        env_name = run_name.split("__ddpg__")[0]
        cfg = configs.get(env_name)
        if cfg is None:
            raise ValueError(f"No config for env {env_name} in {source}.")
        return cfg
    if len(configs) != 1:
        raise ValueError("Config file has multiple envs; provide --run.")
    return next(iter(configs.values()))


def _parse_run_name(run_name: str):
    if "__ddpg__" not in run_name:
        return None, None
    env_name, ts = run_name.split("__ddpg__", 1)
    return env_name, ts


def _load_config_from_yaml(path: Path, run_name: str):
    data = _read_yaml(path)
    configs = data.get("configs", data)
    if not isinstance(configs, dict):
        raise ValueError("Config file must contain a mapping of env_name to config.")
    return _as_config(_select_env_config(configs, run_name, str(path)))


def _load_config_from_wandb_local(run_name: str, root: Path):
    wandb_dir = root / "wandb"
    if not wandb_dir.exists():
        return None
    env_name, ts = _parse_run_name(run_name)
    for run_dir in wandb_dir.glob("run-*"):
        config_path = run_dir / "files" / "config.yaml"
        if not config_path.exists():
            continue
        if ts and run_dir.name.startswith(f"run-{ts}-"):
            cfg = _as_config(_read_yaml(config_path))
            if not env_name or cfg.get("env_name") == env_name:
                return cfg
    return None


def _load_config_from_wandb_cloud(run_name: str, entity: str, project: str):
    import wandb
    import tempfile

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    for run in runs:
        if run.name == run_name:
            cfg = run.config
            if hasattr(cfg, "as_dict"):
                cfg = cfg.as_dict()
            if not isinstance(cfg, dict):
                cfg = getattr(cfg, "data", cfg)
            if isinstance(cfg, dict):
                cfg.pop("_wandb", None)
                return _as_config(cfg)
            if hasattr(run, "json_config") and run.json_config:
                try:
                    cfg = json.loads(run.json_config)
                    return _as_config(cfg)
                except Exception:
                    pass
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    cfg_file = run.file("config.yaml").download(root=tmpdir, replace=True)
                    return _as_config(_read_yaml(Path(cfg_file.name)))
            except Exception:
                pass
            raise ValueError("Unsupported W&B config format.")
    return None


def _infer_entity():
    env_entity = __import__("os").environ.get("WANDB_ENTITY")
    if env_entity:
        return env_entity
    try:
        import wandb
        api = wandb.Api()
        if hasattr(api, "default_entity") and api.default_entity:
            return api.default_entity
    except Exception:
        pass
    return None


def _find_local_ckpt(run_name: str, model_name: str, root: Path):
    wandb_dir = root / "wandb"
    if not wandb_dir.exists():
        return None
    env_name, ts = _parse_run_name(run_name)
    for run_dir in wandb_dir.glob("run-*"):
        config_path = run_dir / "files" / "config.yaml"
        if not config_path.exists():
            continue
        if ts and not run_dir.name.startswith(f"run-{ts}-"):
            continue
        cfg = _as_config(_read_yaml(config_path))
        if env_name and cfg.get("env_name") != env_name:
            continue
        ckpt_dir = run_dir / "files" / "ckpts" / model_name
        if ckpt_dir.exists():
            return ckpt_dir
    return None


def _download_ckpt_from_wandb_cloud(run_name: str, model_name: str, entity: str, project: str):
    import tempfile
    import wandb

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    for run in runs:
        if run.name != run_name:
            continue
        prefix = f"ckpts/{model_name}/"
        tmpdir = Path(tempfile.mkdtemp(prefix="wandb-ckpt-"))
        downloaded = False
        for f in run.files():
            if f.name.startswith(prefix):
                f.download(root=str(tmpdir), replace=True)
                downloaded = True
        if downloaded:
            ckpt_dir = tmpdir / prefix
            if ckpt_dir.exists():
                return ckpt_dir
    return None


def _load_config(run_name: str, config_path: str, entity: str, project: str, allow_cloud: bool):
    repo_root = Path(__file__).resolve().parents[2]
    if config_path:
        return _load_config_from_yaml(Path(config_path), run_name)
    cfg = _load_config_from_wandb_local(run_name, repo_root)
    if cfg is not None:
        return cfg
    if allow_cloud:
        if not entity:
            entity = _infer_entity()
        if not entity:
            raise ValueError("Missing W&B entity for cloud lookup.")
        cfg = _load_config_from_wandb_cloud(run_name, entity, project)
        if cfg is not None:
            return cfg
    raise ValueError("Failed to locate config locally or in W&B cloud.")


def gym_visualize(run_name, model_name="best_model", config_path=None, entity=None, project="jaxrl", allow_cloud=True):
    # Visualization
    import numpy as np
    import orbax.checkpoint as ocp
    import jax
    from jax import random, numpy as jnp
    from jaxrl_learning.algos.ddpg import QNet, ActorNet, make_policy_continuous

    config = _load_config(run_name, config_path, entity, project, allow_cloud)

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
    repo_root = Path(__file__).resolve().parents[2]
    ckpt_dir = _find_local_ckpt(run_name, model_name, repo_root)
    if ckpt_dir is None and allow_cloud:
        if not entity:
            entity = _infer_entity()
        if not entity:
            raise ValueError("Missing W&B entity for cloud checkpoint lookup.")
        ckpt_dir = _download_ckpt_from_wandb_cloud(run_name, model_name, entity, project)
    if ckpt_dir is None:
        ckpt_dir = Path(config["ckpt_path"]) / run_name / model_name
    actor_params = checkpointer.restore(ckpt_dir, abstract_model_params)["actor"]

    # make policy
    policy = make_policy_continuous(env, env_params, actor.apply, actor_params, "none", 0.0, 0.0, 0.0, 1.0)

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
    parser = argparse.ArgumentParser(description="Visualize a DDPG run.")
    parser.add_argument("--run", required=True, help="Run name to load.")
    parser.add_argument("--model", default="best_model", help="Model name to load.")
    parser.add_argument("--config", default=None, help="Path to a config YAML file.")
    parser.add_argument("--entity", default=None, help="W&B entity for cloud lookup.")
    parser.add_argument("--project", default="jaxrl", help="W&B project name.")
    parser.add_argument("--no-cloud", action="store_true", help="Skip W&B cloud lookup.")
    args = parser.parse_args()
    gym_visualize(
        args.run,
        args.model,
        config_path=args.config,
        entity=args.entity,
        project=args.project,
        allow_cloud=not args.no_cloud,
    )
