import argparse
import json
import os
import time
from pathlib import Path
import webbrowser

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import gymnasium as gym
import gymnax
import jax
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
        return None, None, None
    env_name, rest = run_name.split("__ddpg__", 1)
    rest = rest.lstrip("_")
    parts = rest.split("__")
    seed = None
    ts = None
    if len(parts) == 1:
        ts = parts[0]
    elif len(parts) >= 2:
        seed = parts[0]
        ts = parts[1]
    return env_name, ts, seed


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
    env_name, ts, _ = _parse_run_name(run_name)
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
    env_name, ts, _ = _parse_run_name(run_name)
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


def _is_brax_env(env_name: str) -> bool:
    return "brax" in env_name


def _brax_env_name(env_name: str) -> str:
    if env_name.endswith("-brax"):
        env_name = env_name[:-len("-brax")]
    return env_name.lower().replace("-", "_")


def _unstack_pytree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return []
    length = leaves[0].shape[0]
    return [jax.tree_util.tree_map(lambda x, i=i: x[i], tree) for i in range(length)]


def _load_model_params(run_name, model_name, config, entity, project, allow_cloud):
    import orbax.checkpoint as ocp
    from jax import random, numpy as jnp
    from jaxrl_learning.utils.running_mean import RunningMeanStd
    from jaxrl_learning.algos.ddpg import QNet, ActorNet

    key = jax.random.key(config["seed"])
    key, init_reset_key = jax.random.split(key, 2)

    return key, init_reset_key, QNet, ActorNet, ocp


def _restore_model_params(checkpointer, abstract_model_params, run_name, model_name, config, entity, project, allow_cloud):
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
    return checkpointer.restore(ckpt_dir, abstract_model_params)


def gym_visualize(run_name, model_name="best_model", config_path=None, entity=None, project="jaxrl", allow_cloud=True, config=None):
    # Visualization
    import numpy as np
    from jax import random, numpy as jnp
    from jaxrl_learning.algos.ddpg import make_policy_continuous
    from jaxrl_learning.utils.running_mean import RunningMeanStd

    if config is None:
        config = _load_config(run_name, config_path, entity, project, allow_cloud)

    key, init_reset_key, QNet, ActorNet, ocp = _load_model_params(
        run_name, model_name, config, entity, project, allow_cloud
    )
    key, _ = jax.random.split(key, 2)

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
    obs_rms = RunningMeanStd()
    model_params = {
        "critic": critic_params,
        "actor": actor_params,
        "obs_rms_state": obs_rms.init(obs),
    }

    # make abstract_model_params and load model
    abstract_model_params = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, model_params)
    checkpointer = ocp.StandardCheckpointer()
    restored_params = _restore_model_params(
        checkpointer, abstract_model_params, run_name, model_name, config, entity, project, allow_cloud
    )
    actor_params = restored_params["actor"]
    obs_rms_state = restored_params.get("obs_rms_state")

    # make policy
    obs_rms = RunningMeanStd()
    use_norm_obs = bool(config.get("norm_obs", False)) and obs_rms_state is not None
    if config.get("norm_obs", False) and obs_rms_state is None:
        print("Warning: norm_obs=True but obs_rms_state missing in checkpoint; using raw observations.")
    def actor_apply_fn(params, obs):
        if use_norm_obs:
            obs = obs_rms.normalize(obs, obs_rms_state)
        return actor.apply(params, obs)

    policy = make_policy_continuous(
        env,
        env_params,
        actor_apply_fn,
        actor_params,
        exploration_type="none",
    )

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


def brax_html_visualize(
    run_name,
    model_name="best_model",
    config_path=None,
    entity=None,
    project="jaxrl",
    allow_cloud=True,
    steps=None,
    html_path=None,
    config=None,
    open_html=False,
):
    import numpy as np
    from brax import envs
    from brax.io import html
    from jax import random, numpy as jnp

    if config is None:
        config = _load_config(run_name, config_path, entity, project, allow_cloud)
    key, init_reset_key, QNet, ActorNet, ocp = _load_model_params(
        run_name, model_name, config, entity, project, allow_cloud
    )

    env = envs.get_environment(env_name=_brax_env_name(config["env_name"]))
    state = env.reset(init_reset_key)
    ctrl_range = np.array(env.sys.actuator.ctrl_range)
    action_lo = ctrl_range[:, 0]
    action_hi = ctrl_range[:, 1]
    dummy_action = jnp.asarray((action_lo + action_hi) / 2.0, dtype=state.obs.dtype)

    qnet = QNet(features=config["features"])
    critic_params = qnet.init(random.key(0), jnp.concat((state.obs, dummy_action), axis=-1))
    actor = ActorNet(
        features=config["features"],
        action_dim=np.prod(dummy_action.shape),
        action_scale=(action_hi - action_lo) / 2.0,
        action_bias=(action_lo + action_hi) / 2.0,
    )
    actor_params = actor.init(random.key(0), state.obs)
    obs_rms = RunningMeanStd()
    model_params = {
        "critic": critic_params,
        "actor": actor_params,
        "obs_rms_state": obs_rms.init(state.obs),
    }

    abstract_model_params = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, model_params)
    checkpointer = ocp.StandardCheckpointer()
    restored_params = _restore_model_params(
        checkpointer, abstract_model_params, run_name, model_name, config, entity, project, allow_cloud
    )
    actor_params = restored_params["actor"]
    obs_rms_state = restored_params.get("obs_rms_state")

    obs_rms = RunningMeanStd()
    use_norm_obs = bool(config.get("norm_obs", False)) and obs_rms_state is not None
    if config.get("norm_obs", False) and obs_rms_state is None:
        print("Warning: norm_obs=True but obs_rms_state missing in checkpoint; using raw observations.")

    def policy(obs):
        if use_norm_obs:
            obs = obs_rms.normalize(obs, obs_rms_state)
        return actor.apply(actor_params, obs)

    print("rollouting...")
    rollout_steps = int(steps or config.get("eval_num_steps", 1000))

    def scan_step(carry, _):
        action = policy(carry.obs)
        next_state = env.step(carry, action)
        return next_state, next_state.pipeline_state

    final_state, stacked_pipeline = jax.lax.scan(
        scan_step, state, None, length=rollout_steps
    )

    stacked_pipeline = jax.device_get(stacked_pipeline)
    pipeline_states = [jax.device_get(state.pipeline_state)] + _unstack_pytree(stacked_pipeline)

    sys = env.sys.tree_replace({"opt.timestep": env.dt})
    print("rendering...")
    html_str = html.render(sys, pipeline_states)
    if html_path is None:
        html_path = Path.cwd() / "html" / "brax_render.html"
    else:
        html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_str, encoding="utf-8")
    if open_html:
        webbrowser.open(html_path.resolve().as_uri())
    print(f"Wrote brax HTML to {html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a DDPG run.")
    parser.add_argument("--run", required=True, help="Run name to load.")
    parser.add_argument("--model", default="best_model", help="Model name to load.")
    parser.add_argument("--config", default=None, help="Path to a config YAML file.")
    parser.add_argument("--entity", default=None, help="W&B entity for cloud lookup.")
    parser.add_argument("--project", default="jaxrl", help="W&B project name.")
    parser.add_argument("--no-cloud", action="store_true", help="Skip W&B cloud lookup.")
    parser.add_argument("--html", default=None, help="Output path for brax HTML render.")
    parser.add_argument("--steps", type=int, default=None, help="Rollout steps for brax HTML.")
    parser.add_argument("--open-html", action="store_true", help="Open brax HTML in a browser.")
    args = parser.parse_args()
    config = _load_config(args.run, args.config, args.entity, args.project, not args.no_cloud)
    if _is_brax_env(config["env_name"]):
        brax_html_visualize(
            args.run,
            args.model,
            config_path=args.config,
            entity=args.entity,
            project=args.project,
            allow_cloud=not args.no_cloud,
            steps=args.steps,
            html_path=args.html,
            config=config,
            open_html=args.open_html,
        )
    else:
        gym_visualize(
            args.run,
            args.model,
            config_path=args.config,
            entity=args.entity,
            project=args.project,
            allow_cloud=not args.no_cloud,
            config=config,
        )
