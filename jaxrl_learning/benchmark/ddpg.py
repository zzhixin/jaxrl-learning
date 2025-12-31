import jax
from jax import numpy as jnp, random
import argparse
from pathlib import Path
import yaml
from jaxrl_learning.algos.ddpg import make_train
from flax.core.frozen_dict import freeze, unfreeze
import wandb
from datetime import datetime

DEFAULT_CONFIG_PATH = Path(__file__).with_name("ddpg_configs.yaml")


def load_configs(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    configs = data.get("configs", data)
    if not isinstance(configs, dict):
        raise ValueError("Config file must contain a mapping of env_name to config.")
    for cfg in configs.values():
        if "features" in cfg and isinstance(cfg["features"], list):
            cfg["features"] = tuple(cfg["features"])
        if "seed" in cfg and isinstance(cfg["seed"], list):
            cfg["seed"] = tuple(cfg["seed"])
    return configs


def run_multi_seeds(config):
    keys = jnp.stack([random.key(seed) for seed in config["seed"]])
    config = freeze(config)
    config = config.copy({"wandb": False, "save_model": False, "run_name": None})

    # train
    train = make_train()
    metrics, *_ = jax.jit(jax.vmap(train, in_axes=(None, 0)), static_argnums=0)(config, keys)

    # print summary
    best_episodic_return = metrics["eval/best_episodic_return"][:,-1]
    average_episodic_return = jnp.nanmean(metrics["eval/episodic_return"], axis=1)
    latest_episodic_return = metrics["eval/episodic_return"][:,-1]
    import pprint
    print("📝 Summary")
    pprint.pp({
        "best episodic return": {
            "mean": f"{jnp.nanmean(best_episodic_return).item():.3f}",
            "std": f"{jnp.sqrt(jnp.nanvar(best_episodic_return)).item():.3f}",
        },
        "average episodic return": {
            "mean": f"{jnp.nanmean(average_episodic_return).item():.3f}",
            "std": f"{jnp.sqrt(jnp.nanvar(average_episodic_return)).item():.3f}",
        },
        "latest episodic return": {
            "mean": f"{jnp.nanmean(latest_episodic_return).item():.3f}",
            "std": f"{jnp.sqrt(jnp.nanvar(latest_episodic_return)).item():.3f}",
        }
    })

    # wandb
    print("📤 Logging into wandb")
    run_name = config["env_name"] + "__ddpg__" + datetime.now().strftime('%Y%m%d_%H%M%S')
    for i, seed in enumerate(config["seed"]):
        i_config = config.copy({"seed": seed})
        wandb.init(
            project="jaxrl",
            name=run_name+f"__{seed}",
            config=unfreeze(i_config),
            tags=["parallel_runs"],
            reinit="finish_previous",
            settings=wandb.Settings(quiet=True),
        )
        wandb.config = {}
        rollout_batch_size = config["train_interval"]
        num_updates = config["total_timesteps"] // rollout_batch_size
        log_per_update = config["log_interval"] // rollout_batch_size
        for t_update in range(0, num_updates, log_per_update):
            global_steps = t_update * rollout_batch_size
            t_metrics = jax.tree.map(lambda data: data[i][t_update], metrics)
            t_metrics = unfreeze(t_metrics)
            if not global_steps % config["eval_interval"] == 0:
                for key in t_metrics.copy():
                    if 'eval' in key:
                        del t_metrics[key]
            wandb.log(t_metrics)
    wandb.finish()



_CONFIGS = load_configs(DEFAULT_CONFIG_PATH)
Pendulum_v1_config = _CONFIGS.get("Pendulum-v1", {})
MountainCarContinuous_v0_config = _CONFIGS.get("MountainCarContinuous-v0", {})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DDPG benchmarks.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to ddpg config YAML.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all env configs.")
    group.add_argument("--single-env", help="Run a single env by name.")
    args = parser.parse_args()

    configs = load_configs(Path(args.config))

    from jaxrl_learning.algos import ddpg

    if args.all:
        for env_name, config in configs.items():
            print(f"🌏 Running env: {env_name}")
            ddpg.check_config(config)
            if len(config["seed"]) > 1:
                run_multi_seeds(config)
            else:
                ddpg.main(config)
    else:
        if args.single_env not in configs:
            raise ValueError(f"Unknown env: {args.single_env}. Available: {list(configs.keys())}")
        config = configs[args.single_env]
        ddpg.check_config(config)
        if len(config["seed"]) > 1:
            run_multi_seeds(config)
        else:
            ddpg.main(config)
