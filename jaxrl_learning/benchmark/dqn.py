import argparse
from pathlib import Path
import yaml
import jax
from jax import numpy as jnp, random
import wandb
from datetime import datetime

from dataclasses import replace
from jaxrl_learning.algos.dqn import DQNConfig, make_train
from jaxrl_learning.utils.track import save_model, upload_best_model_artifact
from flax.core.frozen_dict import unfreeze
jax.config.update("jax_platform_name", "cuda")


DEFAULT_CONFIG_PATH = Path(__file__).with_name("dqn_configs.yaml")


def load_configs(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    configs = data.get("configs", data)
    if not isinstance(configs, dict):
        raise ValueError("Config file must contain a mapping of env_name to config.")
    return {name: DQNConfig.from_dict(cfg) for name, cfg in configs.items()}


def sweep_seeds(cfg: DQNConfig):
    keys = jnp.stack([random.key(seed) for seed in cfg.seed])
    cfg = replace(cfg, wandb=False, run_name=None, vmap_run=True)

    train = make_train()
    metrics, _, best_model_params = jax.jit(jax.vmap(train, in_axes=(None, 0)), static_argnums=0)(cfg, keys)

    best_episodic_return = metrics["eval/best_episodic_return"][:, -1]
    average_episodic_return = jnp.nanmean(metrics["eval/episodic_return"], axis=1)
    latest_episodic_return = metrics["eval/episodic_return"][:, -1]
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

    print("📤 Logging into wandb")
    run_name = cfg.env_name + "__dqn__" + datetime.now().strftime('%Y%m%d_%H%M%S')
    for i, seed in enumerate(cfg.seed):
        i_config = replace(cfg, seed=seed)
        wandb.init(
            project="jaxrl",
            name=run_name+f"__{seed}",
            config=i_config.to_dict(),
            reinit="finish_previous",
            settings=wandb.Settings(quiet=True),
        )
        wandb.config = {}
        rollout_batch_size = cfg.train_interval
        num_updates = cfg.total_timesteps // rollout_batch_size
        log_per_update = cfg.log_interval // rollout_batch_size
        for t_update in range(0, num_updates, log_per_update):
            global_steps = t_update * rollout_batch_size
            t_metrics = jax.tree.map(lambda data: data[i][t_update], metrics)
            t_metrics = unfreeze(t_metrics)
            if not global_steps % cfg.eval_interval == 0:
                for key in list(t_metrics.keys()):
                    if 'eval' in key:
                        del t_metrics[key]
            wandb.log(t_metrics)
        if cfg.save_model:
            seed_best_params = jax.tree.map(lambda data: data[i], best_model_params)
            model_path = save_model(i_config.to_dict(), seed_best_params, run_name+f"__{seed}", "best_model")
            upload_best_model_artifact(model_path, run_name+f"__{seed}", "best_model")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DQN benchmarks.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to dqn config YAML.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all env configs.")
    group.add_argument("--single-env", help="Run a single env by name.")
    args = parser.parse_args()

    configs = load_configs(Path(args.config))

    from jaxrl_learning.algos import dqn

    if args.all:
        for env_name, config in configs.items():
            print(f"🌏 Running env: {env_name}")
            dqn.check_config(config)
            if len(config.seed) > 1:
                sweep_seeds(config)
            else:
                dqn.main(config)
    else:
        if args.single_env not in configs:
            raise ValueError(f"Unknown env: {args.single_env}. Available: {list(configs.keys())}")
        config = configs[args.single_env]
        dqn.check_config(config)
        if len(config.seed) > 1:
            sweep_seeds(config)
        else:
            dqn.main(config)
