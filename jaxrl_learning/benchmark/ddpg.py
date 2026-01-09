import jax
from jax import numpy as jnp, random
import argparse
from pathlib import Path
import yaml
from jaxrl_learning.algos.ddpg import DDPGConfig, make_train
from dataclasses import replace
from flax.core.frozen_dict import unfreeze
import wandb
from datetime import datetime
from jaxrl_learning.utils.track import save_model, upload_best_model_artifact
import time
jax.config.update("jax_platform_name", "cuda")

DEFAULT_CONFIG_PATH = Path(__file__).with_name("ddpg_configs.yaml")


def load_configs(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    configs = data.get("configs", data)
    if not isinstance(configs, dict):
        raise ValueError("Config file must contain a mapping of env_name to config.")
    return {name: DDPGConfig.from_dict(cfg) for name, cfg in configs.items()}


def sweep_seeds(config: DDPGConfig):
    """
    Vmap on random seeds. \n
    Metrics are tracked (wandb) after all computation.
    Extra metrics "charts/relative_walltime" will be tracked. 
    Treat parallel runs walltime as each run's walltime.
    """
    keys = jnp.stack([random.key(seed) for seed in config.seed])
    config = replace(config, vmap_run=True)
    if not config.wandb:
        confirm = input("The benchmark will not be upload to wandb, metrics can be lost. Are you sure?\n\
1. [Y] Yes, no wandb;\n\
2. [N] No, I will try later\n").strip()
        if confirm == "Y":
            print("Will not upload to wandb")
        else:
            print('Aborted.')
            return 1
    if not config.save_model:
        confirm = input("The best model will not be saved. Are you sure?\n\
1. [Y] Yes, no model save;\n\
2. [N] No, I will try later\n").strip()
        if confirm == "Y":
            print("Will not save model")
        else:
            print('Aborted.')
            return 1

    # train
    time0 = time.perf_counter()
    train = jax.jit(jax.vmap(make_train(), in_axes=(None, 0)), static_argnums=0)
    metrics, _, _, best_model_params = train(config, keys)
    metrics = jax.block_until_ready(metrics)
    elapsed_time = time.perf_counter() - time0

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
    if config.wandb:
        print("📤 Simulate wandb track")
        run_name_prefix = config.env_name + "__ddpg"
        run_name_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        for i, seed in enumerate(config.seed):
            i_config = replace(config, seed=seed)
            run_name = run_name_prefix+f"__{seed}__"+run_name_suffix
            wandb.init(
                project="jaxrl",
                name=run_name,
                config=i_config.to_dict(),
                reinit="finish_previous",
                settings=wandb.Settings(quiet=True),
            )
            wandb.config = {}
            rollout_batch_size = config.update_interval
            num_updates = config.total_timesteps // rollout_batch_size
            log_per_update = config.log_interval // rollout_batch_size
            for t_update in range(0, num_updates, log_per_update):
                global_steps = t_update * rollout_batch_size
                t_metrics = jax.tree.map(lambda data: data[i][t_update], metrics)
                t_metrics = unfreeze(t_metrics)
                t_metrics["charts/relative_walltime"] = elapsed_time*t_update/num_updates
                if not global_steps % config.eval_interval == 0:
                    for key in t_metrics.copy():
                        if 'eval' in key:
                            del t_metrics[key]
                wandb.log(t_metrics)
            if config.save_model:
                seed_best_params = jax.tree.map(lambda data: data[i], best_model_params)
                model_path = save_model(i_config.to_dict(), seed_best_params, run_name, "best_model")
                upload_best_model_artifact(model_path, run_name, "best_model")
        wandb.finish()


_CONFIGS = load_configs(DEFAULT_CONFIG_PATH)
Pendulum_v1_config = _CONFIGS.get("Pendulum-v1", DDPGConfig())
MountainCarContinuous_v0_config = _CONFIGS.get("MountainCarContinuous-v0", DDPGConfig())


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
            if len(config.seed) > 1:
                sweep_seeds(config)
            else:
                ddpg.main(config)
    else:
        if args.single_env not in configs:
            raise ValueError(f"Unknown env: {args.single_env}. Available: {list(configs.keys())}")
        config = configs[args.single_env]
        ddpg.check_config(config)
        if len(config.seed) > 1:
            sweep_seeds(config)
        else:
            ddpg.main(config)
