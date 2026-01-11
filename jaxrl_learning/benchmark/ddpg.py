import jaxrl_learning
import jax
from jax import numpy as jnp, random
import argparse
from pathlib import Path
import yaml
from jaxrl_learning.algos.ddpg import DDPGConfig, make_train
from dataclasses import fields, replace
from typing import get_args, get_origin
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


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(int(item) for item in parts)


def _parse_seed(value: str) -> tuple[int, ...]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(int(item) for item in parts)


def _arg_type_from_field(field_type):
    origin = get_origin(field_type)
    if origin is None:
        return field_type
    if origin is tuple:
        return str
    for arg in get_args(field_type):
        if arg is not type(None):
            return arg
    return str


def add_config_args(parser: argparse.ArgumentParser) -> None:
    for field in fields(DDPGConfig):
        arg_name = f"--{field.name.replace('_', '-')}"
        if field.name == "seed":
            parser.add_argument(
                arg_name,
                type=_parse_seed,
                default=None,
                help="Override seeds as comma-separated ints (e.g., 0,1,2).",
            )
            continue
        if field.name == "features":
            parser.add_argument(
                arg_name,
                type=_parse_int_tuple,
                default=None,
                help="Override features as comma-separated ints (e.g., 256,256).",
            )
            continue
        if field.type is bool:
            parser.add_argument(
                arg_name,
                action=argparse.BooleanOptionalAction,
                default=None,
                help=f"Override {field.name}.",
            )
            continue
        parser.add_argument(
            arg_name,
            type=_arg_type_from_field(field.type),
            default=None,
            help=f"Override {field.name}.",
        )


def apply_overrides(config: DDPGConfig, args: argparse.Namespace) -> DDPGConfig:
    overrides: dict[str, object] = {}
    for field in fields(DDPGConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            overrides[field.name] = value
    if overrides:
        return replace(config, **overrides)
    return config


def print_sweep_summary(metrics):
    # print summary
    import pprint
    best_episodic_return = metrics["eval/best_episodic_return"][:,-1]
    average_episodic_return = jnp.nanmean(metrics["eval/episodic_return"], axis=1)
    latest_episodic_return = metrics["eval/episodic_return"][:,-1]
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


def sweep_seeds(cfg: DDPGConfig):
    """
    Vmap on random seeds. \n
    Metrics are tracked (wandb) after all computation.
    Extra metrics "charts/relative_walltime" will be tracked. 
    Treat parallel runs walltime as each run's walltime.
    """
    keys = jnp.stack([random.key(seed) for seed in cfg.seed])
    cfg = replace(cfg, vmap_run=True)
    if not cfg.wandb:
        confirm = input("The benchmark will not be upload to wandb, metrics can be lost. Are you sure?\n\
1. [Y] Yes, no wandb;\n\
2. [N] No, I will try later\n").strip()
        if confirm == "Y":
            print("Will not upload to wandb")
        else:
            print('Aborted.')
            return 1
    if not cfg.save_model:
        confirm = input("The best model will not be saved. Are you sure?\n\
1. [Y] Yes, no model save;\n\
2. [N] No, I will try later\n").strip()
        if confirm == "Y":
            print("Will not save model")
        else:
            print('Aborted.')
            return 1

    def is_oom_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "resource_exhausted" in msg or "out of memory" in msg

    # train
    time0 = time.perf_counter()
    try:
        train = jax.jit(jax.vmap(make_train(cfg)))
        metrics, _, _, best_model_params = jax.block_until_ready(train(keys))
        elapsed_time = time.perf_counter() - time0
        print_sweep_summary(metrics)
        # wandb
        if cfg.wandb:
            print("📤 Simulate wandb track")
            run_name_prefix = cfg.env_name + "__ddpg"
            run_name_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
            for i, seed in enumerate(cfg.seed):
                i_cfg = replace(cfg, seed=seed)
                run_name = run_name_prefix+f"__{seed}__"+run_name_suffix
                wandb.init(
                    project="jaxrl",
                    name=run_name,
                    config=i_cfg.to_dict(),
                    reinit="finish_previous",
                    settings=wandb.Settings(quiet=True),
                )
                wandb.config = {}
                rollout_batch_size = cfg.update_interval if cfg.update_interval >= cfg.num_env else cfg.num_env
                num_train_steps = cfg.total_timesteps // rollout_batch_size
                logs_per_train_step = cfg.log_interval // rollout_batch_size
                for global_train_step in range(0, num_train_steps, logs_per_train_step):
                    global_steps = global_train_step * rollout_batch_size
                    t_metrics = jax.tree.map(lambda data: data[i][global_train_step], metrics)
                    t_metrics = unfreeze(t_metrics)
                    t_metrics["charts/relative_walltime"] = elapsed_time*global_train_step/num_train_steps
                    if not global_steps % cfg.eval_interval == 0:
                        for key in t_metrics.copy():
                            if 'eval' in key:
                                del t_metrics[key]
                    wandb.log(t_metrics)
                if cfg.save_model:
                    seed_best_params = jax.tree.map(lambda data: data[i], best_model_params)
                    model_path = save_model(i_cfg.to_dict(), seed_best_params, run_name, "best_model")
                    upload_best_model_artifact(model_path, run_name, "best_model")
            wandb.finish()

    except Exception as exc:
        if not is_oom_error(exc):
            raise
        print("⚠️  CUDA OOM during vmap; falling back to sequential seeds.")
        jax.clear_caches()
        train = jax.jit(make_train(cfg))
        run_name_prefix = cfg.env_name + "__ddpg"
        run_name_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_list = []
        for seed in cfg.seed:
            i_cfg = replace(cfg, seed=seed, vmap_run=False)
            ddpg.check_config(i_cfg)
            run_name = run_name_prefix+f"__{seed}__"+run_name_suffix
            if i_cfg.wandb:
                wandb.init(project=i_cfg.project_name, name=run_name, config=i_cfg.to_dict())

            # training
            key = random.key(seed)
            train = jax.jit(make_train(i_cfg))
            metrics, _, _, best_model_params = jax.block_until_ready(train(key))

            if i_cfg.save_model:
                model_path = save_model(i_cfg.to_dict(), best_model_params, run_name, "best_model")
                if i_cfg.wandb:
                    upload_best_model_artifact(model_path, run_name, "best_model")
            if i_cfg.wandb:
                wandb.finish()

            metrics_list.append(metrics)
        all_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *metrics_list)
        print_sweep_summary(all_metrics)


def select_single_seed(config: DDPGConfig) -> DDPGConfig:
    seed = config.seed
    if isinstance(seed, (list, tuple)):
        seed = seed[0]
    return replace(config, seed=seed, vmap_run=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DDPG benchmarks.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to ddpg config YAML.",
    )
    parser.add_argument(
        "--ignore-vmap",
        action="store_true",
        help="Ignore multi-seed vmap and run only the first seed.",
    )
    add_config_args(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all env configs.")
    group.add_argument("--single-env", help="Run a single env by name.")
    args = parser.parse_args()

    configs = load_configs(Path(args.config))

    from jaxrl_learning.algos import ddpg

    if args.all:
        for env_name, config in configs.items():
            print(f"🌏 Running env: {env_name}")
            config = apply_overrides(config, args)
            if args.ignore_vmap:
                config = select_single_seed(config)
            ddpg.check_config(config)
            if not args.ignore_vmap and len(config.seed) > 1:
                sweep_seeds(config)
            else:
                ddpg.main(config)
    else:
        if args.single_env not in configs:
            raise ValueError(f"Unknown env: {args.single_env}. Available: {list(configs.keys())}")
        config = configs[args.single_env]
        config = apply_overrides(config, args)
        if args.ignore_vmap:
            config = select_single_seed(config)
        ddpg.check_config(config)
        if not args.ignore_vmap and len(config.seed) > 1:
            sweep_seeds(config)
        else:
            ddpg.main(config)
