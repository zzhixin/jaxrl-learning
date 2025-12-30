import argparse
from pathlib import Path
import yaml

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
    return configs


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

    from jaxrl_learning.algos.ddpg import check_config, main

    if args.all:
        for env_name, config in configs.items():
            print(f"Running env: {env_name}")
            check_config(config)
            main(config)
    else:
        if args.single_env not in configs:
            raise ValueError(f"Unknown env: {args.single_env}. Available: {list(configs.keys())}")
        config = configs[args.single_env]
        check_config(config)
        main(config)
