#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import numpy as np
import yaml

import jax
from jax import random
from jax import numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze

from jaxrl_learning.algos.ddpg import check_config, train


def find_default_config_path():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "jaxrl_learning" / "benchmark" / "ddpg_configs.yaml"
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parent / "jaxrl_learning" / "benchmark" / "ddpg_configs.yaml"


DEFAULT_CONFIG_PATH = find_default_config_path()


def parse_args():
    parser = argparse.ArgumentParser(description="Run DDPG (new train) across seeds.")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated seeds.")
    parser.add_argument("--env", default="MountainCarContinuous-v0", help="Env name in config YAML.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    return parser.parse_args()

def load_config(config_path: str, env_name: str):
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    configs = data.get("configs", data)
    if env_name not in configs:
        raise ValueError(f"Env {env_name} not found in config: {config_path}")
    cfg = dict(configs[env_name])
    if "features" in cfg and isinstance(cfg["features"], list):
        cfg["features"] = tuple(cfg["features"])
    return cfg


def run_one(seed, env_name, config_path):
    cfg = load_config(config_path, env_name)
    cfg["seed"] = seed
    cfg["wandb"] = False
    cfg["silent"] = True
    if env_name:
        cfg["env_name"] = env_name
    cfg["run_name"] = f"seed{seed}"
    cfg = freeze(cfg)
    check_config(unfreeze(cfg))
    key = random.key(seed)
    metrics, _, _ = train(cfg, key)
    metrics = jax.block_until_ready(metrics)
    eval_ret = jnp.asarray(metrics["eval/episodic_return"])
    eval_ret = float(jnp.nanmean(eval_ret))
    return eval_ret


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = []
    for seed in seeds:
        ret = run_one(seed, args.env, args.config)
        print(f"seed {seed}: eval_return={ret:.4f}")
        results.append(ret)
    arr = np.array(results, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    nan_count = int(np.isnan(arr).sum())
    print(f"mean={mean:.4f} std={std:.4f} nan_count={nan_count}")


if __name__ == "__main__":
    main()
