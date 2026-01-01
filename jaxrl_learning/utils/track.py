from flax.core.frozen_dict import freeze, unfreeze
import jax
import wandb
from pathlib import Path
import orbax.checkpoint as ocp


def make_track_metrics_callback(config):
    def track_metrics_callback(metrics, global_steps):
        metrics_to_track = unfreeze(metrics)
        if not global_steps % config["eval_interval"] == 0:
            for key in metrics_to_track.copy():
                if 'eval' in key:
                    del metrics_to_track[key]
        if config["wandb"]:
            wandb.log(metrics_to_track)

    return track_metrics_callback


def save_model(config, model_params, run_name, model_name):
    # path = ocp.test_utils.erase_and_create_empty(config["ckpt_path"])
    use_wandb = False
    if isinstance(config, dict):
        wandb_enabled = config.get("wandb", False)
    else:
        wandb_enabled = getattr(config, "wandb", False)
    if wandb_enabled:
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
    if not config["silent"]:
        jax.debug.print(f"{model_name} saved.")
    return model_path


def upload_best_model_artifact(model_path, run_name, model_name="best_model"):
    try:
        import wandb
    except Exception:
        return
    if wandb.run is None:
        return
    if not model_path or not Path(model_path).exists():
        return
    artifact_name = f"{run_name}-{model_name.replace('_', '-')}"
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_dir(str(model_path))
    wandb.run.log_artifact(artifact, aliases=["best"])
    

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


def make_track_and_save_callback(config, run_name):
    def track_and_save_callback(global_steps, metrics, model_params_to_save, is_best_model):
        if not config["silent"] and (global_steps % config["log_interval"] == 0):
            print(f"global_steps: {global_steps},  episode_return: {metrics["eval/episodic_return"]}")
        if not config["vmap_run"] and (global_steps % config["log_interval"] == 0):
            track_metrics_callback = make_track_metrics_callback(config)
            track_metrics_callback(metrics, global_steps)
        if not config["vmap_run"] and config["save_model"] and (global_steps % config["eval_interval"] == 0) and is_best_model:
            save_model_fn = make_save_model(config, run_name, "best_model")
            save_model_fn(model_params_to_save)
    return track_and_save_callback
