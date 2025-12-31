from flax.core.frozen_dict import freeze, unfreeze
import jax
import wandb
from pathlib import Path
import atexit
import orbax.checkpoint as ocp


def make_log_metrics_callback(config):
    def log_metrics_callback(metrics, global_steps):
        metrics_to_log = unfreeze(metrics)
        if not global_steps % config["eval_interval"] == 0:
            for key in metrics_to_log.copy():
                if 'eval' in key:
                    del metrics_to_log[key]
        if not config["silent"]:
            print(f"global_steps: {global_steps},  episode_return: {metrics["eval/episodic_return"]}")
        if config["wandb"]:
            wandb.log(metrics_to_log)

    return log_metrics_callback


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
    if not config["silent"]:
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


# def make_logging_and_save(config, run_name):
#     def logging_and_save(global_steps, metrics, model_params_to_save, is_best_model):
#         if config["save_model"]:
#             save_model_fn = make_save_model(config, run_name, "best_model")
#             jax.lax.cond((global_steps % config["eval_interval"] == 0) & is_best_model,
#                         lambda: jax.debug.callback(save_model_fn, model_params_to_save),
#                         lambda: None)

#         log_metrics_callback = make_log_metrics_callback(config)
#         jax.lax.cond(global_steps % config["log_interval"] == 0,
#                     lambda: jax.debug.callback(log_metrics_callback, metrics, global_steps),
#                     lambda: None)
#     return logging_and_save