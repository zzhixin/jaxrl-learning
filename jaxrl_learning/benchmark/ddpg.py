Pendulum_v1_config = {
    "seed": 0,
    "project_name": "jaxrl",
    "env_name": "Pendulum-v1",
    "total_timesteps": 200_000,
    "features": (128, 64),
    "lr_critic": 1e-3,
    "lr_actor": 5e-4,
    "gamma": 0.99,
    "tau": 0.001,
    "target_update_interval": 1,
    "exploration_noise": 0.1,
    "use_eps_gready": False,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "exploration_fraction": 0.5,
    "num_env": 1,
    "train_interval": 4,
    "train_batch_size": 64,
    "buffer_size": 1e6,
    "learning_start": 1e4,
    "eval_interval": 8192,
    "eval_num_steps": 2000,
    "eval_num_env": 16,
    "log_interval": 8192,
    "wandb": False,
    "ckpt_path": '/home/zhixin/jaxrl-learning/ckpts/'
}


if __name__ == "__main__":
    config = Pendulum_v1_config
    from jaxrl_learning.algos.ddpg import check_config, run_training
    check_config(config)
    run_name, qnet_params, actor_params = run_training(config)
