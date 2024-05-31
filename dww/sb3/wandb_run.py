import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from dww.sb3.adaptor import WaterworldAdaptor


def run_environment(
    # core args
    total_timesteps=2e10,
    algorithm=PPO,
    policy="MlpPolicy",
    wandb_project="water-sb3",
    n_envs=10,
    patience=20,
    # non-core extra args
    verbose=1,
    tb_log_folder="runs/",
    model_save_folder="models/",
    gradient_save_freq=200,
    model_save_freq=10000,
    eval_freq=50000,
    wandb_verbose=2,
    device="cuda",
):
    """
    This function sets up and runs a reinforcement learning experiment
    using the Stable Baselines3 (SB3) library, specifically for
    the Waterworld environment.

    Core Arguments:
        - total_timesteps: Total number of timesteps to train for.
        - algorithm: The reinforcement learning algorithm to use (e.g., PPO).
        - policy: The type of policy network (e.g., "MlpPolicy").
        - wandb_project: The name of the Weights & Biases project to log to.
        - n_envs: The number of parallel environments to use for training.
        - patience: The number of evaluations with no improvement before early stopping is triggered.

    Non-Core Arguments:
        - verbose: Verbosity level for the training process.
        - tb_log_folder: Folder to store TensorBoard logs.
        - model_save_folder: Folder to save model checkpoints.
        - gradient_save_freq: Frequency of saving gradients to WandB.
        - model_save_freq: Frequency of saving model checkpoints.
        - eval_freq: Frequency of evaluating the model.
        - wandb_verbose: Verbosity level for WandB logging.
        - device: Device to use for training ("cuda" for GPU or "cpu").
    """
    run = wandb.init(
        project=wandb_project,
        config={
            "algorithm": algorithm.__name__,
            "policy_type": policy,
            "total_timesteps": total_timesteps,
            "env": "Waterworld",
            "device": device,
            "eval_freq": eval_freq,
            "patience": patience,
            "n_envs": n_envs,
        },
        sync_tensorboard=True,
    )
    config = run.config

    vec_env = make_vec_env(
        WaterworldAdaptor,
        n_envs=config["n_envs"],
        vec_env_cls=SubprocVecEnv,
    )
    eval_vec_env = make_vec_env(
        WaterworldAdaptor,
        n_envs=config["n_envs"],
        vec_env_cls=SubprocVecEnv,
    )

    model = algorithm(
        config["policy_type"],
        env=vec_env,
        verbose=verbose,
        tensorboard_log=f"{tb_log_folder}{run.id}",
        device=config["device"],
    )

    save_path = f"{model_save_folder}{run.id}"

    callback_list = []

    if model_save_freq > 0:
        callback_list.append(
            CheckpointCallback(
                save_freq=model_save_freq,
                save_path=save_path,
            )
        )
    stop_train_callback = None
    if config["patience"] >= 0:
        stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=patience, min_evals=20, verbose=1
        )

    if config["eval_freq"] > 0:
        callback_list.append(
            EvalCallback(
                eval_vec_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=max(config["eval_freq"] // config["n_envs"], 1),
                **(
                    {"callback_after_eval": stop_train_callback}
                    if stop_train_callback
                    else {}
                ),
            )
        )

    model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=CallbackList(
            [
                *callback_list,
                WandbCallback(
                    gradient_save_freq=gradient_save_freq,
                    verbose=wandb_verbose,
                ),
            ]
        ),
    )

    run.finish()
