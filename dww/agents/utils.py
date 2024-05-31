from dww.wenv import registry
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from dww.loggers import WandbLogger
import numpy as np
import torch

from typing import Optional, List


def no_initter(*args, **kwargs) -> None:
    return None


def get_env(name, render_mode=None, FPS=None):
    """Convenience function to get just the env"""
    env, _, _ = registry(name, render_mode=render_mode, FPS=FPS)
    env = PettingZooEnv(env)
    return env


def get_logger(
    config,
    project: str,
    path: str,
    train_interval: int = 1000,
    update_interval: int = 1000,
    save_interval: int = 1000,
    group: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> WandbLogger:
    """
    Configures and returns a WandbLogger for experiment tracking and logging to TensorBoard.

    Args:
        config: A dictionary containing the experiment's configuration parameters.
        project (str): The name of the Weights & Biases project to log to.
        path (str): The base directory where TensorBoard logs will be saved.
        train_interval (int): The interval (in training steps) at which to log
                              training metrics. Defaults to 1000.
        update_interval (int): The interval (in training steps) at which to
                               update the model's state in the logger. Defaults to 1000.
        save_interval (int): The interval (in training steps) at which to save
                             the model checkpoint. Defaults to 1000.
        group (str): A name for grouping related runs together in WandB.
        tags (List[str]): A list of tags to associate with the run in WandB.

    Returns:
        Tuple[WandbLogger, str]: A tuple containing:
            - The configured WandbLogger instance.
            - The full path to the TensorBoard log directory .
    """

    initargs = {}

    if group is not None:
        initargs["group"] = group

    if tags is not None:
        initargs["tags"] = tags

    logger = WandbLogger(
        project=project,
        train_interval=train_interval,
        update_interval=update_interval,
        save_interval=save_interval,
        wandb_initargs=initargs,
        config=config,
    )
    logger.wandb_run.config["id"] = logger.wandb_run.id
    path = path + "_" + str(logger.wandb_run.id)
    writer = torch.utils.tensorboard.SummaryWriter(path)
    writer.add_text("args", str(config))
    logger.load(writer)
    return logger, path


def set_seeds(seed, *vec_envs):
    """Set all possible seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    for env in vec_envs:
        env.seed(seed)


def dist(*logits):
    """
    Creates a PyTorch Independent distribution representing a multi-variate normal distribution.

    Args:
        *logits: A variable-length argument list containing two tensors:
            - loc (torch.Tensor): The mean parameter.
            - scale (torch.Tensor): The standard deviation parameter.

    Returns:
        torch.distributions.Independent: A PyTorch distribution object representing a multi-variate normal
            distribution with the given mean and standard deviation
    """
    loc, scale = logits
    if torch.any(torch.isnan(loc)):
        loc = np.zeros_like(loc.detach().cpu())

    if torch.any(torch.isnan(scale)):
        scale = np.zeros_like(scale.detach().cpu())
    return torch.distributions.Independent(torch.distributions.Normal(*logits), 1)


def flatten_dict(dd, separator="_", prefix=""):
    """
    Recursively flattens a nested dictionary into a single-level dictionary.

    Args:
        dd (dict): The nested dictionary to be flattened.
        separator (str, optional): The separator character to use between nested keys. Defaults to '_'.
        prefix (str, optional): The current prefix for keys during the recursive flattening. Defaults to an empty string.

    Returns:
        dict: A flattened dictionary where keys represent the paths to the original values.

    """
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def show(env_name, policies):
    """
    Visualizes the behavior of the given policies in the specified environment.

    Args:
        env_name (str): The name of the environment to render.
        policies (Policy or MultiAgentPolicyManager): The policy(ies) to use.
    """
    FPS = 60.0
    show_env = get_env(env_name, "human", FPS)
    show_collector = Collector(policies, DummyVectorEnv([lambda: show_env]))
    show_result = show_collector.collect(n_episode=2, render=1 / FPS)
    print(show_result)
    show_env.close()
