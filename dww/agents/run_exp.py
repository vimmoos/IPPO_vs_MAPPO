from functools import partial
from dataclasses import asdict

from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector
from tianshou.trainer import OnpolicyTrainer

from dww.agents.utils import get_env, get_logger, set_seeds, flatten_dict
from dww.agents.make import make_ppo_policy, make_map_manager, MyFixedPolicy
from dww.agents.mappo.collector import Collector as Col
from dww.agents.hooks import TrainerHooksManager
from dww.agents.config import PPOConf
from dww.utils import sel_and_rm
from tianshou.utils import LazyLogger


def run_exp(
    conf: PPOConf,
    logger: bool = True,
    policy_make_fn=make_ppo_policy,
    manager_make_fn=make_map_manager,
):
    """
    This function runs a complete experiment for training an agent using Proximal Policy Optimization (PPO)
    in a multi-agent or single-agent environment.

    Args:
        conf (PPOConf): Configuration object containing all the hyperparameters and settings for the experiment.
        logger (bool, optional): If True, wandb logging is enabled. Defaults to True.

    Returns:
        None: This function does not return a value, but prints the final results to the console.
    """

    _get_env = partial(get_env, conf.env_name)

    # ======== Step 1: Environment setup =========
    train_envs = SubprocVectorEnv([_get_env for _ in range(conf.n_envs)])
    test_envs = SubprocVectorEnv([_get_env for _ in range(conf.n_envs)])

    set_seeds(conf.seed, train_envs, test_envs)  # Set seeds for reproducibility

    # ======== Step 2: Agent setup =========
    policies = [policy_make_fn(conf) for _ in range(conf.n_agents)]

    if conf.n_agents > 1:  # If multi-agent, add a fixed policy
        policies.append(MyFixedPolicy())

    policy = manager_make_fn(policies, conf.env)

    # ======== Step 3: Collector setup =========
    train_collector = Col(  # Collector for gathering training data
        policy,
        conf.env.agents,
        train_envs,
        conf.cls_buffer(
            buffer_num=len(train_envs),
            **sel_and_rm(asdict(conf._conf), "buffer_"),
        ),
    )

    test_collector = Collector(policy, test_envs)

    # ======== Step 4: Logging Setup =========
    if logger:  # If logging is enabled
        logger, path = get_logger(
            flatten_dict(asdict(conf)),
            **sel_and_rm(asdict(conf._conf), "log_"),
        )
        conf._conf.log_path = path
    else:
        logger = LazyLogger()  # Create a lazy logger if logging is disabled

    # ======== Step 5: Hooks Setup =========
    hooks_man = TrainerHooksManager(
        conf.log_path, conf.env.agents, policy, conf
    )  # Create a hooks manager for training callbacks

    # ======== Step 6: Training =========
    result = OnpolicyTrainer(  # Create and run the PPO trainer
        policy,
        train_collector,
        test_collector,
        logger=logger,
        **sel_and_rm(asdict(conf._conf), "trainer_"),
        **hooks_man.hooks(),
    ).run()

    # ======== Step 7: Print Results =========
    print(f"\n==========Result==========\n{result}")
