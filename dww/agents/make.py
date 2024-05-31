from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.policy import PPOPolicy, MultiAgentPolicyManager
from dataclasses import asdict
from tianshou.policy.random import RandomPolicy

import numpy as np
from typing import Optional, Union, Any
from tianshou.data import Batch
import os
import torch

from dww.agents.config import PPOConf, MinimumConf
from dww.utils import sel_and_rm
import joblib
from dww.agents.utils import show


class MyRandPolicy(RandomPolicy):
    """
    This class represents a custom random policy that inherits from the base `RandomPolicy` class.
    It's designed to sample random actions from the provided action space.
    """

    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        act = [self.action_space.sample() for _ in range(len(batch))]
        return Batch(act=np.asarray(act))


class MyFixedPolicy(RandomPolicy):
    """
    This class represents a custom "fixed" policy that always returns the same action.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        act = [np.array([0.5, 0.5], dtype=np.float32) for _ in range(len(batch))]
        return Batch(act=np.asarray(act))


def make_embedder(conf: MinimumConf):
    """
    Creates a neural network for processing input observations.

    Args:
        conf (MinimumConf): Configuration object containing hyperparameters for the embedder, such as:
            - obs_shape: The shape of the input observations.
            - input_embedding_hidden: List of hidden layer sizes for the embedder network.
            - device: The device (e.g., 'cpu' or 'cuda') on which the model should run.

    Returns:
        Net: A neural network model (instance of the `Net` class)
             This model is moved to the specified device for computation.
    """
    return Net(
        conf.obs_shape,
        hidden_sizes=conf.input_embedding_hidden,
        device=conf.device,
    ).to(conf.device)


def make_ppo_policy_from_file(path: str, agent_id: str):
    """
    Loads a trained PPO (Proximal Policy Optimization) policy from a file and its associated configuration.

    Args:
        path (str): The path to the directory where the policy and configuration files are saved.
        agent_id (str): A unique identifier for the agent whose policy you want to load.

    Returns:
        tuple: A tuple containing:
            - ppo (Policy): The loaded PPO policy object.
            - conf (PPOConf): The configuration object used to train the policy.

    """
    conf = joblib.load(os.path.join(path, "config.pth"))
    conf = PPOConf(**conf)
    ppo = make_ppo_policy(conf)
    ppo.load_state_dict(torch.load(os.path.join(path, f"{agent_id}.pth")))
    return ppo, conf


def make_ppo_policy(conf: PPOConf):
    """
    Creates a Proximal Policy Optimization (PPO) policy with associated actor and critic networks.

    Args:
        conf (PPOConf): Configuration object containing hyperparameters for the PPO algorithm and networks.

    Returns:
        PPOPolicy: A complete PPO policy instance.
    """

    embedder = make_embedder(conf)
    actor = ActorProb(embedder, conf.act_shape, device=conf.device).to(conf.device)
    critic = Critic(
        embedder if conf.input_embedding_shared else make_embedder(conf),
        device=conf.device,
    ).to(conf.device)

    actor_critic = ActorCritic(actor, critic)
    conf.modules_initter(actor_critic)

    return PPOPolicy(
        actor,
        critic,
        conf.optim(actor_critic.parameters(), lr=conf.lr),
        action_space=conf.env.action_space,
        **sel_and_rm(asdict(conf), "ppo_"),
    )


def make_map_manager(agents, env, action_bound_method="tanh"):
    """
    Creates a MultiAgentPolicyManager for managing multiple agent policies within a multi-agent environment.

    Args:
        agents (list): A list of policy objects, where each policy corresponds to a different agent.
        env (Environment): The multi-agent environment where the policies will interact.
        action_bound_method (str, optional): The method used to bound actions within a valid range.
                                             Defaults to 'tanh'.

    Returns:
        MultiAgentPolicyManager: An instance of the MultiAgentPolicyManager.
    """
    return MultiAgentPolicyManager(
        agents,
        env,
        action_bound_method=action_bound_method,
    )


def make_random_policy(conf: PPOConf):
    return MyRandPolicy(conf.env.action_space)


def load_and_show_single(path):
    """
    Loads trained policies from a specified directory, creates a multi-agent policy manager,
    and then displays the multi-agent policy's behavior in the environment.

    Args:
        path (str): The path to the directory containing the trained policies and their configuration.
    """
    agents_path = sorted(
        [
            f[:-4]
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f != "config.pth"
        ]
    )
    # print(sorted(agents_path))

    conf = joblib.load(os.path.join(path, "config.pth"))
    conf = PPOConf(**conf)

    policies = [make_ppo_policy_from_file(path, p) for p in agents_path]

    policy = make_map_manager(
        conf,
        policies,
        conf.env,
    )
    show(conf.env_name, policy)
