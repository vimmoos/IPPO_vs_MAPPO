from dww.agents.config import PPOConf

from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic

from dww.agents.mappo.core import MultiAgentPolicyManager
from dww.agents.mappo.ppo import MAPPOPolicy
from dww.utils import sel_and_rm
from dataclasses import asdict


from dww.agents.make import make_embedder

import os
import joblib
import torch


def make_mappo_policy(conf: PPOConf):
    """
    Creates a Multi-Agent Proximal Policy Optimization (MAPPO) policy with
    associated actor and critic networks.

    Args:
        conf (PPOConf): Configuration object containing hyperparameters
                        for the PPO algorithm and networks.

    Returns:
        MAPPOPolicy: A complete MAPPO policy instance, ready for training
                     and interaction with a multi-agent environment.
    """
    embedder = make_embedder(conf)
    actor = ActorProb(embedder, conf.act_shape, device=conf.device).to(conf.device)
    critic = Critic(
        embedder if conf.input_embedding_shared else make_embedder(conf),
        device=conf.device,
    ).to(conf.device)

    actor_critic = ActorCritic(actor, critic)
    conf.modules_initter(actor_critic)

    return MAPPOPolicy(
        actor,
        critic,
        conf.optim(actor_critic.parameters(), lr=conf.lr),
        action_space=conf.env.action_space,
        **sel_and_rm(asdict(conf), "ppo_"),
    )


def make_mappo_policy_from_file(path: str, agent_id: str):
    """
    Loads a Multi-Agent PPO (MAPPO) policy from a file,
    including its configuration and a customized critic network.

    Args:
        path (str): The directory path containing the saved policy
                     and configuration files.
        agent_id (str): The unique identifier of the agent for which
                        the policy is to be loaded.

    Returns:
        MAPPOPolicy: The loaded MAPPO policy with the customized critic network.
    """
    conf = joblib.load(os.path.join(path, "config.pth"))
    conf = PPOConf(**conf)
    ppo = make_mappo_policy(conf)
    ppo.set_critic(
        Critic(
            Net(
                conf.obs_shape[0] * conf.n_agents,
                hidden_sizes=conf.input_embedding_hidden,
                device=conf.device,
            ).to(conf.device),
            device=conf.device,
        ).to(conf.device)
    )
    di = torch.load(os.path.join(path, f"{agent_id}.pth"))
    ppo.load_state_dict(di)
    return ppo


def make_mappo_manager(conf: PPOConf, agents, env, action_bound_method="tanh"):
    """
    Creates a MultiAgentPolicyManager for managing multiple MAPPO
    policies with a shared critic network.

    Args:
        conf (PPOConf): The configuration object containing hyperparameters
                       for the MAPPO policies.
        agents (list): A list of MAPPOPolicy objects representing the policies
                          of individual agents.
        env (Environment): The environment in which the agents will interact.
        action_bound_method (str): The method used to bound the actions within
                                    a valid range. Defaults to "tanh".

    Returns:
        MultiAgentPolicyManager: An instance of MultiAgentPolicyManager
                                 that handles the interaction of multiple MAPPO
                                 policies with a shared critic network.
    """
    critic = Critic(
        Net(
            conf.obs_shape[0] * conf.n_agents,
            hidden_sizes=conf.input_embedding_hidden,
            device=conf.device,
        ).to(conf.device),
        device=conf.device,
    ).to(conf.device)
    return MultiAgentPolicyManager(
        agents,
        critic,
        env,
        action_bound_method=action_bound_method,
    )
