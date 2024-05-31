from dataclasses import dataclass
from typing import Optional, List
import torch
import os
from tianshou.policy import MultiAgentPolicyManager
from dww.agents.config import PPOConf

import joblib


@dataclass
class TrainerHooksManager:
    """
    This dataclass manages training hooks for a multi-agent reinforcement learning setup using PPO.

    Attributes:
        log_path (str): The directory where logs and checkpoints will be saved.
        agents_id (List[str]): A list of agent IDs involved in the training.
        policy (MultiAgentPolicyManager): Manages the policies of multiple agents.
        conf (PPOConf): Configuration object holding settings for the PPO algorithm and the experiment.
        rew_thresh (float, optional): Reward threshold to trigger stopping the training.
        nbest_policy (int): Counter to keep track of the number of "best" policies saved.

    """

    log_path: str
    agents_id: List[str]
    policy: MultiAgentPolicyManager
    conf: PPOConf
    rew_thresh: Optional[float] = None
    nbest_policy: int = 0

    # REMEBER TO REGISTER
    USE_HOOKS = [
        "stop_fn",
        "reward_metric",
        "save_best_fn",
        "save_checkpoint_fn",
    ]

    def save_best_fn(self, policy):
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        policy_path = os.path.join(self.log_path, f"policy_{self.nbest_policy}")
        if not os.path.exists(policy_path):
            os.mkdir(policy_path)
        for k, pol in policy.policies.items():
            torch.save(
                pol.state_dict(),
                os.path.join(policy_path, f"{k}.pth"),
            )
        joblib.dump(self.conf.dump(), os.path.join(policy_path, "config.pth"))

        self.nbest_policy += 1

    def save_checkpoint_fn(self, epoch, env_step, gradient_step):
        ckpt_path = os.path.join(self.log_path, f"checkpoint_{epoch}_{env_step}")
        os.mkdir(ckpt_path)

        for k, pol in self.policy.policies.items():
            torch.save(
                pol.state_dict(),
                os.path.join(ckpt_path, f"policy_{k}.pth"),
            )
        return ckpt_path

    def stop_fn(self, mean_rewards):
        if self.rew_thresh is not None:
            return mean_rewards >= self.rew_thresh

    def reward_metric(self, rews):
        """Define a global reward measure across agents"""
        if rews.shape[1] > 1:
            return rews[:, :-1].mean(axis=1)
        return rews.mean(axis=1)

    def hooks(self):
        return {k: getattr(self, k) for k in dir(self) if k in self.USE_HOOKS}
