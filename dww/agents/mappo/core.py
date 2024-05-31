## ================NOTE====================
## MultiAgentPolicyManager implementation from Tianshou library with some minor modification
## to adjust it to our use case

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.env.pettingzoo_env import PettingZooEnv
from dww.agents.mappo.ppo import MAPPOPolicy


class MultiAgentPolicyManager(BasePolicy):
    """Multi-agent policy manager for MARL.

    This multi-agent policy manager accepts a list of
    :class:`~tianshou.policy.BasePolicy`. It dispatches the batch data to each
    of these policies when the "forward" is called. The same as "process_fn"
    and "learn": it splits the data and feeds them to each policy. A figure in
    :ref:`marl_example` can help you better understand this procedure.
    """

    def __init__(
        self,
        policies: List[MAPPOPolicy],
        ccritic: torch.nn.Module,
        env: PettingZooEnv,
        **kwargs: Any,
    ) -> None:
        super().__init__(action_space=env.action_space, **kwargs)
        assert len(policies) == len(
            env.agents
        ), "One policy must be assigned for each agent."

        self.agent_idx = env.agent_idx
        for i, policy in enumerate(policies):
            # agent_id 0 is reserved for the environment proxy
            # (this MultiAgentPolicyManager)
            policy.set_agent_id(env.agents[i])

        self.policies = dict(zip(env.agents, policies))
        if len(env.agents) > 1:
            self.learn_pols = dict(zip(env.agents[:-1], policies[:-1]))
        else:
            self.learn_pols = self.policies

        for policy in self.learn_pols.values():
            policy.set_critic(ccritic)

        self.env = env

    def last_agent(self):
        k = self.env.agents[-1]
        return k, self.policies[k]

    def critic_inp(self, batch):
        idxs = [np.nonzero(batch.obs.agent_id == k)[0] for k in self.learn_pols.keys()]
        return (
            np.hstack([batch[idx].obs.obs for idx in idxs]),
            np.hstack([batch[idx].obs_next.obs for idx in idxs]),
        )

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to every policy's process_fn.

        Save original multi-dimensional rew in "save_rew", set rew to the
        reward of each agent during their "process_fn", and restore the
        original reward afterwards.
        """
        results = {}
        # reward can be empty Batch (after initial reset) or nparray.
        has_rew = isinstance(buffer.rew, np.ndarray)
        critic_inp, critic_inp_next = self.critic_inp(batch)

        if has_rew:
            save_rew, buffer._meta.rew = buffer.rew, Batch()

        for agent, policy in self.learn_pols.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent)[0]
            if len(agent_index) == 0:
                results[agent] = Batch()
                continue
            tmp_batch, tmp_indice = batch[agent_index], indice[agent_index]
            # print(tmp_batch)
            if has_rew:
                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent]]
                buffer._meta.rew = save_rew[:, self.agent_idx[agent]]
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, "obs"):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, "obs"):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            tmp_batch.critic_inp = critic_inp
            tmp_batch.critic_inp_next = critic_inp_next
            results[agent] = policy.process_fn(tmp_batch, buffer, tmp_indice)

        k, p = self.last_agent()
        results[k] = batch[np.nonzero(batch.obs.agent_id == k)[0]]

        if has_rew:  # restore from save_rew
            buffer._meta.rew = save_rew
        return Batch(results)

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch]] = None,
        **kwargs: Any,
    ) -> Batch:
        results: List[
            Tuple[bool, np.ndarray, Batch, Union[np.ndarray, Batch], Batch]
        ] = []
        for agent_id, policy in self.policies.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent_id)[0]
            if len(agent_index) == 0:
                # (has_data, agent_index, out, act, state)
                results.append((False, np.array([-1]), Batch(), Batch(), Batch()))
                continue
            tmp_batch = batch[agent_index]
            if isinstance(tmp_batch.rew, np.ndarray):
                # reward can be empty Batch (after initial reset) or nparray.
                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent_id]]
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, "obs"):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, "obs"):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            out = policy(
                batch=tmp_batch,
                state=None if state is None else state[agent_id],
                **kwargs,
            )
            act = out.act
            each_state = (
                out.state
                if (hasattr(out, "state") and out.state is not None)
                else Batch()
            )
            results.append((True, agent_index, out, act, each_state))
        holder = Batch.cat(
            [
                {"act": act}
                for (has_data, agent_index, out, act, each_state) in results
                if has_data
            ]
        )
        state_dict, out_dict = {}, {}
        for (agent_id, _), (has_data, agent_index, out, act, state) in zip(
            self.policies.items(), results
        ):
            if has_data:
                holder.act[agent_index] = act
            state_dict[agent_id] = state
            out_dict[agent_id] = out
        holder["out"] = out_dict
        holder["state"] = state_dict
        return holder

    def learn(
        self, batch: Batch, **kwargs: Any
    ) -> Dict[str, Union[float, List[float]]]:
        """Dispatch the data to all policies for learning.

        :return: a dict with the following contents:

        ::

            {
                "agent_1/item1": item 1 of agent_1's policy.learn output
                "agent_1/item2": item 2 of agent_1's policy.learn output
                "agent_2/xxx": xxx
                ...
                "agent_n/xxx": xxx
            }
        """
        results = {}
        info = np.hstack([batch[idx].obs for idx in self.learn_pols.keys()])
        info_next = np.hstack([batch[idx].obs_next for idx in self.learn_pols.keys()])
        for agent_id, policy in self.learn_pols.items():
            data = batch[agent_id]
            if not data.is_empty():
                data.critic_inp = info
                data.critic_inp_next = info_next
                out = policy.learn(batch=data, **kwargs)
                for k, v in out.items():
                    results[agent_id + "/" + k] = v
        return results
