from dataclasses import dataclass, field, asdict
import gymnasium as gym
import torch
from typing import Callable, List, Type, Optional
from tianshou.data import ReplayBuffer, VectorReplayBuffer
from dww.agents.utils import get_env, no_initter
from dww.agents.var import VarRef


@dataclass
class MinimumConf:
    env: Optional[gym.Env] = None
    env_name: str = "default_single"
    device: str = "cuda"

    optim: torch.optim.Optimizer = torch.optim.Adam
    lr: float = 1e-3
    modules_initter: Callable[..., None] = no_initter

    input_embedding_hidden: List[int] = field(default_factory=lambda: [64, 64])
    input_embedding_shared: bool = True

    seed: int = 42
    n_envs: int = 10
    n_agents: int = 1

    cls_buffer: Type[ReplayBuffer] = VectorReplayBuffer
    # buffer_alpha: float = 0.6
    # buffer_beta: float = 0.4
    buffer_total_size: int = 100_000

    log_path: str = ""
    log_project: str = "test_ppo"
    log_group: str = "ti-learns"
    log_tags: List[str] = field(default_factory=lambda: ["ti"])

    trainer_max_epoch: int = 20
    trainer_step_per_epoch: int = 20480
    trainer_repeat_per_collect: int = 10
    trainer_episode_per_test: int = 10
    trainer_batch_size: int = 512
    trainer_episode_per_collect: int = 16

    def __post_init__(self):
        if isinstance(self.optim, str):
            self.optim = VarRef(self.optim).resolve()

        if isinstance(self.modules_initter, str):
            self.modules_initter = VarRef(self.modules_initter).resolve()

        if self.log_path == "":
            self.log_path = f"log/{self.env_name}/ppo"
        if self.env is None:
            self.env = get_env(self.env_name)

    @property
    def obs_shape(self):
        return self.env.observation_space.shape or self.env.observation_space.n

    @property
    def act_shape(self):
        return self.env.action_space.shape or self.env.action_space.n

    def dump(self):
        ret = asdict(self)
        del ret["env"]

        ret["optim"] = ret["optim"].__module__ + ":" + ret["optim"].__name__
        ret["modules_initter"] = (
            ret["modules_initter"].__module__ + ":" + ret["modules_initter"].__name__
        )
        return ret


@dataclass
class PPOConf:
    _conf: MinimumConf

    ppo_dist_fn: Type[torch.distributions.Distribution]

    actor_hidden: List[int] = field(default_factory=list)
    critic_hidden: List[int] = field(default_factory=list)

    ppo_discount_factor: float = 0.95
    ppo_eps_clip: float = 0.4
    ppo_dual_clip: Optional[float] = None
    ppo_value_clip: bool = True
    ppo_advantage_normalization: bool = True
    ppo_recompute_advantage: bool = False
    ppo_vf_coef: float = 0.25
    ppo_ent_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_gae_lambda: float = 0.95
    ppo_reward_normalization: bool = True

    def __post_init__(self):
        if isinstance(self.ppo_dist_fn, str):
            self.ppo_dist_fn = VarRef(self.ppo_dist_fn).resolve()

        if isinstance(self._conf, dict):
            self._conf = MinimumConf(**self._conf)

    def __getattribute__(self, what: str):
        try:
            ret = object.__getattribute__(self, what)
        except AttributeError:
            ret = object.__getattribute__(self._conf, what)
        return ret

    def dump(self):
        ret = asdict(self)
        ret["_conf"] = self._conf.dump()

        ret["ppo_dist_fn"] = (
            ret["ppo_dist_fn"].__module__ + ":" + ret["ppo_dist_fn"].__name__
        )

        return ret
