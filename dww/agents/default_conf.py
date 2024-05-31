from dww.agents.config import PPOConf, MinimumConf
from dww.agents.utils import dist, no_initter

from tianshou.data import VectorReplayBuffer
import torch

default_min = dict(
    log_project="final_ppo",
    device="cuda",
    optim=torch.optim.Adam,
    lr=1e-3,
    modules_initter=no_initter,
    input_embedding_hidden=[64, 64],
    input_embedding_shared=True,
    seed=42,
    n_envs=40,
    cls_buffer=VectorReplayBuffer,
    buffer_total_size=100_000,
    trainer_max_epoch=100,
    trainer_step_per_epoch=20480,
    trainer_repeat_per_collect=10,
    trainer_episode_per_test=10,
    trainer_batch_size=512,
    trainer_episode_per_collect=16,
)
default_ppo = dict(
    ppo_dist_fn=dist,
    ppo_discount_factor=0.95,
    ppo_eps_clip=0.2,
    ppo_dual_clip=None,
    ppo_value_clip=True,
    ppo_advantage_normalization=True,
    ppo_recompute_advantage=False,
    ppo_vf_coef=0.25,
    ppo_ent_coef=0.0,
    ppo_max_grad_norm=0.5,
    ppo_gae_lambda=0.95,
    ppo_reward_normalization=True,
    actor_hidden=[],
    critic_hidden=[],
)

conf_single_ppo = PPOConf(
    MinimumConf(
        env_name="default_single",
        log_group="single_ppo",
        log_tags=["ppo", "single"],
        n_agents=1,
        **default_min,
    ),
    **default_ppo,
)
conf_large_single_ppo = PPOConf(
    MinimumConf(
        env_name="default_large_single",
        log_group="large_single_ppo",
        log_tags=["ppo", "single", "large"],
        n_agents=1,
        **default_min,
    ),
    **default_ppo,
)

conf_double_ppo = PPOConf(
    MinimumConf(
        env_name="default_env",
        log_group="double_ppo",
        log_tags=["ppo", "double"],
        n_agents=2,
        **default_min,
    ),
    **default_ppo,
)
conf_deep_double_ppo = PPOConf(
    MinimumConf(
        env_name="default_env",
        log_group="deep_double_ppo",
        log_tags=["ppo", "double", "deep"],
        n_agents=2,
        **default_min,
    ),
    **default_ppo
    | dict(
        actor_hidden=[64, 64],
        critic_hidden=[64, 64],
    ),
)

conf_quad_ppo = PPOConf(
    MinimumConf(
        env_name="default_4env",
        log_group="quad_ppo",
        log_tags=["ppo", "quad"],
        n_agents=4,
        **default_min,
    ),
    **default_ppo,
)

conf_large_double_ppo = PPOConf(
    MinimumConf(
        env_name="default_large_env",
        log_group="large_double_ppo",
        log_tags=["ppo", "double", "large"],
        n_agents=2,
        **default_min,
    ),
    **default_ppo,
)

conf_coop_double_ppo = PPOConf(
    MinimumConf(
        env_name="default_coop_env",
        log_group="double_coop_ppo",
        log_tags=["ppo", "double", "coop"],
        n_agents=2,
        **default_min,
    ),
    **default_ppo,
)

conf_large_coop_double_ppo = PPOConf(
    MinimumConf(
        env_name="default_large_coop_env",
        log_group="large_double_coop_ppo",
        log_tags=["ppo", "double", "coop", "large"],
        n_agents=2,
        **default_min,
    ),
    **default_ppo,
)
conf_coop_quad_ppo = PPOConf(
    MinimumConf(
        env_name="default_4coop2_env",
        log_group="quad_coop_ppo",
        log_tags=["ppo", "quad", "coop"],
        n_agents=4,
        **default_min,
    ),
    **default_ppo,
)

conf_coop_local08_quad_ppo = PPOConf(
    MinimumConf(
        env_name="local08_4coop2_env",
        log_group="local08_quad_coop_ppo",
        log_tags=["ppo", "quad", "coop", "local08"],
        n_agents=4,
        **default_min,
    ),
    **default_ppo,
)

conf_large_coop_local08_quad_ppo = PPOConf(
    MinimumConf(
        env_name="local08_large_4coop2_env",
        log_group="local08_large_quad_coop_ppo",
        log_tags=["ppo", "quad", "coop", "local08", "large"],
        n_agents=4,
        **default_min,
    ),
    **default_ppo,
)
