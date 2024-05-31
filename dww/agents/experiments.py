from dww.agents import default_conf as c


confs = [
    c.conf_large_double_ppo,
    c.conf_large_single_ppo,
    c.conf_large_coop_double_ppo,
    c.conf_deep_double_ppo,
]
confs2 = [
    c.conf_coop_quad_ppo,
    c.conf_quad_ppo,
    c.conf_double_ppo,
    c.conf_coop_double_ppo,
]

conf_final = [
    c.conf_coop_double_ppo,
    c.conf_large_coop_double_ppo,
    c.conf_coop_local08_quad_ppo,
    c.conf_large_coop_local08_quad_ppo,
    c.conf_coop_quad_ppo,
    c.conf_double_ppo,
    c.conf_large_double_ppo,
    c.conf_quad_ppo,
]

conf_test = [c.conf_coop_double_ppo]
