from .config import WaterWorldConfig
from .registries import WaterWorldRegistry

# get the global registry of environment and populate it with defaults
WaterWorldRegistry.getinstance().update(
    [

        # ---- Multiagent configs with NO COOP ----
        # 2 Learning agents no coop necessary
        ("default_env", WaterWorldConfig(n_pursuers=3, n_coop=1)),
        # 2 learning agents no coop necessary, large sensor range
        (
            "default_large_env",
            WaterWorldConfig(n_pursuers=3, n_coop=1, sensor_range=0.5),
        ),

        # 4 Learning agents no coop necessary
        ("default_4env", WaterWorldConfig(n_pursuers=5, n_coop=1)),

        # ---- Multiagent configs WITH COOP 2 ----
        # 2 Learning agents coop necessary
        # - base
        ("default_coop_env", WaterWorldConfig(n_pursuers=3, n_coop=2)),
        # - large sensor range
        (
            "default_large_coop_env",
            WaterWorldConfig(n_pursuers=3, n_coop=2, sensor_range=0.5),
        ),

        # 4 learning agents 2-coop necessary
        # - base
        ("default_4coop2_env", WaterWorldConfig(n_pursuers=5, n_coop=2)),
        # - with local ratio reward assignment
        ("local08_4coop2_env", WaterWorldConfig(n_pursuers=5, n_coop=2, local_ratio=0.8)),
        # - with local ratio and large sensors
        ("local08_large_4coop2_env", WaterWorldConfig(n_pursuers=5, n_coop=2, local_ratio=0.8, sensor_range=0.5)),

        # --- Single agent configs for baseline ---
        ("default_single", WaterWorldConfig(n_pursuers=1, n_coop=1)),
        (
            "default_large_single",
            WaterWorldConfig(n_pursuers=1, n_coop=1, sensor_range=0.5),
        ),
    ]
)
