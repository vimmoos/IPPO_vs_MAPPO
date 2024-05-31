import inspect
from collections import OrderedDict
from dataclasses import dataclass, field, asdict, fields, MISSING
from typing import Union, Literal, ClassVar, List, Tuple

import dict_hash
from pettingzoo.sisl.waterworld.waterworld_base import WaterworldBase


class ErrorEnvParamsCompliance(Exception):
    def __init__(self, msg, what):
        super().__init__(msg)
        self.what = what


def suggest_defaults():
    """print the guessed dataclass defaults from the wenv __init__ signature"""
    for k, v in inspect.signature(WaterworldBase.__init__).parameters.items():
        if k != "self":
            print(f"{k}: {type(v.default).__name__} = {v.default}")


TRenderMode = Union[None, Literal["human"], Literal["rgb_array"]]


@dataclass(frozen=True)
class WaterWorldConfig:
    """Configuration class for the environment properties.
    This has attributes that map directly to the wenv constructor, just typed, with
    some convenience methods.

    NOTE: for the purpose of our project we will consider only a few parameters
    as actually manipulable. All the remaining ones are considered to be fixed
    in order to limit the variations and concentrate on the study of those parameters.

    We still put these in the dataclass and disallow to provide them in the constructor,
    if you want to changes those you must forcefully `setattr` on them, as this is Frozen dataclass.
    If you do forcefully change those attributes then `.raise_compliant()` will raise an error.
    """

    # config params that are allowed to be changed for experiment purposes
    MANIPULABLE: ClassVar[str] = ["n_pursuers", "n_coop", "local_ratio", "sensor_range"]

    # non-behvior relevant parameters
    NON_RELEVANT: ClassVar[str] = ["render_mode", "FPS"]

    # manipulable
    n_pursuers: int = 2  # the number of learning agents
    n_coop: int = (
        1  # the number of agents that must simult. collide with food to eat it
    )
    local_ratio: float = 1.0  # controls multi-agent credit assignment
    sensor_range: float = 0.2  # controls how much of the wenv is actually observable

    # non-relevant to simulation behavior
    render_mode: TRenderMode = None
    FPS: int = 15

    # fixed
    n_evaders: int = field(default=5, init=False)
    n_poisons: int = field(default=10, init=False)
    n_obstacles: int = field(default=1, init=False)
    n_sensors: int = field(default=30, init=False)
    radius: float = field(default=0.015, init=False)
    obstacle_radius: float = field(default=0.1, init=False)
    obstacle_coord: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.5, 0.5)], init=False
    )
    pursuer_max_accel: float = field(default=0.5, init=False)
    pursuer_speed: float = field(default=0.2, init=False)
    evader_speed: float = field(default=0.1, init=False)
    poison_speed: float = field(default=0.1, init=False)
    poison_reward: float = field(default=-1.0, init=False)
    food_reward: float = field(default=10.0, init=False)
    encounter_reward: float = field(default=0.01, init=False)
    thrust_penalty: float = field(default=-0.5, init=False)
    speed_features: bool = field(default=True, init=False)
    max_cycles: int = field(default=500, init=False)

    def to_env_kwargs(self):
        """Return a **-able dict to use for actually building the wenv"""
        return asdict(self, dict_factory=OrderedDict)  # type: ignore

    def env_hash_relevant(self):
        """Return a dictionary with only the behavior relevant parameters of this config.
        I.e. exclude FPS, render_mode"""
        d = self.to_env_kwargs()
        for name in self.NON_RELEVANT:
            del d[name]
        return d

    def env_signature(self):
        """Get consistent sha256 for the config (relevant params only)"""
        return dict_hash.sha256(self.env_hash_relevant())

    def manipulable_values(self):
        d = self.to_env_kwargs()
        for fld in fields(self):
            if fld.name not in self.MANIPULABLE:
                del d[fld.name]
        return d


def raise_compliant(ww_config: WaterWorldConfig):
    """
    check that values for non-manipulable attributes are left at default for every declared field.
    This should be called before official exps as sanity check.
    """
    for fld in fields(ww_config):
        # if the filed is not MANIPULABLE and not NON_RELEVANT
        if (
            fld.name not in ww_config.MANIPULABLE
            and fld.name not in ww_config.NON_RELEVANT
        ):
            # if the field has a default value, then the curr. val. must be equal to that
            if fld.default is not MISSING:
                if fld.default != getattr(ww_config, fld.name):
                    raise ErrorEnvParamsCompliance(
                        f"NON-MANIPULABLE attribute `{fld.name}` was changed! (default)",
                        {
                            "fld": fld,
                            "expected": fld.default,
                            "value": getattr(ww_config, fld.name),
                            "kind": "default",
                        },
                    )
                continue
            # if the field has a default factory, then the curr. val must be equal to its result
            elif fld.default_factory is not MISSING:
                if fld.default_factory() != getattr(ww_config, fld.name):
                    raise ErrorEnvParamsCompliance(
                        f"NON-MANIPULABLE attribute `{fld.name}` was changed! (default-factory)",
                        {
                            "fld": fld,
                            "expected": fld.default_factory(),
                            "value": getattr(ww_config, fld.name),
                            "kind": "default_factory",
                        },
                    )
                continue
            else:
                raise ErrorEnvParamsCompliance(
                    f"There is field `{fld.name}` that is not in MANIPULABLE nor NON_RELEVANT, which does "
                    "Not have neither default nor default_factory. This should not happen, did you forget to "
                    "Register that as MANIPULABLE or NON_RELVANT, or to give it a default?",
                    {
                        "fld": fld,
                        "value": getattr(ww_config, fld.name),
                        "expected": None,
                        "kind": "missing_default",
                    },
                )
