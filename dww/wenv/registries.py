"""
Module dedicated to managing wenv configurations.

Intended api:
1. Get a reference to the global wenv registry
2. Use the registry to retrieve an allocate a WaterWorld wenv

>>> env_registry = WaterWorldRegistry.getinstance()
>>> env_registry.update({'default_env': WaterWorldConfig()})
>>> my_env, my_conf, my_sha = env_registry('default_env')

Now my_env is the usual multiagent wenv wrapper from pettingzoo:
- `.agent_iter()`
- `.reset()`
- `.render()`
- `.last()`
- `.step(action)`
- `.close()`


"""
from collections import OrderedDict
from typing import Dict, Optional, Tuple, MutableMapping

from pettingzoo.sisl import waterworld_v4
from pettingzoo import AECEnv

from dww.wenv.config import WaterWorldConfig, TRenderMode

TEnvID = str
TSha = str
TConfigDict = Dict[TEnvID, WaterWorldConfig]
TBySha = Dict[TSha, TEnvID]


class WaterWorldRegistry(MutableMapping):
    """A custom Dictionary str -> WaterWorldConfig.
    provides 2 convenience methods:
    - .put to register multiple wenv configs at once
    - .getenv to allocate an environment by name

    You can use this class in a Singleton-Patter style with .getinstance
    But you can also instantiate regularly to escape the pattern if needed.

    The registry also maintains an inverse index from config sha256 to config name.
    This is useful if you want to track which config corresponded to that sha256 after the fact

    This implements the MutableMapping ABC, so we get stuff like .clear and .update for free.

    Intended api:
    >>> w = WaterWorldRegistry()                                                        # create a registry
    >>> w.update({'foo': WaterWorldConfig(), 'bar': WaterWorldConfig(n_pursuers=5)})    # add configs to registry
    >>> len(w)                                                                          # check len
    2
    >>> wenv, conf, sign = w('foo')                                                      # get wenv by name
    >>> 'foo' == w.get_name_by_sha(sign) and sign == conf.env_signature()               # check everything matches
    True
    >>> w.clear()                                                                       # delete all confs
    >>> len(w)                                                                          # check len
    0
    """

    INSTANCE = None

    _by_name: TConfigDict
    _by_sha: TBySha

    def __init__(self):
        self._by_name = OrderedDict()
        self._by_sha = OrderedDict()

    def __getitem__(self, name: TEnvID):
        return self._by_name[name]

    def __setitem__(self, key, value: WaterWorldConfig) -> WaterWorldConfig:
        self._by_name[key] = value
        self._by_sha[value.env_signature()] = key
        return value

    def __delitem__(self, key):
        v = self[key]
        del self._by_name[key]
        del self._by_sha[v.env_signature()]

    def __len__(self):
        return len(self._by_name)

    def __iter__(self):
        return iter(self._by_name)

    @classmethod
    def getinstance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = cls()
        return cls.INSTANCE

    # configs: Dict[str, WaterWorldConfig] = field(default_factory=dict, init=False)

    def __call__(self, name, render_mode: TRenderMode = None, FPS: Optional[int] = None) \
            -> Tuple[AECEnv, WaterWorldConfig, TSha]:
        """Allocate an actual pettingzoo.sisl.waterworld_v4 wenv.
        You get back the the .wenv(**kwargs) object, the used config, the sha256 of relevant params

        :param name: The name of the environment config
        :param render_mode: Optionally override render mode while allocating
        :param FPS: Optionally override FPS while allocating
        :return: (wenv, WaterWorldConfig, sha256)
        """
        return self._getenv(name, render_mode, FPS)

    def _getenv(self, name, render_mode: TRenderMode = None, FPS: Optional[int] = None):
        _conf = self[name]  # get the config dataclass associate with this name
        sign = _conf.env_signature()  # compute configuration hash
        kwargs = _conf.to_env_kwargs()  # get the kwargs to pass to the underlying pettingzoo wenv

        # optionally override the non-behavior-relevant  arguments
        if render_mode is not None:
            kwargs['render_mode'] = render_mode
        if FPS is not None:
            kwargs['FPS'] = FPS

        _env = waterworld_v4.env(**kwargs)  # allocate the actual pettingzoo wenv

        setattr(_env, "__drlmeta__", {'conf': _conf, 'sign': sign})  # set metadata
        # return environment, config and sha-signature
        return _env, _conf, sign

    def get_name_by_sha(self, sha: TSha) -> TEnvID:
        return self._by_sha[sha]
