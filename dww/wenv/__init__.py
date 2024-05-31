from .registries import WaterWorldRegistry
# this makes sure default envs are loaded when importing this packages
# this will actually also create the main registry if nobody created before
from . import default_envs

# we export the main registry reference under convenience name
registry = WaterWorldRegistry.getinstance()

__all__ = ['WaterWorldRegistry', 'registry']
