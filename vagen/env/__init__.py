from .sokoban import SokobanEnv,SokobanConfig
from .frozenlake import FrozenLakeEnv,FrozenLakeConfig
REGISTERED_ENV = {
    "sokoban": {
        "env_cls": SokobanEnv,
        "config_cls": SokobanConfig,
    },
    "frozenlake": {
        "env_cls": FrozenLakeEnv,
        "config_cls": FrozenLakeConfig,
    }
}