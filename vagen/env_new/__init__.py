from .sokoban import SokobanEnv,SokobanConfig
from .frozenlake import FrozenLakeEnv,FrozenLakeConfig
REGISTERED_ENV = {
    "sokoban": {
        "env": SokobanEnv,
        "config": SokobanConfig,
    },
    "frozenlake": {
        "env": FrozenLakeEnv,
        "config": FrozenLakeConfig,
    }
}