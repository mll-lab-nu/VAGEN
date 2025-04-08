from .sokoban import SokobanEnv,SokobanConfig
from .frozenlake import FrozenLakeEnv,FrozenLakeConfig
from .navigation import NavigationEnv, NavigationConfig
REGISTERED_ENV = {
    "sokoban": {
        "env_cls": SokobanEnv,
        "config_cls": SokobanConfig,
    },
    "frozenlake": {
        "env_cls": FrozenLakeEnv,
        "config_cls": FrozenLakeConfig,
    },
    "navigation": {
        "env_cls": NavigationEnv,
        "config_cls": NavigationConfig
    }
}