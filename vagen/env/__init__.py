from .sokoban import SokobanEnv,SokobanConfig
from .frozenlake import FrozenLakeEnv,FrozenLakeConfig
from .navigation import NavigationEnv, NavigationConfig
from .svg import SVGDINOEnv, SVGDINOConfig
from .svg import SVGEnv, SVGConfig

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
    },
    "svg": {
        "env_cls": SVGEnv,
        "config_cls": SVGConfig
    },
    "svgdino": {
        "env_cls": SVGDINOEnv,
        "config_cls": SVGDINOConfig
    }
}