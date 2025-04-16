from .sokoban import SokobanEnv,SokobanConfig
from .frozenlake import FrozenLakeEnv,FrozenLakeConfig, FrozenLakeService
# from .navigation import NavigationEnv, NavigationConfig
# from .svg import SVGEnv, SVGConfig, SVGService

REGISTERED_ENV = {
    "sokoban": {
        "env_cls": SokobanEnv,
        "config_cls": SokobanConfig,
    },
    "frozenlake": {
        "env_cls": FrozenLakeEnv,
        "config_cls": FrozenLakeConfig,
        "service_cls": FrozenLakeService
    },
    # "navigation": {
    #     "env_cls": NavigationEnv,
    #     "config_cls": NavigationConfig
    # },
    # "svg": {
    #     "env_cls": SVGEnv,
    #     "config_cls": SVGConfig,
    #     "service_cls": SVGService
    # },
}