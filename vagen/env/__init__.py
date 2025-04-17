from .sokoban import SokobanEnv,SokobanEnvConfig
from .frozenlake import FrozenLakeEnv,FrozenLakeEnvConfig, FrozenLakeService
from .navigation import NavigationEnv, NavigationEnvConfig, NavigationServiceConfig, NavigationService
from .svg import SVGEnv, SvgEnvConfig, SVGService

REGISTERED_ENV = {
    "sokoban": {
        "env_cls": SokobanEnv,
        "config_cls": SokobanEnvConfig,
    },
    "frozenlake": {
        "env_cls": FrozenLakeEnv,
        "config_cls": FrozenLakeEnvConfig,
        "service_cls": FrozenLakeService
    },
    "navigation": {
        "env_cls": NavigationEnv,
        "config_cls": NavigationEnvConfig,
        "service_cls": NavigationService,
        "service_config_cls": NavigationServiceConfig
    },
    "svg": {
        "env_cls": SVGEnv,
        "config_cls": SvgEnvConfig,
        "service_cls": SVGService
    },
}