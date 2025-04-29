from .sokoban import SokobanEnv,SokobanEnvConfig
from .frozenlake import FrozenLakeEnv,FrozenLakeEnvConfig, FrozenLakeService
# from .navigation import NavigationEnv, NavigationEnvConfig, NavigationServiceConfig, NavigationService
# from .svg import SVGEnv, SvgEnvConfig, SVGService, SVGServiceConfig
# from .primitive_skill import PrimitiveSkillEnv, PrimitiveSkillEnvConfig, PrimitiveSkillService, PrimitiveSkillServiceConfig
from .alfworld import ALFWorldEnv, ALFWorldEnvConfig, ALFWorldService, ALFWorldServiceConfig
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
    # "navigation": {
    #     "env_cls": NavigationEnv,
    #     "config_cls": NavigationEnvConfig,
    #     "service_cls": NavigationService,
    #     "service_config_cls": NavigationServiceConfig
    # },
    # "svg": {
    #     "env_cls": SVGEnv,
    #     "config_cls": SvgEnvConfig,
    #     "service_cls": SVGService,
    #     "service_config_cls": SVGServiceConfig
    # },
    # "primitive_skill": {
    #     "env_cls": PrimitiveSkillEnv,
    #     "config_cls": PrimitiveSkillEnvConfig,
    #     "service_cls": PrimitiveSkillService,
    #     "service_config_cls": PrimitiveSkillServiceConfig
    # },
    "alfworld": {
        "env_cls": ALFWorldEnv,
        "config_cls": ALFWorldEnvConfig,
        "service_cls": ALFWorldService,
        "service_config_cls": ALFWorldServiceConfig
    },
}