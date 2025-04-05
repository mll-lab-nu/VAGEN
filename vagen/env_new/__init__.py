from .sokoban import SokobanEnv,SokobanConfig
REGISTERED_ENV = {
    "sokoban": {
        "env": SokobanEnv,
        "config": SokobanConfig,
    }
}