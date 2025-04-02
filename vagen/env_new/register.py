
REGISTERED_ENVS = {}

def register(cls=None, *, name=None):
    """
    A decorator to register environment classes in the REGISTERED_ENVS dictionary.
    
    Args:
        cls: The class to register
        name: Optional custom name for the environment. If not provided, 
              the class name will be used
              
    Usage:
        @register
        class SokobanEnv(BaseEnv):
            pass
            
        @register(name="custom_sokoban")
        class CustomSokobanEnv(BaseEnv):
            pass
    """
    def _register(cls):
        # Use provided name or class name as the registry key
        key = name if name is not None else cls.__name__
        
        # Register the class
        REGISTERED_ENVS[key] = cls
        
        # Return the class unchanged so it can be used normally
        return cls
    
    # Handle case when decorator is used with no arguments: @register
    if cls is not None:
        return _register(cls)
    
    # Handle case when decorator is used with arguments: @register(name="...")
    return _register