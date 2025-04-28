import io
import base64
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union

# -------------- serialize and deserialize observation --------------

def serialize_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize an observation dictionary that might contain non-serializable objects.
    
    Args:
        observation: Observation dictionary from environment
        
    Returns:
        Serialized observation
    """
    serialized_obs = observation.copy()
    
    # Handle multi_modal_data if present
    if "multi_modal_data" in serialized_obs:
        serialized_multi_modal = {}
        for key, values in serialized_obs["multi_modal_data"].items():
            serialized_values = []
            for value in values:
                # Check if it's a PIL Image
                if hasattr(value, "mode") and hasattr(value, "save"):
                    serialized_values.append(serialize_pil_image(value))
                # Add more type checks as needed
                else:
                    serialized_values.append(value)
            serialized_multi_modal[key] = serialized_values
        serialized_obs["multi_modal_data"] = serialized_multi_modal
    
    return serialized_obs

def deserialize_observation(serialized_obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize an observation dictionary.
    
    Args:
        serialized_obs: Serialized observation
        
    Returns:
        Deserialized observation
    """
    deserialized_obs = serialized_obs.copy()
    
    # Handle multi_modal_data if present
    if "multi_modal_data" in deserialized_obs:
        deserialized_multi_modal = {}
        for key, values in deserialized_obs["multi_modal_data"].items():
            deserialized_values = []
            for value in values:
                if isinstance(value, dict):
                    if "__pil_image__" in value:
                        deserialized_values.append(deserialize_pil_image(value))
                    elif "__numpy_array__" in value:
                        deserialized_values.append(deserialize_numpy_array(value))
                    else:
                        deserialized_values.append(value)
                else:
                    deserialized_values.append(value)
            deserialized_multi_modal[key] = deserialized_values
        deserialized_obs["multi_modal_data"] = deserialized_multi_modal
    
    return deserialized_obs

# -------------- serialize and deserialize step (obs, reward, done, info) --------------
def serialize_step_result(result_tuple: Tuple[Dict, float, bool, Dict]) -> Tuple[Dict, float, bool, Dict]:
    """Serialize step result tuple handling NumPy types and custom objects."""
    observation, reward, done, info = result_tuple
    
    serialized_observation = serialize_observation(observation)
    
    # Handle reward (might be a NumPy float)
    if hasattr(reward, 'item'):
        serialized_reward = float(reward)
    else:
        serialized_reward = reward
        
    # Handle done flag (might be a NumPy boolean)
    if isinstance(done, (list, tuple, np.ndarray)):
        done = done[0]
    if hasattr(done, 'item'):
        serialized_done = bool(done)
    else:
        serialized_done = done
        
    # Handle info dictionary
    serialized_info = serialize_info(info)
    
    return (serialized_observation, serialized_reward, serialized_done, serialized_info)

def deserialize_step_result(serialized_result: Tuple[Dict, float, bool, Dict]) -> Tuple[Dict, float, bool, Dict]:
    """
    Deserialize the step result tuple.
    
    Args:
        serialized_result: The serialized tuple (observation, reward, done, info).
        
    Returns:
        The deserialized tuple.
    """
    serialized_observation, reward, done, serialized_info = serialized_result
    
    # Process the observation using the existing deserialize_observation function.
    observation = deserialize_observation(serialized_observation)
    
    # The info dictionary might contain objects that require special handling.
    info = deserialize_dict(serialized_info)
    
    return (observation, reward, done, info)

# -------------- utils for previous functions --------------

def serialize_pil_image(img) -> str:
    """
    Serialize a PIL Image to a base64 string.
    
    Args:
        img: PIL Image object
        
    Returns:
        Base64 encoded string of the image
    """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {"__pil_image__": img_str}

def deserialize_pil_image(serialized_data: Dict[str, str]):
    """
    Deserialize a base64 string back to a PIL Image.
    
    Args:
        serialized_data: Dictionary with "__pil_image__" key containing base64 string
        
    Returns:
        PIL Image object
    """
    from PIL import Image
    img_data = base64.b64decode(serialized_data["__pil_image__"])
    return Image.open(io.BytesIO(img_data))

def serialize_numpy_array(arr) -> Dict[str, Any]:
    """
    Serialize a numpy array to a serializable format.
    
    Args:
        arr: Numpy array
        
    Returns:
        Dictionary with serialized array data
    """
    return {
        "__numpy_array__": {
            "data": arr.tolist(),
            "dtype": str(arr.dtype),
            "shape": arr.shape
        }
    }

def deserialize_numpy_array(serialized_data: Dict[str, Any]):
    """
    Deserialize data back to a numpy array.
    
    Args:
        serialized_data: Dictionary with "__numpy_array__" key
        
    Returns:
        Numpy array
    """
    array_data = serialized_data["__numpy_array__"]
    return np.array(array_data["data"], dtype=np.dtype(array_data["dtype"])).reshape(array_data["shape"])

def serialize_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize info dictionary that might contain various types including Proposition objects."""
    return serialize_dict(info)

def serialize_dict(obj: Any) -> Any:
    """Recursively serialize objects that may contain NumPy types or custom objects."""
    if isinstance(obj, dict):
        return {k: serialize_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(serialize_dict(x) for x in obj)
    elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):  # Detect NumPy arrays and scalars
        if obj.ndim == 0:  # Scalar
            if np.issubdtype(obj.dtype, np.floating):
                return float(obj)
            elif np.issubdtype(obj.dtype, np.integer):
                return int(obj)
            elif np.issubdtype(obj.dtype, np.bool_):
                return bool(obj)
            else:
                return obj.item()  # Generic conversion
        else:  # Array
            return serialize_dict(obj.tolist())
    # Handle Proposition objects
    elif obj.__class__.__name__ == 'Proposition':
        return str(obj)
    # Handle other possible custom types
    elif hasattr(obj, '__dict__'):
        try:
            return str(obj)
        except Exception as e:
            return f"serialize error: <{obj.__class__.__name__} object> cannot be serilized ({e})"
    else:
        return obj

def deserialize_dict(obj: Any) -> Any:
    """
    Recursively deserialize any special objects in the dictionary.
    
    Args:
        obj: The object to deserialize.
        
    Returns:
        The deserialized object.
    """
    if isinstance(obj, dict):
        # Check for special serialization markers.
        if "__pil_image__" in obj:
            # Process PIL images using the existing function.
            return deserialize_pil_image(obj)
        elif "__numpy_array__" in obj:
            # Process NumPy arrays using the existing function.
            return deserialize_numpy_array(obj)
        else:
            return {k: deserialize_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(deserialize_dict(x) for x in obj)
    else:
        return obj
