import io
import base64
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union

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