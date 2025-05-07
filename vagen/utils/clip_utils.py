from typing import Dict, List, Tuple, Optional, Any, Union
import torch

def is_clip_needed(step_batch_result:Dict[Any, Tuple[Dict, float, bool, Dict]]):
    for env_id, (_,_,_,info) in step_batch_result.items():
        if "clip_reward" in info:
            return True
    return False

def get_clip_score_batch(model, preprocess, device, step_batch_result: Dict[Any, Tuple[Dict, float, bool, Dict]]):
    ids = []
    images = []
    texts = []
    
    # Collect all image-text pairs
    for env_id, (_, _, _, info) in step_batch_result.items():
        if "grounding_clip" in info and info["grounding_clip"] and info["observation_content"]:
            texts.append(info["observation_content"])
            images.append(info["observation_image"])
            ids.append((env_id, "grounding_clip"))
            info.pop("observation_image")
        
        if "worldmodeling_clip" in info and info["worldmodeling_clip"] and info["prediction_content"]:
            texts.append(info["prediction_content"])
            images.append(info["prediction_image"])
            ids.append((env_id, "worldmodeling_clip"))
            info.pop("prediction_image")
    
    # If no CLIP evaluations needed, return the original batch
    if not ids:
        return step_batch_result
    
    # Create a modifiable copy of the batch result
    new_step_batch_result = {}
    for env_id, step_data in step_batch_result.items():
        new_step_batch_result[env_id] = list(step_data)
    
    # Process images and texts with CLIP
    image_inputs = torch.cat([preprocess(image).unsqueeze(0).to(device) for image in images])
    text_inputs = clip.tokenize(texts).to(device)
    
    with torch.no_grad():
        # Encode images and texts
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarities = (text_features@image_features.T).cpu().numpy()
    print("similarities", similarities)
    
    
    
    # Update rewards and metrics
    for i, (env_id, clip_type) in enumerate(ids):
        similarity_score = similarities[i][i]
        info = new_step_batch_result[env_id][3]
        
        if "metrics" not in info:
            info["metrics"] = {}
        if "turn_metrics" not in info["metrics"]:
            info["metrics"]["turn_metrics"] = {}
            
        info["metrics"]["turn_metrics"][clip_type] = similarity_score
        
        # Add similarity score to reward, weighted by clip weight
        if clip_type + "_weight" in info:
            weight = info[clip_type + "_weight"]
            new_step_batch_result[env_id][1] += similarity_score * weight
    
    # Convert back to tuples
    for env_id, step_data in new_step_batch_result.items():
        obs, reward, done, info = step_data
        step_batch_result[env_id] = (obs, reward, done, info)
        
    return step_batch_result

if __name__ == "__main__":
    # test the function
    import clip
    from PIL import Image
    image_path_list=["/home/kangrui/projects/vagen/cat.jpeg","/home/kangrui/projects/vagen/sokoban_1.png"]
    step_batch_result={}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50x64", device=device)
    for idx,path in enumerate(image_path_list):
        env_id=idx
        obs={}
        reward=0
        info={}
        info["grounding_clip"]=True
        info["worldmodeling_clip"]=True
        info["observation_content"]="the player is above the box, and the box is above the goal"
        info["prediction_content"]="the player is below the box, and the box is below the goal"
        info["observation_image"]=Image.open(path)
        info["prediction_image"]=Image.open(path)
        info["grounding_clip_weight"]=1.0
        info["worldmodeling_clip_weight"]=1.0
        step_batch_result[env_id]=(obs,reward,False,info)
    step_batch_result=get_clip_score_batch(model,preprocess,device,step_batch_result)
    print(step_batch_result)