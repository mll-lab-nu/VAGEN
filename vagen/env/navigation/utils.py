import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional

def draw_target_box(
    image,
    instance_detections: Dict[str, np.ndarray],
    object_id: str,
    output_path: str,
    color: tuple = (0, 255, 0),  # Default color: green
    thickness = 1
):
    """Draw a bounding box around the target object.
    
    Args:
        image: The image to draw on
        instance_detections: Dictionary of object IDs to bounding boxes
        object_id: ID of the target object
        output_path: Path to save the output image
        color: Color of the bounding box (RGB)
        thickness: Thickness of the bounding box lines
    """
    if object_id in instance_detections:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if image is None:
            raise ValueError(f"Could not read image")
        
        # Get coordinates for the specified object ID
        bbox = instance_detections[object_id]
        start_point = (int(bbox[0]), int(bbox[1]))  # Upper left corner
        end_point = (int(bbox[2]), int(bbox[3]))    # Lower right corner
        
        # Draw the rectangle
        cv2.rectangle(
            image,
            start_point,
            end_point,
            color,
            thickness
        )
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(image_rgb)
        output_image.save(output_path)
    else:
        # Save the original image if the object is not found
        image.save(output_path)

def draw_boxes(image, classes_and_boxes, image_path):
    """Draw bounding boxes around multiple objects.
    
    Args:
        image: The image to draw on
        classes_and_boxes: Dictionary mapping class names to bounding boxes
        image_path: Path to save the output image
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    font.size = 8
    
    valid_objs = ['Cart', 'Potato', 'Faucet', 'Ottoman', 'CoffeeMachine', 'Candle', 'CD', 'Pan', 'Watch',
                'HandTowel', 'SprayBottle', 'BaseballBat', 'CellPhone', 'Kettle', 'Mug', 'StoveBurner', 'Bowl', 
                'Spoon', 'TissueBox', 'Apple', 'TennisRacket', 'SoapBar', 'Cloth', 'Plunger', 'FloorLamp', 
                'ToiletPaperHanger', 'Spatula', 'Plate', 'Glassbottle', 'Knife', 'Tomato', 'ButterKnife', 
                'Dresser', 'Microwave', 'GarbageCan', 'WateringCan', 'Vase', 'ArmChair', 'Safe', 'KeyChain', 
                'Pot', 'Pen', 'Newspaper', 'Bread', 'Book', 'Lettuce', 'CreditCard', 'AlarmClock', 'ToiletPaper', 
                'SideTable', 'Fork', 'Box', 'Egg', 'DeskLamp', 'Ladle', 'WineBottle', 'Pencil', 'Laptop', 
                'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'SaltShaker', 'PepperShaker', 'Pillow', 
                'Bathtub', 'SoapBottle', 'Statue', 'Fridge', 'Toaster', 'LaundryHamper']
    
    # Loop through each class and its associated bounding boxes
    for class_name, box in classes_and_boxes.items():
        if class_name.split('|')[0] in valid_objs:
            color = tuple(np.random.choice(range(256), size=3))
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
            
    image.save(image_path)