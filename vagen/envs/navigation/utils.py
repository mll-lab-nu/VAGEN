"""
Utility functions for the navigation environment.

Bounding box drawing helpers (optional, used for debugging/visualization).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional


def draw_target_box(
    image: Image.Image,
    instance_detections: Dict[str, np.ndarray],
    object_id: str,
    output_path: str,
    color: tuple = (0, 255, 0),
    thickness: int = 1,
) -> None:
    """Draw a bounding box around the target object and save."""
    if object_id not in instance_detections:
        image.save(output_path)
        return

    import cv2

    img_arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    bbox = instance_detections[object_id]
    start = (int(bbox[0]), int(bbox[1]))
    end = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img_arr, start, end, color, thickness)
    img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    Image.fromarray(img_rgb).save(output_path)


VALID_OBJECT_TYPES = [
    "Cart", "Potato", "Faucet", "Ottoman", "CoffeeMachine", "Candle", "CD",
    "Pan", "Watch", "HandTowel", "SprayBottle", "BaseballBat", "CellPhone",
    "Kettle", "Mug", "StoveBurner", "Bowl", "Spoon", "TissueBox", "Apple",
    "TennisRacket", "SoapBar", "Cloth", "Plunger", "FloorLamp",
    "ToiletPaperHanger", "Spatula", "Plate", "Glassbottle", "Knife", "Tomato",
    "ButterKnife", "Dresser", "Microwave", "GarbageCan", "WateringCan",
    "Vase", "ArmChair", "Safe", "KeyChain", "Pot", "Pen", "Newspaper",
    "Bread", "Book", "Lettuce", "CreditCard", "AlarmClock", "ToiletPaper",
    "SideTable", "Fork", "Box", "Egg", "DeskLamp", "Ladle", "WineBottle",
    "Pencil", "Laptop", "RemoteControl", "BasketBall", "DishSponge", "Cup",
    "SaltShaker", "PepperShaker", "Pillow", "Bathtub", "SoapBottle", "Statue",
    "Fridge", "Toaster", "LaundryHamper",
]


def draw_boxes(
    image: Image.Image,
    classes_and_boxes: Dict[str, np.ndarray],
    image_path: str,
) -> None:
    """Draw bounding boxes around multiple objects and save."""
    draw = ImageDraw.Draw(image)
    for class_name, box in classes_and_boxes.items():
        if class_name.split("|")[0] in VALID_OBJECT_TYPES:
            color = tuple(np.random.choice(range(256), size=3))
            draw.rectangle(
                [box[0], box[1], box[2], box[3]], outline=color, width=1
            )
    image.save(image_path)
