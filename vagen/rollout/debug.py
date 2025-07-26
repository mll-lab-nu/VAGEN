from transformers.models.internvl import InternVLProcessor
from transformers.models.got_ocr2 import GotOcr2ImageProcessorFast
from transformers.models.internvl.video_processing_internvl import InternVLVideoProcessor
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os
from verl.utils.tokenizer import hf_tokenizer


# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
# image_processor = GotOcr2ImageProcessorFast(
#     crop_to_patches=False,
#     data_format="channels_first",
#     default_to_square=True,
#     do_center_crop=None,
#     do_convert_rgb=True,
#     do_normalize=True,
#     do_rescale=True,
#     do_resize=True,
#     rescale_factor=0.00392156862745098,
#     size={"height":448, "width": 448},
#     max_patches=12,
#     min_patches=1,
#     resample=3,
#     return_tensors="pt",
#     image_mean=IMAGENET_MEAN,
#     image_std=IMAGENET_STD
# )

# print("Image processor configuration:")
# print(image_processor)
# print()

# tokenizer = hf_tokenizer("OpenGVLab/InternVL3-1B-hf", trust_remote_code=False)
# processor = InternVLProcessor(
#     image_processor=image_processor,
#     image_seq_length=512,
#     tokenizer=tokenizer,
#     chat_template=tokenizer.chat_template,
#     video_processor=InternVLVideoProcessor()
# )
# print(processor)




# image_path = "vagen/rollout/a.jpg"
# if os.path.exists(image_path):
#     image = Image.open(image_path)
#     print(f"Loaded image: {image_path}")
#     print(f"Original image size: {image.size}")
#     print(f"Image mode: {image.mode}")
#     print()
    
#     # Process the image
#     processed_inputs = image_processor(image)
#     # processed_inputs_2 = processor(text=[f"<img><IMG_CONTEXT></img>"], images=[image], return_tensors="pt")
    
#     # print(processed_inputs_2['input_ids'])
#     # print(processed_inputs_2['input_ids'].shape)
#     # print(tokenizer.decode(processed_inputs_2['input_ids'][0]))


#     processed_inputs_2 = processor(text=[f"<IMG_CONTEXT>"], images=[image], return_tensors="pt")
    
#     print(processed_inputs_2.keys())
#     print(processed_inputs_2['input_ids'].shape)
#     # print(tokenizer.decode(processed_inputs_2['input_ids'][0]))




processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")
tokenizer = hf_tokenizer("OpenGVLab/InternVL3-1B-hf", trust_remote_code=False)

image_path = "vagen/rollout/a.jpg"
if os.path.exists(image_path):
    image = Image.open(image_path)
image_list = [image] * 10
text = "<image>" * 10


proxy_text = [f"<IMG_CONTEXT>"] * len(image_list)
model_inputs = processor(text=proxy_text, images=image_list, return_tensors='pt')
input_ids = model_inputs['input_ids'] # |img| * D
for i in range(len(image_list)):
    decoded_tokens = tokenizer.decode(input_ids[i], skip_special_tokens=False)
    text = text.replace('<image>', decoded_tokens, 1)
print(len(tokenizer.encode(text)))





# print(processor.image_processor)
# processed_inputs = processor(text=[f"<IMG_CONTEXT>"], images=[image], return_tensors="pt")
# print(processed_inputs.keys())
# print(processed_inputs['input_ids'].shape)
# print(processor.tokenizer.decode(processed_inputs['input_ids'][0]))