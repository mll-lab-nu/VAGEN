import json

files = [
    'crossviewQA_tinybench_baseRL.jsonl',
    'crossviewQA_tinybench_cogmap_and_reasoning.jsonl', 
    'crossviewQA_train_baseRL.jsonl',
    'crossviewQA_train_cogmap_and_reasoning.jsonl',
    'crossviewQA_tinybench_cogmap_and_reasoning_plain.jsonl',
    'crossviewQA_train_cogmap_and_reasoning_plain.jsonl', 
]

for file in files:
    # Read all lines first
    with open(file, 'r') as f:
        lines = f.readlines()
    
    # Process and write back
    with open(file, 'w') as f:
        for line in lines:
            data = json.loads(line)
            # Get number of images
            num_images = len(data['images'])
            
            # Add <image>\n prefix for each image
            image_prefix = '<image>\n' * num_images
            
            # Add prefix to question_str
            data['question_str'] = image_prefix + data['question_str']
            
            # Write each processed line
            f.write(json.dumps(data) + '\n')