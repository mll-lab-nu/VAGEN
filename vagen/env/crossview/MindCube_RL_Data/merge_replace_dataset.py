import json

file_pairs = [
    ('crossviewQA_tinybench_replace_baseRL.jsonl', 'crossviewQA_tinybench_baseRL.jsonl'),
    ('crossviewQA_tinybench_replace_cogmap_and_reasoning.jsonl', 'crossviewQA_tinybench_cogmap_and_reasoning.jsonl'),
    ('crossviewQA_tinybench_replace_cogmap_and_reasoning_plain.jsonl', 'crossviewQA_tinybench_cogmap_and_reasoning_plain.jsonl'),
]

for file_pair in file_pairs:
    file1 = file_pair[0]
    file2 = file_pair[1]
    
    # Load both files
    with open(file1) as f1, open(file2) as f2:
        data1 = {json.loads(line)['id']: line for line in f1}
        data2 = {json.loads(line)['id']: line for line in f2}
    
    # Merge data1 into data2
    data2.update(data1)
    
    # Write merged results back to file2
    with open(file2, 'w') as f:
        f.writelines(data2.values())