"""
The script is used to align the cogmap and reasoning RL prompts with the one used in SFT.
"""

"""SFT Prompt:
<image>
<image>
[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints. You will then create a detailed cognitive map representing the scene using a 10x10 grid coordinate system.

[Rules]
1. Focus ONLY on these categories of objects in the scene: {blue ball, wall, open space, sliding door, grey chairs}
2. Create a cognitive map with the following structure in the bird's view:
   - A 10x10 grid where [0,0] is at the top-left corner and [9,9] is at the bottom-right corner
   - up = towards the top of the grid (decreasing y)
   - right = towards the right of the grid (increasing x)
   - down = towards the bottom of the grid (increasing y)
   - left = towards the left of the grid (decreasing x)
   - inner = straight into the 2D map (perpendicular to the grid, pointing away from you)
   - outer = straight out of the 2D map (perpendicular to the grid, pointing towards you)
   - Include positions of all objects from the specified categories
   - Estimate the center location (coordinates [x, y]) of each instance within provided categories
   - If a category contains multiple instances, include all of them
   - Each object's estimated location should accurately reflect its real position in the scene, preserving the relative spatial relationships among all objects
   - Combine and merge information from the images since they are pointing to the same scene, calibrating the object locations accordingly
   - Include camera positions and directions for each view
3. Carefully integrate information from all views to create a single coherent spatial representation.


[Output]
1. Given the provided views and main objects mentioned in the above rules, you **MUST** present your cognitive map in the following JSON format **before your reasoning**:
{
  "objects": [
    {"name": "object_name", "position": [x, y], "facing": "direction"},
    {"name": "object_without_orientation", "position": [x, y]}
  ],
  "views": [
    {"name": "View/Image 1", "position": [x, y], "facing": "direction"},
    {"name": "View/Image 2", "position": [x, y], "facing": "direction"}
  ]
}

2. Next, please also provide your reasons step by step in details, then provide *ONE* correct answer selecting from the options. Your response's format should be like "<CogMap>
 <Your cognitive map>
<Reasoning>
 ...
<Answer>
 Therefore, my answer is <selected option>". Your <selected option> must be in the format like "A. Above". Your option must be from the available options.
[Question]
Based on these two views showing the same scene, which direction did you move from the first view to the second view? A. Forward-left  B. Forward-right.
"""

"""RL Prompt:
<image>
<image>
<image>
<image>
[Task]
Your task is to analyze the spatial arrangement of objects in the scene by examining the provided images, which show the scene from different viewpoints. You will then create a detailed cognitive map representing the scene using a 10x10 grid coordinate system.

[Rules]
1. Focus ONLY on these categories of objects in the scene: {incense burner, light brown wall, gate, stone fountain, decorated wall}
2. Create a cognitive map with the following structure in the bird's view:
   - A 10x10 grid where [0,0] is at the top-left corner and [9,9] is at the bottom-right corner
   - up = towards the top of the grid (decreasing y)
   - right = towards the right of the grid (increasing x)
   - down = towards the bottom of the grid (increasing y)
   - left = towards the left of the grid (decreasing x)
   - inner = straight into the 2D map (perpendicular to the grid, pointing away from you)
   - outer = straight out of the 2D map (perpendicular to the grid, pointing towards you)
   - Include positions of all objects from the specified categories
   - Estimate the center location (coordinates [x, y]) of each instance within provided categories
   - If a category contains multiple instances, include all of them
   - Each object's estimated location should accurately reflect its real position in the scene, preserving the relative spatial relationships among all objects
   - Combine and merge information from the images since they are pointing to the same scene, calibrating the object locations accordingly
   - Include camera positions and directions for each view
3. Carefully integrate information from all views to create a single coherent spatial representation.


[Output]
1. Given the provided views and main objects mentioned in the above rules, you **MUST** present your cognitive map in the following JSON format **before your reasoning**:
{
  "objects": [
    {"name": "object_name", "position": [x, y], "facing": "direction"},
    {"name": "object_without_orientation", "position": [x, y]}
  ],
  "views": [
    {"name": "View/Image 1", "position": [x, y], "facing": "direction"},
    {"name": "View/Image 2", "position": [x, y], "facing": "direction"}
  ]
}

2. Next, based on your generated cognitive map, please generate the answer to the question. For example, if you think the correct answer is 'A. Above' from ' A. Above B. Under C. Front D. Behind', you must output 'A. Above'. Your answer format should be like "<think><Your cognitive map> your following reasoning</think><answer><Your answer></answer>"
[Question]
Based on these four images (image 1, 2, 3, and 4) showing the incense burner from different viewpoints (front, left, back, and right), with each camera aligned with room walls and partially capturing the surroundings: If you are standing at the viewpoint presented in image 3, then you turn right and move forward, will you get closer to the gate? A. Yes B. No
"""

import orjsonl

input_file = "crossviewQA_tinybench_cogmap_and_reasoning_bak.jsonl"
output_file = "crossviewQA_tinybench_cogmap_and_reasoning.jsonl"

replaced_str = """\
2. Next, based on your generated cognitive map, please generate the answer to the question. For example, if you think the correct answer is 'A. Above' from ' A. Above B. Under C. Front D. Behind', you must output 'A. Above'. Your answer format should be like "<think><Your cognitive map> your following reasoning</think><answer><Your answer></answer>"\
"""

new_str = """\
2. Next, please also provide your reasons step by step in details, then provide *ONE* correct answer selecting from the options. Your response's format should be like \"<CogMap>\n <Your cognitive map>\n<Reasoning>\n ... \n<Answer>\n Therefore, my answer is <selected option>\". Your <selected option> must be in the format like \"A. Above\". Your option must be from the available options.\
"""

data = orjsonl.load(input_file)
result = []
for item in data:
    item["question_str"] = item["question_str"].replace(replaced_str, new_str)
    result.append(item)

orjsonl.save(output_file, result)
