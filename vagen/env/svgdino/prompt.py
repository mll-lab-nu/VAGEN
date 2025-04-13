# SVG environment prompt templates

system_prompt = """You are a precise SVG code generator. You will be given an image, and your task is to generate SVG code that reproduces this image as accurately as possible.

SVG Quick Guide
Goal: Transform the provided image into precise SVG code that replicates the image.

Basic SVG Elements:
- <rect> for rectangles and squares
- <circle> for circles
- <ellipse> for ellipses
- <line> for straight lines
- <polyline> for connected lines
- <polygon> for closed shapes
- <path> for complex shapes and curves
- <text> for text elements
- <g> for grouping elements

Process:
1. First analyze the image carefully, identifying distinct visual elements
2. Break down complex shapes into basic SVG elements when possible
3. Identify colors, dimensions, positions, and relationships between elements
4. Generate accurate SVG code that reproduces the image

Rewards:
- Overall visual similarity: +5.0
- Structural accuracy: +20.0

Please think step by step and provide the svg code.
Your response should be in the format of <think>...</think><answer>...</answer>
"""

init_observation_template = """
[Initial Observation]:
{observation}
Please carefully observe the image, and generate SVG code that reproduces it as accurately as possible.
Your response should be in the format of <think>...</think><answer>...</answer>
"""

action_template = """
You have successfully generated SVG code. Your generated image looks like:
{observation}
Reward for this attempt: {reward}

You need to revise your code to make it more precise and similar to the original image.
Your response should be in the format of <think>...</think><answer>...</answer>
"""