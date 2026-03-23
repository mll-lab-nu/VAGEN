# SVG (Image-to-SVG Code Generation)

SVG is an image-to-code environment where the LLM generates SVG code to reproduce a target image. Scoring uses DINO, DreamSim, and structural similarity.

## Installation

Install the additional dependencies:

```bash
pip install beautifulsoup4 lxml cairosvg svgpathtools dreamsim fire
```

The dataset (`starvector/svg-icons-simple`) and model checkpoints (DINOv2, DreamSim) are downloaded automatically on first server startup.

## Evaluation

Start the SVG server, then run evaluation:

```bash
# Terminal 1: start server
python -m vagen.envs.svg.serve

# Terminal 2: run eval
python -m vagen.evaluate.run_eval --config examples/evaluate/svg/config.yaml
```

## Training

Start the SVG server, then run training:

```bash
# Terminal 1: start server
python -m vagen.envs.svg.serve

# Terminal 2: run training
# (update examples/train/svg/train_svg.yaml with your training script)
```
