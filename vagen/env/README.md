### Navigation
```
# Additional dependencies:
pip install ai2thor==5.0.0
pip install numpy==1.25.1

# For headless servers, additional setup is required:
# Install required packages
apt-get install -y pciutils
apt-get install -y xorg xserver-xorg-core xserver-xorg-video-dummy

# Start X server in a tmux window
python vagen/env/navigation/startx.py 1

```


### FrozenLake
```
# Additional dependencies:
pip install gymnasium
pip install "gymnasium[toy-text]"
```

### SVG
```
# Additional dependencies:
pip install bs4
pip install svgpathtools
pip install cairosvg

# Then run experiment of SVG simply copy the code below
bash vagen\examples\debug_svg_vision_grpo\run.sh
```

### SVGDino
```
# Additional dependencies:
pip install bs4
pip install svgpathtools
pip install cairosvg
pip install flask

# create server for reward model
python vagen/env/svgdino/reward_model_server.py --port 5000

# Then run experiment of SVG simply copy the code below
bash vagen\examples\debug_svgdino_vision_grpo\run.sh
```