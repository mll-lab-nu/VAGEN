## For Debugging
```
# Start a General Server in debug mode
python vagen/env/server.py --debug
```

## Environment Installation
### SVG
```
pip install "bs4"
pip install "svgpathtools"
pip install "cairosvg"
pip install "dreamsim"

# Probably you also need:
apt-get update && apt-get install -y libcairo2
```

### Navigation
```
pip install ai2thor==5.0.0
pip install numpy==1.25.1

# Refer to https://github.com/EmbodiedBench/EmbodiedBench, probably you also need:
apt-get -y install libvulkan1
apt install vulkan-tools
```

Below is outdated for backup purpose:
```
# For headless servers, additional setup is required:
# Install required packages
apt-get install -y pciutils
apt-get install -y xorg xserver-xorg-core xserver-xorg-video-dummy
#Start X server in a tmux window
python vagen/env/navigation/startx.py 1
```

### PrimitiveSkill
```
pip install --upgrade mani_skill
python -m mani_skill.utils.download_asset "PickSingleYCB-v1"
python -m mani_skill.utils.download_asset partnet_mobility_cabinet

# Refer to https://github.com/haosulab/ManiSkill, probably you also need:
apt-get update && apt-get install -y libx11-6
apt-get update && apt-get install -y libgl1-mesa-glx
```
