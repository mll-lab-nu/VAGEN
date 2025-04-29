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
sudo apt-get update && sudo apt-get install -y libcairo2
```

### Navigation
```
pip install ai2thor==5.0.0
pip install numpy==1.25.1

# Refer to https://github.com/EmbodiedBench/EmbodiedBench, probably you also need:
sudo apt-get update && sudo apt-get -y install libvulkan1
sudo apt install vulkan-tools
```

Below is outdated for backup purpose:
```
# export CUDA_VISIBLE_DEVICES
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
sudo apt-get update && sudo apt-get install -y libx11-6
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx
```

### ALFWorld
```
pip install ai2thor==2.1.0
pip install alfworld==0.3.2
pip3 install numpy==1.23.5
pip3 install protobuf==3.20.3
pip3 install pydantic==1.10.14
pip3 install pydantic-core==2.16.3
pip3 uninstall frozenlist gradio murmurhash preshed spacy srsly thinc weasel aiosignal annotated-types blis catalogue cloudpathlib cymem

# Set the data path and download before running the server
export ALFWORLD_DATA=<storage_path>
alfworld-download

# on a new window, start a startx port and then start server
python vagen/env/server.py
```
