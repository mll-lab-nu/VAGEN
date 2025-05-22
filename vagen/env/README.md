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

Below is outdated information kept for backup purposes:
```
# export CUDA_VISIBLE_DEVICES
# For headless servers, additional setup is required:
# Install required packages
sudo apt-get install -y pciutils
sudo apt-get install -y xorg xserver-xorg-core xserver-xorg-video-dummy
#Start X server in a tmux window
python vagen/env/navigation/startx.py 1
```

### PrimitiveSkill
```
pip install mani_skill==3.0.0b20
python -m mani_skill.utils.download_asset "PickSingleYCB-v1" -y
python -m mani_skill.utils.download_asset partnet_mobility_cabinet -y

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
pip3 install "pydantic<2.0.0"
pip3 uninstall frozenlist gradio murmurhash preshed spacy srsly thinc weasel aiosignal annotated-types blis catalogue cloudpathlib cymem

# skip this two install if you already installed in navigation
apt-get install -y pciutils
apt-get install -y xorg xserver-xorg-core xserver-xorg-video-dummy

# Set the data path and download before running the server
export ALFWORLD_DATA=<storage_path>
# or you can add this path directly into bashrc
# echo 'export ALFWORLD_DATA=<storage_path>' >> ~/.bashrc

# download dataset
alfworld-download

# on new windows, start a startx port and then start server
python vagen/env/alfworld/startx.py 0
python vagen/server/server.py
```

## Benchmark your Env and Service
env/service running time varies on different devices, you can benchmark current env/service or debug your own env/service as follow:
### Start a env benchmark
```
# run env script
./scripts/benchmark/env_benchmark/frozenlake/run.sh
```

### Service test
```
# start a server in a tmux session
python vagen/server/server.py

# run service script
./scripts/benchmark/service_benchmark/frozenlake/run.sh
```

