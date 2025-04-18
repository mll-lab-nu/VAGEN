## For Development Usage
```
# Start a General Server in debug mode
python vagen/env/server.py --debug
```
### Navigation
```

# For headless servers, additional setup is required:
# Install required packages
apt-get install -y pciutils
apt-get install -y xorg xserver-xorg-core xserver-xorg-video-dummy

# Start X server in a tmux window
python vagen/env/navigation/startx.py 1
```

### Maniskill
python -m mani_skill.utils.download_asset "PickSingleYCB-v1"
python -m mani_skill.utils.download_asset partnet_mobility_cabinet