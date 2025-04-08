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
