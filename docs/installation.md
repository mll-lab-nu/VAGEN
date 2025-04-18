

Before running experiments, ensure you have set up the environment properly:

```bash
# Create a new conda environment
conda create -n vagen python=3.10 -y
conda activate vagen

# Install verl
git clone https://github.com/JamesKrW/verl.git
cd verl
pip install -e .
cd ../

# Install VAGEN
git clone https://github.com/RAGEN-AI/VAGEN.git
cd VAGEN
bash scripts/install.sh

# go to release branch of verl
cd ../verl
git checkout release
cd ../VAGEN

# Login to wandb for experiment tracking
wandb login
```