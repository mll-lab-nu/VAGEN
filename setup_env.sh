#!/bin/bash
# Exit on error
set -e

# ======================
# USER CONFIGURABLE PARAMETERS
# ======================
# Set these values before running the script
GITHUB_USERNAME=""
GITHUB_EMAIL=""
WANDB_API_KEY=""  # Add your Weights & Biases API key here
TOGETHER_API_KEY=""  # Add your Together API key here if you have one

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting environment setup script...${NC}"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to validate parameters
validate_parameters() {
  local missing_params=0

  if [ -z "$GITHUB_USERNAME" ]; then
    echo -e "${RED}ERROR: GITHUB_USERNAME is not set at the top of the script${NC}"
    missing_params=1
  fi

  if [ -z "$GITHUB_EMAIL" ]; then
    echo -e "${RED}ERROR: GITHUB_EMAIL is not set at the top of the script${NC}"
    missing_params=1
  fi

  if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${RED}WARNING: WANDB_API_KEY is not set at the top of the script${NC}"
    echo -e "${YELLOW}You will need to manually login to wandb later${NC}"
  fi

  if [ $missing_params -eq 1 ]; then
    echo -e "${RED}Please set the required parameters at the top of the script and try again.${NC}"
    exit 1
  fi
}

# ======================
# NEW: PERSISTENT ENV VARIABLES
# ======================
setup_persistent_env_vars() {
  echo -e "${YELLOW}Setting up persistent environment variables...${NC}"
  
  # Determine shell config files to update
  SHELL_TYPE=$(basename "$SHELL")
  BASH_CONFIG="$HOME/.bashrc"
  ZSH_CONFIG="$HOME/.zshrc"
  
  # Define the environment variables to add
  ENV_VARS=(
    "export WANDB_API_KEY=\"$WANDB_API_KEY\""
  )
  
  # Add Together API key if provided
  if [ -n "$TOGETHER_API_KEY" ]; then
    ENV_VARS+=("export TOGETHER_API_KEY=\"$TOGETHER_API_KEY\"")
  fi
  
  # Function to add environment variables to a config file
  add_env_to_config() {
    local config_file="$1"
    
    if [ -f "$config_file" ]; then
      echo -e "${YELLOW}Adding environment variables to $config_file...${NC}"
      
      # Add a section marker for our environment variables
      echo "" >> "$config_file"
      echo "# === Environment Variables added by setup script ===" >> "$config_file"
      
      # Add each environment variable, avoiding duplicates
      for env_var in "${ENV_VARS[@]}"; do
        # Extract the variable name from the export command
        var_name=$(echo "$env_var" | cut -d'=' -f1 | cut -d' ' -f2)
        
        # Check if the variable is already in the file
        if ! grep -q "^export $var_name=" "$config_file"; then
          echo "$env_var" >> "$config_file"
          echo -e "${GREEN}Added $var_name to $config_file${NC}"
        else
          echo -e "${YELLOW}$var_name already exists in $config_file, updating...${NC}"
          # Remove existing line and add updated one
          sed -i.bak "/^export $var_name=/d" "$config_file"
          echo "$env_var" >> "$config_file"
        fi
      done
    else
      echo -e "${YELLOW}$config_file does not exist, skipping...${NC}"
    fi
  }
  
  # Update bash config
  add_env_to_config "$BASH_CONFIG"
  
  # Update zsh config if it exists
  if [ -f "$ZSH_CONFIG" ]; then
    add_env_to_config "$ZSH_CONFIG"
  fi
  
  echo -e "${GREEN}Environment variables will now persist across all shell sessions.${NC}"
  echo -e "${YELLOW}You'll need to restart your terminal or run 'source ~/.bashrc' (or ~/.zshrc) for them to take effect in the current session.${NC}"
}

# ======================
# 1. MINICONDA SETUP
# ======================
setup_miniconda() {
  echo -e "${YELLOW}Setting up Miniconda for Ubuntu...${NC}"
  
  # Use Linux URL for Ubuntu
  MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  
  MINICONDA_INSTALLER="/tmp/miniconda.sh"
  echo "Downloading Miniconda from $MINICONDA_URL"
  curl -sSL "$MINICONDA_URL" -o "$MINICONDA_INSTALLER"
  
  bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
  
  CONDA_PATH="$HOME/miniconda3/bin"
  echo "export PATH=\"$CONDA_PATH:\$PATH\"" >> "$HOME/.bashrc"
  
  if [ -f "$HOME/.zshrc" ]; then
    echo "export PATH=\"$CONDA_PATH:\$PATH\"" >> "$HOME/.zshrc"
  fi
  
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  
  "$HOME/miniconda3/bin/conda" init bash
  if [ -f "$HOME/.zshrc" ]; then
    "$HOME/miniconda3/bin/conda" init zsh
  fi
  
  rm "$MINICONDA_INSTALLER"
  
  echo -e "${GREEN}Miniconda installed successfully!${NC}"
}

# ======================
# NEW: INSTALL GIT LFS AND UNZIP FOR UBUNTU
# ======================
install_git_lfs_and_unzip() {
  echo -e "${YELLOW}Installing Git LFS and unzip for Ubuntu...${NC}"
  
  # Update package lists
  echo -e "${YELLOW}Updating package lists...${NC}"
  sudo apt-get update
  
  # Install Git LFS
  if command_exists git-lfs; then
    echo -e "${GREEN}Git LFS is already installed.${NC}"
  else
    echo -e "${YELLOW}Installing Git LFS...${NC}"
    sudo apt-get install -y git-lfs
    git lfs install
    echo -e "${GREEN}Git LFS installed successfully!${NC}"
  fi
  
  # Install unzip
  if command_exists unzip; then
    echo -e "${GREEN}unzip is already installed.${NC}"
  else
    echo -e "${YELLOW}Installing unzip...${NC}"
    sudo apt-get install -y unzip
    echo -e "${GREEN}unzip installed successfully!${NC}"
  fi
  
  # Verify installation
  echo -e "${YELLOW}Verifying Git LFS installation...${NC}"
  git lfs version
  
  echo -e "${YELLOW}Verifying unzip installation...${NC}"
  unzip -v | head -n 1
  
  echo -e "${GREEN}Git LFS and unzip are now ready to use.${NC}"
}

# ======================
# 2. VIM INSTALLATION
# ======================
install_vim() {
  echo -e "${YELLOW}Checking and installing vim for Ubuntu...${NC}"
  if command_exists vim; then
    echo "vim is already installed."
  else
    echo -e "${YELLOW}Installing vim...${NC}"
    sudo apt-get update && sudo apt-get install -y vim
  fi
  echo -e "${GREEN}Vim setup complete!${NC}"
}

# ======================
# 3. GIT CONFIGURATION
# ======================
setup_git_config() {
  echo -e "${YELLOW}Setting up Git configuration...${NC}"
  
  # Set global Git configuration
  git config --global user.name "$GITHUB_USERNAME"
  git config --global user.email "$GITHUB_EMAIL"
  
  # Set some helpful Git aliases
  git config --global alias.co checkout
  git config --global alias.br branch
  git config --global alias.ci commit
  git config --global alias.st status
  git config --global alias.unstage 'reset HEAD --'
  git config --global alias.last 'log -1 HEAD'
  
  # Set default branch name to main
  git config --global init.defaultBranch main
  
  # Configure Git to use the fancy diff highlighting when available
  git config --global color.ui auto
  
  echo -e "${GREEN}Git configuration complete!${NC}"
  echo -e "Git user: $(git config --global user.name)"
  echo -e "Git email: $(git config --global user.email)"
}

# ======================
# 4. SSH KEYS SETUP
# ======================
setup_ssh_keys() {
  echo -e "${YELLOW}Setting up SSH keys for GitHub...${NC}"
  
  SSH_DIR="$HOME/.ssh"
  KEY_FILE="$SSH_DIR/id_ed25519_github"
  PUB_KEY_FILE="${KEY_FILE}.pub"
  
  mkdir -p "$SSH_DIR"
  chmod 700 "$SSH_DIR"
  
  # If the public key already exists, just output it, don't generate a new one
  if [ -f "$PUB_KEY_FILE" ]; then
    echo -e "${GREEN}SSH key already exists. Here is your public key:${NC}"
    echo ""
    cat "$PUB_KEY_FILE"
    echo ""
    echo -e "${YELLOW}Make sure this key is added to your GitHub account at:${NC}"
    echo "https://github.com/settings/keys"
    return
  fi
  
  ssh-keygen -t ed25519 -f "$KEY_FILE" -N ""
  
  if [ ! -f "$SSH_DIR/config" ]; then
    touch "$SSH_DIR/config"
    chmod 600 "$SSH_DIR/config"
  fi
  
  # Check if the GitHub config already exists in the SSH config file
  if ! grep -q "Host github.com" "$SSH_DIR/config"; then
    cat >> "$SSH_DIR/config" << EOF
# GitHub configuration
Host github.com
  HostName github.com
  User git
  IdentityFile $KEY_FILE
  IdentitiesOnly yes
EOF
  fi
  
  eval "$(ssh-agent -s)"
  ssh-add "$KEY_FILE"
  
  echo -e "${GREEN}SSH keys generated successfully!${NC}"
  echo -e "${YELLOW}Your GitHub SSH public key:${NC}"
  echo ""
  cat "$PUB_KEY_FILE"
  echo ""
  echo -e "${YELLOW}Add this key to your GitHub account at:${NC}"
  echo "https://github.com/settings/keys"
}

# ======================
# 5. WANDB SETUP
# ======================
setup_wandb() {
  echo -e "${YELLOW}Setting up Weights & Biases...${NC}"
  
  # Check if wandb is installed
  if ! command_exists wandb; then
    echo -e "${YELLOW}Installing wandb CLI...${NC}"
    pip install wandb
  fi
  
  # Login to wandb if API key is provided
  if [ -n "$WANDB_API_KEY" ]; then
    echo -e "${YELLOW}Logging in to Weights & Biases...${NC}"
    wandb login "$WANDB_API_KEY"
    echo -e "${GREEN}Successfully logged in to Weights & Biases!${NC}"
  else
    echo -e "${YELLOW}No Weights & Biases API key provided.${NC}"
    echo -e "${YELLOW}You can log in manually later with:${NC} wandb login"
  fi
}

# ======================
# 6. VAGEN SETUP
# ======================
setup_vagen() {
  echo -e "${YELLOW}Setting up VAGEN environment...${NC}"
  
  # Source conda to make it available in current shell
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniconda3/bin/activate" ]]; then
    source "$HOME/miniconda3/bin/activate"
  fi

  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  
  # Create conda environment
  echo -e "${YELLOW}Creating vagen conda environment...${NC}"
  conda create -n vagen python=3.10 -y
  
  # Activate the environment
  conda activate vagen
  
  # Clone and install verl
  echo -e "${YELLOW}Cloning and installing verl...${NC}"
  if [ -d "verl" ]; then
    echo "verl directory already exists, updating..."
    cd verl
    git pull
  else
    git clone https://github.com/JamesKrW/verl.git
    cd verl
  fi
  pip install -e .
  cd ../
  
  # Clone and install vagen
  echo -e "${YELLOW}Cloning and installing vagen...${NC}"
  if [ -d "VAGEN" ]; then
    echo "vagen directory already exists, updating..."
    cd VAGEN
    git pull
  else
    git clone https://github.com/williamzhangNU/VAGEN.git
    cd VAGEN
  fi
  git checkout crossview_py
  bash scripts/install.sh
  echo -e "${YELLOW}Setting up CrossView environment...${NC}"
  cd vagen/env/crossview
  
  # Clone CrossViewQA dataset and pull LFS files
  git clone https://huggingface.co/datasets/MLL-Lab/MindCube
  cd MindCube
  git lfs pull
  unzip data.zip -d extracted_images
  mv extracted_images/data/* extracted_images/*
  rm -r extracted_images/data
  # TODO
  cd ..

  # Setup wandb
  setup_wandb
  
  echo -e "${GREEN}VAGEN environment setup complete!${NC}"
  cd ../
}

# Main execution
main() {
  # Validate user parameters first
  validate_parameters
  
  # Miniconda setup
  if command_exists conda; then
    echo -e "${GREEN}Miniconda is already installed. Skipping installation.${NC}"
  else
    setup_miniconda
    
    # Make conda available in the current shell session after installation
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/miniconda3/bin/activate" ]]; then
      source "$HOME/miniconda3/bin/activate"
    fi
  fi
  
  # Install Git LFS and unzip - NEW ADDITION
  install_git_lfs_and_unzip
  
  # Install vim
  install_vim
  
  # Configure Git
  setup_git_config
  
  # Set up SSH Key
  setup_ssh_keys
  
  # Set up persistent environment variables
  setup_persistent_env_vars
  
  # Set up VAGEN environment
  setup_vagen
  
  echo -e "${GREEN}Setup complete! You may need to restart your terminal for all changes to take effect.${NC}"
  echo -e "${YELLOW}To apply environment variables in the current shell, run:${NC}"
  echo -e "source ~/.bashrc  # or ~/.zshrc if using zsh"
  
  if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${YELLOW}Remember to log in to Weights & Biases with:${NC} wandb login"
  fi
}

main
