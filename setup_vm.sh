#!/bin/bash

# Update package list and upgrade existing packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3 python3-pip

# Install tmux
sudo apt install -y tmux

sudo apt install -y libgl1-mesa-glx

sudo apt install -y python3-venv

# Git
sudo apt install git -y

# Verify installations
echo "Python version:"
python --version

echo "Pip version:"
pip --version

echo "Tmux version:"
tmux -V

echo "Installation complete!"
