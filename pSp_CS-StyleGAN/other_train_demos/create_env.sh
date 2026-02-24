#!/bin/bash

# conda remove --name pytorch124a --all -y
# conda remove --name pytorch124b --all -y
# Create a new Conda environment
conda create --name styleGANenv python=3.11 -y

conda activate styleGANenv
# conda remove --name myenv --all
# we can clone the environment for backup
# conda create --name pytorch124b --clone pytorch124a

#!/bin/bash

############################################
# Install required Python packages
pip install torch torchvision torchaudio \
pip install matplotlib opencv-python ninja pyarrow pytorch_fid deepface scikit-learn 
pip install notebook jupyterlab
#############################################
#!rm -rf /home/ids/yuhe/.cache/torch_extensions







# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# pip3 install torch torchvision torchaudio
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# pip3 install matplotlib opencv-python ninja pyarrow pytorch_fid scikit-learn
# pip3 install notebook jupyterlab

## conda remove --name pytorch12.1_env --all
## conda create --name styleGANmlp --clone pytorch12.1_env 

## conda create --name myclone --clone myenv

# conda config --set auto_activate_base true
# conda config --set auto_activate_base false


# conda create -n cuda11.8env python=3.11

# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia


# pip install matplotlib

# pip install opencv-python

# pip install ninja


# conda create --name myclone --clone myenv

# conda remove --name myenv --all
# conda info --envs
# pip uninstall numpy
# pip install numpy==1.26.4


# Cloning an environment
# Use the terminal for the following steps:
# You can make an exact copy of an environment by creating a clone of it:
# conda create --name myclone --clone myenv

# Note
# Replace myclone with the name of the new environment. Replace myenv with the name of the existing environment that you want to copy.


# https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html



# Removing an environment
# To remove an environment, in your terminal window, run:


# You may instead use conda env remove --name myenv.
# To verify that the environment was removed, in your terminal window, run:
# conda info --envs


# Create an environment for styleGAN mlp
# Alternative: Using a Shell Script
# If you plan to use this frequently, you can save the commands in a shell script (e.g., setup_env.sh):

# #!/bin/bash
# conda create --name pytorch12.1 python=3.11 -y
# conda activate pytorch12.1_env
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# pip install matplotlib opencv-python ninja pyarrow

# chmod +x setup_env.sh

# conda create --name styleGANmlp --clone pytorch12.1_env 


# pip uninstall jupyter jupyterlab notebook -y 
# pip install notebook jupyterlab
# pip install pyarrow









