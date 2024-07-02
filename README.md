# FRCNN
1. Set up an environment using an AMD graphics card for GPU model training
AMD Drivers & ROCm on Ubuntu

Here are the new official instructions for installing the AMD drivers using the package manager. The necessary steps have been copied here for your convenience. Make sure your system is up to date before installing the drivers. There was an issue with kernel 6.5.0-14, so if you're running this kernel make sure you update to a newer kernel before installing ROCm. See this link for more info on this issue.

Copy and paste the following commands to install ROCm:

sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"

sudo usermod -a -G render,video $LOGNAME

Ubuntu 22.04: wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb

sudo apt install ./amdgpu-install_6.0.60002-1_all.deb

sudo apt update

sudo apt install amdgpu-dkms

sudo apt install rocm

Other Requirements

The following step is only required with certain consumer grade GPUs, or if your CPU contains an integrated GPU. If you're running a professional card, an RDNA 2 GPU with 16GB of VRAM (i.e. RX 6800 XT, 6900 XT), or a 7900 XTX/XT then the following step is not necessary. Lower tiered cards will require the following step. If your system has a CPU with an integrated GPU (Ryzen 7000) then it may also require this step.

Edit ~/.profile with the following command:

sudo nano ~/.profile

Paste the following line at the bottom of the file, then press ctrl-x and save the file.

For RDNA and RDNA 2 cards:

export HSA_OVERRIDE_GFX_VERSION=10.3.0

For RDNA 3 cards:

export HSA_OVERRIDE_GFX_VERSION=11.0.0

If your CPU contains an integrated GPU then this command might be necessary to ignore the integrated GPU and force the dedicated GPU:

export HIP_VISIBLE_DEVICES=0

Now make sure to restart your computer before continuing. Then you can check if ROCm was installed successfully by running rocminfo. If an error is returned then something went wrong with the installation. Another possibility is that secure boot may cause issues on some systems, so if you received an error here then disabling secure boot may help.
PyTorch

Next we'll download PyTorch with PIP in a Python virtual environment. First install the required software. For Ubuntu:

sudo apt install git python3-pip python3-venv libstdc++-12-dev

For Arch distros:

sudo pacman -S git python-pip python-virtualenv

Now we can install PyTorch, but it's best to create a Python virtual environment before installing packages with PIP. To create a default environment, enter:

python3 -m venv venv

The previous command should have created a folder named venv in your current directory, make sure you don't delete this folder. Next you'll need to activate that environment. Each time you open a new terminal it will need to be re-activated if you want to use PyTorch or any other PIP packages.

. venv/bin/activate

If your venv works at first but then stops working later when using PIP, you might see an error message such as this "error: externally-managed-environment". This usually happens when your system updates to a newer Python version compared to what was used to create your venv. You can solve this by deleting the old venv folder and create a new one as shown above. You'll need to reinstall the PIP packages as well.

Now PyTorch can be installed:

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

Let's verify PyTorch was installed correctly with GPU support, so lets first enter the Python console.

python3

Now enter the following two lines of code. If it returns True then everything was installed correctly.

import torch
torch.cuda.is_available()

Then enter exit() to exit the Python console.
