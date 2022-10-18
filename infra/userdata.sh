#!/bin/bash
set -x

# Adapted from https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/setup/pytorch-install.html

# Configure linux for Neuron repository updates
tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
[neuron]
name=Neuron YUM Repository
baseurl=https://yum.repos.neuron.amazonaws.com
enabled=1
metadata_expire=0
EOF
rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB

# Install OS headers
yum install "kernel-devel-$(uname -r)" "kernel-headers-$(uname -r)" -y

# Update OS packages
yum update -y

# Remove preinstalled packages and install Neuron driver and runtime
yum remove aws-neuron-dkms -y
yum remove aws-neuronx-dkms -y
yum remove aws-neuronx-oci-hook -y
yum remove aws-neuronx-runtime-lib -y
yum remove aws-neuronx-collectives -y
yum install aws-neuronx-dkms-2.* -y
yum install aws-neuronx-oci-hook-2.* -y
yum install aws-neuronx-runtime-lib-2.* -y
yum install aws-neuronx-collectives-2.* -y

# Install EFA Driver (only required for multi-instance training)
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
gpg --fingerprint < aws-efa-installer.key
wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig
tar -xvf aws-efa-installer-latest.tar.gz
cd aws-efa-installer && bash efa_installer.sh --yes
cd ..
rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

# Remove pre-installed package and install Neuron tools
yum remove aws-neuron-tools -y
yum remove aws-neuronx-tools -y
yum install aws-neuronx-tools-2.*  -y

# Install dev tools
yum install git -y

# Set up python venv and install Neuron packages
sudo -u "ec2-user" -i  <<EOF
set -x
cd \$HOME
python3.7 -m venv aws_neuron_venv_pytorch_p37
source aws_neuron_venv_pytorch_p37/bin/activate
python -m pip install -U pip

# Install packages from beta repos
python -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"

# Install Python packages
python -m pip install torch-neuronx=="1.11.0.1.*" "neuronx-cc==2.*" transformers

# Update env vars and activate virtual env
cat >> \$HOME/aws_neuron_venv_pytorch_p37/bin/activate <<EOS
# Add Neuron tools to PATH variable
export PATH=/opt/aws/neuron/bin/:\$PATH
EOS

cat >> /home/ec2-user/.bashrc <<EOS
# Activate Neuron virtual environment
source /home/ec2-user/aws_neuron_venv_pytorch_p37/bin/activate
EOS

# Check Neuron installation
source \$HOME/.bashrc
neuron-ls
EOF
