#!/bin/bash
set -x

USERNAME="ec2-user"

os_name=$(grep ^ID= /etc/os-release | cut -d'=' -f2 | tr -d '"')
echo "working on $os_name..."
if [ "$os_name" == "ubuntu" ]; then
    echo "Running on Ubuntu"
    USERNAME="ubuntu"
else
    echo "Unknown OS"
fi

date > /home/$USERNAME/CLOUDINIT-STARTED


sudo yum update -y
sudo yum install -y vim
# Install compilers, make, etc.
sudo yum groupinstall -y "Development Tools"

sudo yum install -y xorg-x11-xauth xorg-x11-apps glx-utils

##############################
#       Install Docker       #
##############################
do_install_docker() {
    # Installation instructions come from https://docs.docker.com/engine/install/ubuntu/
    for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
        sudo apt-get remove -y $pkg;
    done

    # Add Docker's official GPG key:
    sudo apt-get update -y
    sudo apt-get install -y ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y

    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Post installation instructions come from https://docs.docker.com/engine/install/linux-postinstall/
    sudo groupadd docker
    sudo usermod -aG docker $USER
}


if [ "${install_docker}" = "true" ]; then
    do_install_docker
else
    echo "Skipping Docker installation."
fi

date > /home/$USERNAME/CLOUDINIT-COMPLETED