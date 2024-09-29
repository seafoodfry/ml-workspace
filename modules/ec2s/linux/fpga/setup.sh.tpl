#!/bin/bash
set -x

USERNAME="ec2-user"

os_name=$(grep ^ID= /etc/os-release | cut -d'=' -f2 | tr -d '"')
echo "working on $os_name..."
if [ "$os_name" == "ubuntu" ]; then
    echo "Running on Ubuntu"
    USERNAME="ubuntu"
elif [ "$os_name" == "centos" ]; then
    echo "Running on CentOS"
    USERNAME="centos"
else
    echo "Unknown OS"
fi

date > /home/$USERNAME/CLOUDINIT-STARTED


sudo yum update -y
sudo yum install -y vim





date > /home/$USERNAME/CLOUDINIT-COMPLETED