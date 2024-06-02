#!/bin/bash
set -x
set -e

sudo yum update -y
sudo yum install -y freeglut-devel mesa-libGL-devel

sudo yum install -y xorg-x11-xauth xorg-x11-apps glx-utils

# Taken from https://docs.aws.amazon.com/dcv/latest/adminguide/setting-up-installing-linux-prereq.html
sudo yum install -y gdm gnome-session gnome-classic-session gnome-session-xsession
sudo yum install -y xorg-x11-server-Xorg xorg-x11-fonts-Type1 xorg-x11-drivers 
sudo yum install -y gnome-terminal gnu-free-fonts-common gnu-free-mono-fonts gnu-free-sans-fonts gnu-free-serif-fonts
#sudo yum -y upgrade

sudo yum install -y glx-utils

sudo nvidia-xconfig --preserve-busid --enable-all-gpus
sudo nvidia-xconfig --preserve-busid --enable-all-gpus
sudo rm -rf /etc/X11/XF86Config*
sudo systemctl isolate multi-user.target
sudo systemctl isolate graphical.target


sudo rpm --import https://d1uj6qtbmh3dt5.cloudfront.net/NICE-GPG-KEY
wget https://d1uj6qtbmh3dt5.cloudfront.net/2023.1/Servers/nice-dcv-2023.1-16388-el7-x86_64.tgz
tar -xvzf nice-dcv-2023.1-16388-el7-x86_64.tgz && cd nice-dcv-2023.1-16388-el7-x86_64
sudo yum install -y nice-dcv-server-2023.1.16388-1.el7.x86_64.rpm
sudo yum install -y nice-dcv-web-viewer-2023.1.16388-1.el7.x86_64.rpm
sudo yum install -y nice-xdcv-2023.1.565-1.el7.x86_64.rpm
sudo yum install -y nice-dcv-gl-2023.1.1047-1.el7.x86_64.rpm

sudo systemctl enable dcvserver

touch /home/ec2-user/CLOUDINIT-COMPLETED

sudo reboot