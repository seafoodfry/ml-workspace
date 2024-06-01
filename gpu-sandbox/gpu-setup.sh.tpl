#!/bin/bash
set -x

sudo yum update -y
sudo yum install -y freeglut-devel mesa-libGL-devel

sudo yum install -y xorg-x11-xauth xorg-x11-apps glx-utils

# Taken from https://docs.aws.amazon.com/dcv/latest/adminguide/setting-up-installing-linux-prereq.html
sudo yum install -y gdm gnome-session gnome-classic-session gnome-session-xsession
sudo yum install -y xorg-x11-server-Xorg xorg-x11-fonts-Type1 xorg-x11-drivers 
sudo yum install -y gnome-terminal gnu-free-fonts-common gnu-free-mono-fonts gnu-free-sans-fonts gnu-free-serif-fonts
sudo yum -y upgrade

sudo yum install -y glx-utils

sudo reboot