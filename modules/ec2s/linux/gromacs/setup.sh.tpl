#!/bin/bash
set -x


USERNAME="ubuntu"
date > /home/$USERNAME/CLOUDINIT-STARTED


sudo apt-get update -y
sudo apt-get install -y vim


date > /home/$USERNAME/CLOUDINIT-COMPLETED