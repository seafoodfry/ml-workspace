#!/bin/bash
set -x

sudo yum update -y
sudo yum install -y vim
# Install compilers, make, etc.
sudo yum groupinstall -y "Development Tools"

sudo yum install -y xorg-x11-xauth xorg-x11-apps glx-utils