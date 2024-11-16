#!/bin/bash
set -x
set -e

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

##################
# Customizations #
##################
do_install_basics() {
    sudo yum update -y
    sudo yum install -y vim
}

do_ubuntu_install_basics() {
    sudo apt-get update -y
    sudo apt-get install -y vim
}

do_install_nice_dcv() {
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

    cat << 'EOF' > /home/ec2-user/dcv-diagnostics.sh 
#!/bin/bash

sudo dcvgldiag

sudo DISPLAY=:0 XAUTHORITY=$(ps aux | grep "X.*\-auth" | grep -v grep | sed -n 's/.*-auth \([^ ]\+\).*/\1/p') xhost | grep "SI:localuser:dcv$"

sudo DISPLAY=:0 XAUTHORITY=$(ps aux | grep "X.*\-auth" | grep -v grep | sed -n 's/.*-auth \([^ ]\+\).*/\1/p') glxinfo | grep -i "opengl.*version"

sudo systemctl status dcvserver

dcv list-endpoints -j

EOF
    chmod +x /home/ec2-user/dcv-diagnostics.sh


    cat << 'EOF' > /home/ec2-user/dcv-setup.sh 
#!/bin/bash


dcv list-endpoints -j

dcv create-session dcvdemo

sudo passwd ec2-user

EOF
    chmod +x /home/ec2-user/dcv-setup.sh

}

do_install_glfw() {
    # Install GLFW.
    sudo yum install -y libX11-devel libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel
    sudo yum install -y wayland-devel wayland-protocols-devel libxkbcommon-devel

    cd /home/ec2-user
    mkdir glfw
    cd glfw/
    wget -O glfw.zip https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
    unzip glfw.zip
    cd glfw-3.4/

    cmake -S . -B build

    cd build/
    make
    sudo make install
}

if [[ "$os_name" == "ubuntu" ]]; then
    # Thus far we've only used ubuntu for ML, so we keeping it simple.
    do_ubuntu_install_basics
else
    do_install_basics
    do_install_nice_dcv
    do_install_glfw
fi

date > /home/$USERNAME/CLOUDINIT-COMPLETED
sudo reboot