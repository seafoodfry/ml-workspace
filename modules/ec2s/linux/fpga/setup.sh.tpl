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

############
# NICE DCV #
############
do_install_nice_dcv() {
    # Pre-requisites according to
    # https://fpga-development-on-ec2.workshop.aws/en/3-launching-f1-instances/setting-up-gui-environment.html#installation-process
    sudo yum -y install kernel-devel
    sudo yum -y groupinstall 'Server with GUI'
    sudo yum -y groupinstall "GNOME Desktop"
    sudo yum -y install glx-utils
    sudo systemctl isolate multi-user.target
    sudo systemctl isolate graphical.target

    # Downlaod and unpack.
    sudo rpm --import https://d1uj6qtbmh3dt5.cloudfront.net/NICE-GPG-KEY
    wget https://d1uj6qtbmh3dt5.cloudfront.net/2023.1/Servers/nice-dcv-2023.1-16388-el7-x86_64.tgz
    tar -xvzf nice-dcv-2023.1-16388-el7-x86_64.tgz && cd nice-dcv-2023.1-16388-el7-x86_64

    # Install and enable.
    sudo yum install -y nice-dcv-server-2023.1.16388-1.el7.x86_64.rpm
    sudo yum install -y nice-dcv-web-viewer-2023.1.16388-1.el7.x86_64.rpm
    sudo yum install -y nice-xdcv-2023.1.565-1.el7.x86_64.rpm
    sudo yum install -y nice-dcv-gl-2023.1.1047-1.el7.x86_64.rpm
    sudo systemctl enable dcvserver
    sudo systemctl start dcvserver

    # Allow username:password convo for signing in.
    # sudo sed -i 's/#authentication="none"/authentication="system"/' /etc/dcv/dcv.conf
    # and we use grep to verify the change, where the output of bellow command should be: authentication="system"
    grep 'authentication=' /etc/dcv/dcv.conf

    # Restart the service.
    sudo systemctl restart dcvserver
    # optionally you can verify that the server has restarted correctly:
    sudo systemctl status -f dcvserver

    # Install an additional firewall.
    # TODO

    cat << 'EOF' > /home/$USERNAME/dcv-diagnostics.sh
#!/bin/bash

sudo dcvgldiag

sudo DISPLAY=:0 XAUTHORITY=$(ps aux | grep "X.*\-auth" | grep -v grep | sed -n 's/.*-auth \([^ ]\+\).*/\1/p') xhost | grep "SI:localuser:dcv$"

sudo DISPLAY=:0 XAUTHORITY=$(ps aux | grep "X.*\-auth" | grep -v grep | sed -n 's/.*-auth \([^ ]\+\).*/\1/p') glxinfo | grep -i "opengl.*version"

sudo systemctl status dcvserver

dcv list-endpoints -j

EOF
    chmod +x /home/$USERNAME/dcv-diagnostics.sh

    cat << 'EOF' > /home/$USERNAME/dcv-setup.sh
#!/bin/bash


dcv list-endpoints -j

dcv create-session f1-session

sudo passwd ec2-user

EOF
    chmod +x /home/$USERNAME/dcv-setup.sh
}

if [[ "$os_name" == "ubuntu" ]]; then
    do_ubuntu_install_basics
else
    do_install_basics
    do_install_nice_dcv
fi




date > /home/$USERNAME/CLOUDINIT-COMPLETED