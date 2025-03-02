#!/bin/bash
set -x


USERNAME="ubuntu"
date > /home/$USERNAME/CLOUDINIT-STARTED


sudo apt-get update -y
sudo apt-get install -y vim

#######################
### Install GROMACS ###
#######################
do_install_gromacs() {
    sudo apt-get update -y
    # Installs gcc, g++, and cmake.
    sudo apt-get install build-essential cmake curl git -y

    cd /home/$USERNAME/
    # Link and checksum come from
    # https://manual.gromacs.org/2025.0/download.html#source-code
    curl -O https://ftp.gromacs.org/gromacs/gromacs-2025.0.tar.gz
    echo "4e9f043fea964cb2b4dd72d6f39ea006 gromacs-2025.0.tar.gz" | md5sum -c

    tar xfz gromacs-2025.0.tar.gz
    cd gromacs-2025.0
    mkdir build
    cd build
    cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON
    make -j $(nproc)
    make check
    sudo make install
    echo "source /usr/local/gromacs/bin/GMXRC" >> /home/$USERNAME/.bashrc
}

echo "starting gromacs installation..." >> /home/$USERNAME/CLOUDINIT-STARTED
date >> /home/$USERNAME/CLOUDINIT-STARTED
do_install_gromacs
echo "finished gromacs installation" >> /home/$USERNAME/CLOUDINIT-STARTED
date >> /home/$USERNAME/CLOUDINIT-STARTED

# Create user setup script.
cat << 'EOF' > /home/$USERNAME/user-setup.sh
#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh

EOF
    chmod +x /home/$USERNAME/user-setup.sh

date > /home/$USERNAME/CLOUDINIT-COMPLETED