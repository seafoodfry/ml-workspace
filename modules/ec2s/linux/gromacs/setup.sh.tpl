#!/bin/bash
set -euo pipefail
set -x


USERNAME="ubuntu"
LOGFILE="/home/$USERNAME/CLOUDINIT-STARTED"

log() {
    echo "$(date) - $@" >> "$LOGFILE"
}

log "Starting cloud-init script..."


log "Installing vim..."
sudo apt-get update -y
sudo apt-get install -y vim

#######################
### Install GROMACS ###
#######################
log "Installing compilers..."
sudo apt-get update -y
# Installs gcc, g++, and cmake.
sudo apt-get install build-essential cmake curl git -y

log "Downloading gromacs..."
cd /home/$USERNAME/
# Link and checksum come from
# https://manual.gromacs.org/2025.0/download.html#source-code
curl -O https://ftp.gromacs.org/gromacs/gromacs-2025.0.tar.gz
echo "4e9f043fea964cb2b4dd72d6f39ea006 gromacs-2025.0.tar.gz" | md5sum -c

log "Building gromacs..."
tar xfz gromacs-2025.0.tar.gz
cd gromacs-2025.0
mkdir build
cd build
cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON
make -j $(nproc)

log "Installing gromacs..."
sudo make install
echo "source /usr/local/gromacs/bin/GMXRC" >> /home/$USERNAME/.bashrc

log "Testing gromacs..."
make check

log "Finished installing gromacs"

# Create user setup script.
log "Creating user setup script..."
cat << 'EOF' > /home/$USERNAME/user-setup.sh
#!/bin/bash

# Install uv.
curl -LsSf https://astral.sh/uv/install.sh | sh

EOF
    chmod +x /home/$USERNAME/user-setup.sh

date > /home/$USERNAME/CLOUDINIT-COMPLETED