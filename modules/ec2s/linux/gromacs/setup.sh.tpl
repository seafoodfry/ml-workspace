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


log "Installing compilers..."
sudo apt-get update -y
# Installs gcc, g++, and cmake.
sudo apt-get install build-essential cmake curl git -y

# NOTE: g5g specific steps...
# GPU support for gromacs requires cmake >= 3.28 and ubuntu has 3.21.
# The pytorch cmake is 3.26.
#mv /opt/pytorch/bin/cmake /opt/pytorch/bin/pytorch-cmake
wget https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-aarch64.sh
sudo bash cmake-3.31.6-linux-aarch64.sh --prefix=/usr/local --skip-license

###########
### NFS ###
###########
# See https://docs.aws.amazon.com/efs/latest/ug/mounting-fs-install-nfsclient.html
log "Installing NFS client..."
sudo apt-get -y install nfs-common

##########################
### OpenMPI and OpenMP ###
##########################
log "Installing openmpi..."
sudo apt-get install -y openmpi-bin libopenmpi-dev libomp-dev


# Create user setup script.
log "Creating user setup script..."
cat << 'EOF' > /home/$USERNAME/user-setup.sh
#!/bin/bash
set -euo pipefail
set -x

NFS_DNS="$1"

# Install uv.
curl -LsSf https://astral.sh/uv/install.sh | sh

# Mount EFS.
# See
# https://docs.aws.amazon.com/efs/latest/ug/mounting-fs-mount-cmd-dns-name.html
# and
# https://docs.aws.amazon.com/efs/latest/ug/nfs-automount-efs.html
sudo mkdir -p /mnt/efs

sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport ${NFS_DNS}:/ /mnt/efs

sudo chown -R ${USER}:${USER} /mnt/efs

EOF
    chmod +x /home/$USERNAME/user-setup.sh
    chown ${USERNAME}:${USERNAME} /home/$USERNAME/user-setup.sh


# Create user setup script.
log "Creating ssh step 1 script..."
cat << 'EOF' > /home/$USERNAME/config-ssh-pt1.sh
#!/bin/bash
set -euo pipefail
set -x

# Create shared directory on EFS.
SHARED_DIR="/mnt/efs/ssh_pubkeys"
mkdir -p $SHARED_DIR

# Get hostname for unique key identification.
HOSTNAME=$(hostname)

# Generate key if it doesn't exist.
if [ ! -f ~/.ssh/id_rsa_mpi ]; then
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa_mpi
    echo "Generated new SSH key for MPI"
else
    echo "SSH key already exists, using existing key"
fi

# Create a combined keys file for easier setup.
cat ~/.ssh/id_rsa_mpi.pub >> $SHARED_DIR/all_nodes.pub
echo "Added key to combined keys file"

echo "Key generation and sharing complete. Run the setup script on each node."
EOF
    chmod +x /home/$USERNAME/config-ssh-pt1.sh
    chown ${USERNAME}:${USERNAME} /home/$USERNAME/config-ssh-pt1.sh

log "Creating ssh step 2 script..."
cat << 'EOF' > /home/$USERNAME/config-ssh-pt2.sh
#!/bin/bash
set -euo pipefail
set -x

SHARED_DIR="/mnt/efs/ssh_pubkeys"

# Add all public keys to authorized_keys.
cat $SHARED_DIR/all_nodes.pub >> ~/.ssh/authorized_keys

# Create SSH config to use the right key and disable host checking.
cat > ~/.ssh/config << INNEREOF
Host ip-10-0-101-*
  IdentityFile ~/.ssh/id_rsa_mpi
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
INNEREOF
chmod 600 ~/.ssh/config
EOF
    chmod +x /home/$USERNAME/config-ssh-pt2.sh
    chown ${USERNAME}:${USERNAME} /home/$USERNAME/config-ssh-pt2.sh

#######################
### Install GROMACS ###
#######################
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
export=/usr/local/cuda/bin/nvcc
cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON -DGMX_GPU=CUDA -DCMAKE_CUDA_ARCHITECTURES=native #-DGMX_MPI=ON
make -j $(nproc)

log "Installing gromacs..."
sudo make install
echo "source /usr/local/gromacs/bin/GMXRC" >> /home/$USERNAME/.bashrc

#log "Testing gromacs..."
#make check

log "Finished installing gromacs"

date > /home/$USERNAME/CLOUDINIT-COMPLETED

sudo reboot