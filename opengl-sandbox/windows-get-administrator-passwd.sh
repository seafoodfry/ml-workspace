#!/bin/bash

INSTANCE_ID=$1

export KEY="op://Private/SSH Key - numerical recipies/private key"

# Function to securely delete the file
secure_delete() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        dd if=/dev/urandom of="$1" bs=1m count=1 2>/dev/null
        rm -P "$1"
        echo "Deleted: $1"
    else
        # Linux and other Unix-like systems
        if command -v shred &> /dev/null; then
            shred -u "$1"
            echo "Deleted: $1"
        else
            dd if=/dev/urandom of="$1" bs=1M count=1 2>/dev/null
            rm -f "$1"
            echo "Deleted: $1"
        fi
    fi
}


# For some reason, when looking at the private key in 1pass it will show it to you as being in OpenSSH format.
# But when you just look at the contents of the key through the SSH_KEY variable, then you get the plain ol'
# private key format header.
# And when you actually dump the key into a file after reading it, it magically becomes an RSA formatted private key.
# So really all we have to do is dump it into a file.
SSH_KEY=$(op read "${KEY}")


# macos doesn't have /dev/shm, instead you have to create a RAM disk which is a whole thing.
TEMP_KEY_FILE=$(mktemp)
# Ensure the file is removed on script exit
trap 'secure_delete "$TEMP_KEY_FILE"' EXIT

# Write the key to the temporary file
echo "$SSH_KEY" > "$TEMP_KEY_FILE"

# Use the key with AWS CLI
./run-cmd-in-shell.sh aws ec2 get-password-data --instance-id "$INSTANCE_ID" --priv-launch-key "$TEMP_KEY_FILE"
