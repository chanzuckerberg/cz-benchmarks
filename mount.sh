#!/bin/bash

# Update and upgrade packages
apt-get update || yum update -y
apt-get upgrade -y || yum upgrade -y

# Install EFS and NFS utilities
# Try both apt-get and yum since we don't know the distro
apt-get install -y amazon-efs-utils || yum install -y amazon-efs-utils
apt-get install -y nfs-common || yum install -y nfs-utils

# Set EFS configuration variables
file_system_id_1="fs-0649987be9564e17f"
efs_mount_point_1="/mnt/efs/fs1"

# Create mount point directory
mkdir -p "${efs_mount_point_1}"

# Add mount configuration to fstab
if [ -f "/sbin/mount.efs" ]; then
    echo "${file_system_id_1}:/ ${efs_mount_point_1} efs tls,_netdev" >> /etc/fstab
else
    echo "${file_system_id_1}.efs.us-west-2.amazonaws.com:/ ${efs_mount_point_1} nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport,_netdev 0 0" >> /etc/fstab
fi

# Update EFS client info if needed
if [ -f "/sbin/mount.efs" ]; then
    if ! grep -ozP 'client-info]\nsource' '/etc/amazon/efs/efs-utils.conf'; then
        echo -e "\n[client-info]\nsource=liw" >> /etc/amazon/efs/efs-utils.conf
    fi
fi

# Attempt to mount with retries

mount -a -t efs,nfs4 defaults
if [ $? = 0 ] || [ $retry_cnt -lt 1 ]; then
    echo "File system mounted successfully"
    break
fi
echo "File system not available, retrying to mount."
((retry_cnt--))
sleep $wait_time