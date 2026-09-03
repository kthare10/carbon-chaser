#!/bin/bash

# this script should run as root and has been tested with ubuntu 20.04
export DEBIAN_FRONTEND=noninteractive

apt-get update && apt-get -y upgrade
apt-get install -y linux-headers-$(uname -r)
sudo apt install -y python3-pip
sudo python3 -m pip install --break-system-packages paramiko pandas sage-data-client

apt-get install -y build-essential make zlib1g-dev librrd-dev libpcap-dev autoconf automake libarchive-dev iperf3 htop bmon vim wget pkg-config git python-dev python3-pip libtool
pip install --upgrade pip

######################
### INSTALL CONDOR ###
######################

# Remove default single-machine config that conflicts with multi-node pool
rm -f /etc/condor/config.d/00-minicondor
rm -f /etc/condor/config.d/00-security

cat << EOF > /etc/condor/config.d/50-main.config
DAEMON_LIST = MASTER, STARTD

CONDOR_HOST = $3

USE_SHARED_PORT = TRUE

NETWORK_INTERFACE = $1

ENABLE_IPV6 = FALSE

# the nodes have shared filesystem
UID_DOMAIN = \$(CONDOR_HOST)
TRUST_UID_DOMAIN = TRUE
FILESYSTEM_DOMAIN = \$(FULL_HOSTNAME)

#--     Authentication settings
SEC_PASSWORD_FILE = /etc/condor/pool_password
SEC_DEFAULT_AUTHENTICATION = REQUIRED
SEC_DEFAULT_AUTHENTICATION_METHODS = FS,PASSWORD
SEC_READ_AUTHENTICATION = OPTIONAL
SEC_CLIENT_AUTHENTICATION = OPTIONAL
SEC_ENABLE_MATCH_PASSWORD_AUTHENTICATION = TRUE
DENY_WRITE = anonymous@*
DENY_ADMINISTRATOR = anonymous@*
DENY_DAEMON = anonymous@*
DENY_NEGOTIATOR = anonymous@*
DENY_CLIENT = anonymous@*

#--     Privacy settings
SEC_DEFAULT_ENCRYPTION = OPTIONAL
SEC_DEFAULT_INTEGRITY = REQUIRED
SEC_READ_INTEGRITY = OPTIONAL
SEC_CLIENT_INTEGRITY = OPTIONAL
SEC_READ_ENCRYPTION = OPTIONAL
SEC_CLIENT_ENCRYPTION = OPTIONAL

#-- With strong security, do not use IP based controls
ALLOW_WRITE = *
ALLOW_NEGOTIATOR = *

# dynamic slots
SLOT_TYPE_1 = cpus=100%,disk=100%,swap=100%
SLOT_TYPE_1_PARTITIONABLE = TRUE
NUM_SLOTS = 1
NUM_SLOTS_TYPE_1 = 1

#-- Same as fabric-worker-gpu.sh: claim reuse is a startd-side decision.
# Without this, consecutive DAG jobs ride one claim and the job's RANK is
# never re-evaluated after the first match.
CLAIM_WORKLIFE = 0

#-- The advertised carbon value must be able to KEEP UP with the replay.
# The STARTD_CRON refreshes CarbonIntensity every 60s, but the startd only
# pushes its ad to the collector every UPDATE_INTERVAL (default 300s), and
# the negotiator ranks on the collector's copy. Measured on a live pool:
# ads were 208-268s old -- at CARBON_ACCEL=300 that is 17-22 REPLAYED
# HOURS, more than a diurnal cycle, and the per-site skew (60s = 5
# replayed hours) means sites get compared at different replayed instants.
# That is the same class of incoherence the shared replay clock was built
# to eliminate, reappearing via ad propagation instead of clock
# derivation. Two placements were observed landing on a site whose
# advertised value exceeded another site's trace-wide maximum.
UPDATE_INTERVAL = 60

EOF

condor_store_cred -f /etc/condor/pool_password -p p3g@sus

systemctl enable condor
systemctl restart condor

##########################
### INSTALL SINGULARITY ##
##########################

apt-get update
apt-get install -y software-properties-common
add-apt-repository -y ppa:apptainer/ppa
apt-get update
apt-get install -y apptainer

##########################
### INSTALL DOCKER      ##
##########################
cd
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io

#groupadd docker
usermod -aG docker condor

sudo tee /etc/docker/daemon.json > /dev/null <<'DEOF'
{
  "ipv6": true,
  "fixed-cidr-v6": "2001:db8:1::/64"
}
DEOF

sudo systemctl daemon-reload || true
sudo systemctl restart docker
sudo systemctl enable docker || true

apt-get install -y docker-compose-plugin

############################
### SETUP DEFAULT USER ####
############################
cd
usermod -a -G docker ubuntu

echo 'export GOPATH=${HOME}/go' >> /home/ubuntu/.bashrc
echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> /home/ubuntu/.bashrc

