#!/bin/bash

# this script should run as root and has been tested with ubuntu 20.04
export DEBIAN_FRONTEND=noninteractive

apt-get update && apt-get -y upgrade
apt-get install -y linux-headers-$(uname -r)
sudo apt install -y python3-pip
sudo python3 -m pip install --break-system-packages paramiko pandas sage-data-client
apt-get install -y build-essential make zlib1g-dev librrd-dev libpcap-dev autoconf automake libarchive-dev iperf3 htop bmon vim wget pkg-config git python-dev python3-pip libtool

######################
### INSTALL CONDOR ###
######################


# Remove default single-machine config that conflicts with multi-node pool
rm -f /etc/condor/config.d/00-minicondor
rm -f /etc/condor/config.d/00-security

cat << EOF > /etc/condor/config.d/50-main.config
DAEMON_LIST = MASTER, COLLECTOR, NEGOTIATOR, SCHEDD, STARTD

#-- Carbon-aware placement REQUIRES re-negotiating every job.
# By default the schedd keeps a claim on a machine for CLAIM_WORKLIFE seconds
# (1200) and runs subsequent jobs on it directly -- the SchedLog says
# "match (slot1@X) switching to job N" -- WITHOUT going back to the
# negotiator. A whole workflow then fits inside one claim, so RANK is
# evaluated once for the first job and every later segment inherits that
# placement. Observed exactly this: 4 segments, 1 negotiation, all on the
# site that happened to be cleanest at submit time.
#
# 0 = relinquish the claim after each job, so every segment is a real
# placement decision against current CarbonIntensity. The cost is waiting for
# a negotiation cycle between segments, hence the shorter interval below.
# NOTE: CLAIM_WORKLIFE is a STARTD knob -- the copy that actually matters is
# in fabric-worker-gpu.sh / fabric-worker.sh, on the machines whose slots run
# the jobs. It is kept here too only so the submit node's own startd (if it
# ever advertises slots) behaves consistently; setting it here alone was
# observed to fix NOTHING, because this file never configures the workers.
CLAIM_WORKLIFE = 0
NEGOTIATOR_INTERVAL = 20

#-- The job's RANK must actually decide the match.
# HTCondor's DEFAULT NEGOTIATOR_PRE_JOB_RANK is
#   (10000000 * My.Rank) + (1000000 * (RemoteOwner =?= UNDEFINED))
#   - (100000 * Cpus) - Memory
# and it is consulted BEFORE the job's own rank. Those last two terms encode
# "prefer a smaller slot", which with partitionable slots means "prefer the
# machine that already has a dynamic slot carved out of it". Measured on this
# pool: TACC's leftover slot1 (Cpus=2, Mem=3701) scored 796,299 against CLEM's
# full slot1 (Cpus=4, Mem=15989) at 584,011 -- a 212,288-point lead, while the
# carbon difference the job cared about was worth 12.6. The first placement
# therefore created an asymmetry that pinned every later segment to the same
# site, and `rank = -CarbonIntensity` never got a vote.
#
# Keep "prefer an unclaimed slot"; drop the size bias so carbon decides.
NEGOTIATOR_PRE_JOB_RANK = 1000000 * (RemoteOwner =?= UNDEFINED)

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

#-- Allow workers to advertise to the collector
ALLOW_ADVERTISE_STARTD = condor@* condor@password
ALLOW_ADVERTISE_MASTER = condor@* condor@password
ALLOW_ADVERTISE_SCHEDD = condor@* condor@password
ALLOW_DAEMON = condor@* condor@password
ALLOW_READ = *

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
### INSTALL TSTAT       ####
############################

apt-get install -y libpcap-dev

cd
wget http://www.tstat.polito.it/download/tstat-3.1.1.tar.gz
tar -xzvf tstat-3.1.1.tar.gz
cd tstat-3.1.1
./autogen.sh
./configure --enable-libtstat --enable-zlib
make && make install

############################
### SETUP DEFAULT USER ####
############################
cd
usermod -a -G docker ubuntu

echo 'export GOPATH=${HOME}/go' >> /home/ubuntu/.bashrc
echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> /home/ubuntu/.bashrc
