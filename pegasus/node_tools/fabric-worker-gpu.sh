#!/bin/bash

# GPU worker setup script - should run as root
# Based on fabric-worker.sh with NVIDIA driver/CUDA and HTCondor GPU support added
export DEBIAN_FRONTEND=noninteractive

apt-get update && apt-get -y upgrade
apt-get install -y linux-headers-$(uname -r)
sudo apt install -y python3-pip
sudo python3 -m pip install --break-system-packages paramiko pandas sage-data-client

apt-get install -y build-essential make zlib1g-dev librrd-dev libpcap-dev autoconf automake libarchive-dev iperf3 htop bmon vim wget pkg-config git python-dev python3-pip libtool
pip install --upgrade pip

##############################################################
### NVIDIA DRIVERS: intentionally NOT installed here.
###
### The original used `ubuntu-drivers autoinstall` + the CUDA repo's
### `cuda` metapackage, which pull the NEWEST driver. That driver
### frequently has no prebuilt module for the running kernel, giving
### "Failed to initialize NVML: Driver/library version mismatch" — and
### DKMS makes it worse, because apt often installs a newer kernel in the
### same transaction so the reboot lands on a kernel with no module at all.
###
### Drivers are installed by fabric/install_gpu_stack.py, which picks the
### highest driver version that HAS a prebuilt module for the running
### kernel and is idempotent (re-running on a healthy node is a no-op).
### Measured NVML power is mandatory here, so a silent driver failure is
### not acceptable.
##############################################################

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

# dynamic slots with GPU resources
SLOT_TYPE_1 = cpus=100%,disk=100%,swap=100%,gpus=100%
SLOT_TYPE_1_PARTITIONABLE = TRUE
NUM_SLOTS = 1
NUM_SLOTS_TYPE_1 = 1

#-- Carbon-aware placement REQUIRES re-negotiating every job, and the knob
# that controls claim reuse lives HERE, in the startd's config -- NOT on the
# submit node, where an earlier fix put it (harmlessly configuring the
# submit node's own startd, which never runs these jobs). With the default
# CLAIM_WORKLIFE (1200s) the schedd runs consecutive DAG jobs on the claim
# it already holds -- SchedLog: "match (slot1@X) switching to job N" --
# without going back to the negotiator, so RANK = -CarbonIntensity is
# evaluated once for the FIRST segment and every later segment inherits
# that placement: no migration, no transfers, ever.
# 0 = give the claim back after each job, so every segment is a fresh
# placement decision against the sites' CURRENT carbon intensities.
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

# GPU discovery
use feature : GPUs

# Advertise GPU attributes
STARTD_ATTRS = \$(STARTD_ATTRS) HasGPU
# DetectedGPUs is a STRING of GPU ids ("GPU-1cae1040"), not a count, so
# `DetectedGPUs > 0` evaluates to ERROR — and an error in a requirements
# expression means NEVER MATCH, silently: the job just sits Idle forever.
# GPUs is the slot's integer GPU count.
HasGPU = ifThenElse(GPUs =!= undefined && GPUs > 0, true, false)

# Carbon-aware matchmaking. A cron job rewrites 99-carbon.config with the
# site's current grid intensity and measured GPU power, so a job's RANK
# expression can prefer clean sites WITHOUT replanning the workflow.
# Defaults are deliberately absent rather than zero: an unset attribute is
# visibly unknown, whereas 0 gCO2/kWh would read as the cleanest grid.
STARTD_ATTRS = \$(STARTD_ATTRS) FabricSite CarbonIntensity CarbonTimestamp GPUWatts
FabricSite = "$4"
STARTD_CRON_JOBLIST = \$(STARTD_CRON_JOBLIST) CARBON
STARTD_CRON_CARBON_EXECUTABLE = /usr/local/bin/carbon_classad.py
STARTD_CRON_CARBON_PERIOD = 60
STARTD_CRON_CARBON_MODE = periodic
STARTD_CRON_CARBON_RECONFIG = false
STARTD_CRON_CARBON_KILL = true

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
echo 'export PATH=/usr/local/cuda/bin:${PATH}' >> /home/ubuntu/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' >> /home/ubuntu/.bashrc
