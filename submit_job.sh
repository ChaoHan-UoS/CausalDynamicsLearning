#!/bin/bash

# Directory where code is stored
CODE_DIR="/users/ac1xch/CDL-DVAE"
# Create a timestamped or unique directory for the snapshot (at submission time)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_DIR="/users/ac1xch/CDL-DVAE_Snapshot_$TIMESTAMP"

# Copy the current state of the code to the snapshot directory
mkdir -p $SNAPSHOT_DIR
cp -r $CODE_DIR/* $SNAPSHOT_DIR

# Submit the SLURM job, passing the snapshot directory as an argument
sbatch --export=SNAPSHOT_DIR=$SNAPSHOT_DIR main_policy.sh
