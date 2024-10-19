#!/bin/bash
#SBATCH --job-name=no2_1fap
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3-23:59:00
#SBATCH --output=/users/ac1xch/CDL-DVAE_%j/out/output_%j.txt

module load Anaconda3/2022.05
module load cuDNN/8.8.0.121-CUDA-12.0.0

source activate tianshou
echo "Begin"

pwd
nvidia-smi

# Create a job-specific directory for the SLURM job output (at execution time)
JOB_DIR="/users/ac1xch/CDL-DVAE_${SLURM_JOB_ID}"
mkdir -p $JOB_DIR

# Copy the pre-submission snapshot into the job-specific directory
cp -r $SNAPSHOT_DIR/* $JOB_DIR

# Delete the snapshot directory to save space after copying
rm -rf $SNAPSHOT_DIR

# Log GPU usage in the background
mkdir -p "${JOB_DIR}/gpu"
GPU_LOG_FILE="gpu_usage_${SLURM_JOB_ID}.log"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw --format=csv,nounits -l 1000 > ${JOB_DIR}/gpu/$GPU_LOG_FILE & NVIDIA_SMI_PID=$!

# Run the code from the job-specific directory
cd "${JOB_DIR}/cdl"
pwd
echo "Task"
python main_policy.py
echo "Done"

# Stop logging GPU usage
kill $NVIDIA_SMI_PID
