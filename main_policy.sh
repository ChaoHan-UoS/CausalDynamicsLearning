#!/bin/bash  
#SBATCH --job-name=Shared_forward
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu  
#SBATCH --gres=gpu:1  
#SBATCH --mem=64G  
#SBATCH --time=3-23:59:00  
#SBATCH --output=out/output%j.txt  
  
module load Anaconda3/2022.05
module load cuDNN/8.8.0.121-CUDA-12.0.0
  
source activate tianshou  
echo "Begin"  
  
pwd
nvidia-smi

mkdir -p gpu
GPU_LOG_FILE="gpu_usage_${SLURM_JOB_ID}.log"
# Start logging GPU usage in the background
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw --format=csv,nounits -l 1000 > gpu/$GPU_LOG_FILE & NVIDIA_SMI_PID=$!
cd /users/ac1xch/CDL-DVAE/cdl  
  
echo "Task"  
python main_policy.py 
echo "Done"

# Stop logging GPU usage
kill $NVIDIA_SMI_PID
