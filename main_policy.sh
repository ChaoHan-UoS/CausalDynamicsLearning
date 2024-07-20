#!/bin/bash  
#SBATCH --job-name=TEST
#SBATCH --partition=gpu  
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
cd /users/ac1xch/CDL-DVAE/cdl  
  
echo "Task"  
python main_policy.py 
echo "Done"
