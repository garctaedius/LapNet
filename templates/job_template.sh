#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -N {num_nodes}
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH -A l-aut_sch-hoef
#SBATCH --output {output_file}
#SBATCH --time 0-00:10:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4

source $HOME/venv/lapnet/bin/activate
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NVIDIA_TF32_OVERRIDE=0
export WANDB_MODE=offline
export JAX_DEFAULT_MATMUL_PRECISION=float32
#python restart_if_dead.py &
srun python /leonardo/home/userexternal/garctaed/develop/gunnar_lapnet/LapNet/main.py --config config.py
