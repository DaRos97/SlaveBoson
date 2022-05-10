#!/bin/bash
# Job name:
#SBATCH --job-name=p_24_1
# Partition:
#SBATCH --partition=shared-cpu
#
# Processors:
#SBATCH --array=0-80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Wall clock limit:
#SBATCH --time=6:00:00
#
# Memory:
# #SBATCH --mem=4000
#
#
# Output:
#SBATCH --output=p_24_1.out
#SBATCH --error=p_24_1.err
#
# Email at end:
#SBATCH --mail-user=dario.rossi@unige.ch
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

srun qp24_1.qsub $SLURM_ARRAY_TASK_ID

