#!/bin/bash
# Job name:
#SBATCH --job-name=kagome-2rings-PD
# Partition:
#SBATCH --partition=shared-cpu
#
# Processors:
#SBATCH --array=1-63
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Wall clock limit:
#SBATCH --time=10:00:00
#
# Memory:
#SBATCH --mem=2000
#
#
# Output:
#SBATCH --output=kagome-test_%A.out
#SBATCH --error=kagome-test_%A.err
#
# Email at end:
#SBATCH --mail-user=johannes.motruk@unige.ch
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

echo "Starting job kagome test, task number " $SLURM_ARRAY_TASK_ID " of " $SLURM_ARRAY_TASK_COUNT

srun kagome.qsub $SLURM_ARRAY_TASK_COUNT $SLURM_ARRAY_TASK_ID

echo "Task number " $SLURM_ARRAY_TASK_ID " complete"

