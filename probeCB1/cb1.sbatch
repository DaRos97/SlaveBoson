#!/bin/bash
# Job name:
#SBATCH --job-name=cb1
# Partition:
#SBATCH --partition=shared-cpu
#
# Processors:
#SBATCH --array=54-55,63-66,72-79
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Wall clock limit:
#SBATCH --time=12:00:00
#
# Memory:
# #SBATCH --mem=2000
#
#
# Output:
#SBATCH --output=SB.out
#SBATCH --error=SB.err
#
# Email at end:
# #SBATCH --mail-user=dario.rossi@unige.ch
# #SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

srun qcb1.qsub $SLURM_ARRAY_TASK_ID

