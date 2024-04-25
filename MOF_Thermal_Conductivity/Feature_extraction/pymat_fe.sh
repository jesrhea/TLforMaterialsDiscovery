#!/bin/bash
#PBS -l select=1:ncpus=250:mem=700gb
#PBS -l walltime=71:0:0
#PBS -N pymat_fe 

# Move to directory that called qsub command
cd $PBS_O_WORKDIR

module load tools/prod
module load Python/3.10.8-GCCcore-12.2.0

# activate fe_env (has pandas installed)
source /rds/general/user/jkd19/home/working_dir/envs/fe_env/bin/activate
 
# Run script
python3 /rds/general/user/jkd19/home/working_dir/feature_extraction/pymat/pymat-nomp.py
