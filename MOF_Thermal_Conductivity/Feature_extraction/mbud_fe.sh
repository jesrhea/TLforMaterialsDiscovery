#!/bin/bash
#PBS -l select=1:ncpus=2:mem=5gb
#PBS -l walltime=00:20:00
#PBS -N mbud_fe 

# Move to directory that called qsub command
cd $PBS_O_WORKDIR

module load tools/prod
module load Python/3.10.8-GCCcore-12.2.0

# activate fe_env (has pandas installed)
source /rds/general/user/jkd19/home/working_dir/envs/fe_env/bin/activate
 
# Run script
python3 /rds/general/user/jkd19/home/working_dir/feature_extraction/mbud/mbud_funcs.py
