#!/bin/bash
#PBS -l select=1:ncpus=200:mem=800gb
#PBS -l walltime=00:20:0
#PBS -N zeopp_fe 

# Move to directory that called qsub command
cd $PBS_O_WORKDIR

module load tools/prod
module load Python/3.10.8-GCCcore-12.2.0

# activate fe_env (has pandas installed)
source /rds/general/user/jkd19/home/working_dir/envs/fe_env/bin/activate
 
# Run script
python3 /rds/general/user/jkd19/home/working_dir/feature_extraction/zeopp/zeopp_mp.py
#python3 /rds/general/user/jkd19/home/working_dir/feature_extraction/zeopp/zeopp_no-mp.py
