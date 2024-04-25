#!/bin/bash
#PBS -l walltime=15:00:00
#PBS -lselect=1:ncpus=8:mem=99gb
 
setenv OMP_NUM_THREADS=1
setenv OPENBLAS_NUM_THREADS=1
setenv MKL_NUM_THREADS=1
#OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1


ml purge
ml anaconda3/personal

source activate mofid_a

#cd $PBS_O_WORKDIR
 
python3 