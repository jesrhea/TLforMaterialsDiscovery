#!/bin/bash
#PBS -lwalltime=70:00:00
#PBS -lselect=1:ncpus=4:mem=200gb:ngpus=1:gpu_type=RTX6000

cd $PBS_O_WORKDIR

ml purge
ml anaconda3/personal
source activate test1

python3 run_model.py

