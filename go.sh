#! /bin/bash

export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
python hub2d.ti.py 4.0 . 0.9 
