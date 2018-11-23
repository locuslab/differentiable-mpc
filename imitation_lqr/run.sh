#!/bin/bash

export OMP_NUM_THREADS=1

for SEED in {0..7}; do
    ./train.py --n_state=3 --n_ctrl=3 --T=5 --seed=$SEED --no-cuda \
        &> /dev/null &
done

wait
