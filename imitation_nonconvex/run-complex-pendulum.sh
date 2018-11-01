#!/bin/bash

args_all_modes() {
    echo --n_epoch 250 --mode sysid --no-cuda $*
    echo --n_epoch 250 --mode empc --no-cuda --learn_dx $*
}

args_all_sizes() {
    DATA=$1
    SEED=$2
    args_all_modes --data $DATA --seed $SEED --n_train 100
}

args_all_seeds() {
    DATA=$1
    for SEED in {0..7}; do
        args_all_sizes $DATA $SEED
    done
}

run_single() {
    ./il_exp.py $* &> /dev/null
}
export -f run_single


export OMP_NUM_THREADS=1

MAX_PROCS=16
args_all_seeds ./data/pendulum-complex.pkl | parallel --no-notice --max-procs $MAX_PROCS run_single &
wait
