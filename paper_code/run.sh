#!/bin/bash

SAVEDIR=$1
EXP_NAME=$2

if [ "$4" == "view" ]; then
  echo "Viewing..."
  CUDA_LAUNCH_BLOCKING=1 python gui.py -s "$SAVEDIR/" \
    --expname "$SAVEDIR/$EXP_NAME" \
    --configs arguments/baseline.py \
    --start_checkpoint "$5" --view-test
else
  echo "Training starting..."
  CUDA_LAUNCH_BLOCKING=1 python gui.py -s "$SAVEDIR/" \
    --expname "$SAVEDIR/$EXP_NAME" \
    --configs arguments/baseline.py \
    --test_iterations 2000 \
    --subset $3
fi
