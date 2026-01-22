#!/bin/bash

SAVEDIR=$1
EXP_NAME=$2

# Try to determine base path automatically
if [ -d "/DATA/$SAVEDIR" ]; then
  BASEDIR="/DATA"
elif [ -d "/data/$SAVEDIR" ]; then
  BASEDIR="/data"
else
  echo "Error: Could not find data directory for $SAVEDIR"
  exit 1
fi

if [ "$4" == "view" ]; then
  echo "Viewing..."
  CUDA_LAUNCH_BLOCKING=1 python gui.py -s "$BASEDIR/$SAVEDIR/" \
    --expname "$SAVEDIR/$EXP_NAME" \
    --configs arguments/baseline.py \
    --subset $3 \
    --start_checkpoint "$5" --view-test
else
  echo "Training starting..."
  CUDA_LAUNCH_BLOCKING=1 python gui.py -s "$BASEDIR/$SAVEDIR/" \
    --expname "$SAVEDIR/$EXP_NAME" \
    --configs arguments/baseline.py \
    --test_iterations 2000 \
    --subset $3 \
    --test-frames 1
fi
