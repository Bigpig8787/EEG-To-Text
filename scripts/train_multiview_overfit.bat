@echo off
REM Overfit sanity check — single fixed batch, 100 epochs.
REM Goal: verify the multiview encoder + BART pipeline can drive loss to ~0
REM       on one batch. If loss does not converge, model wiring is broken.
python train_multiview_overfit.py ^
    -t task1_task2_taskNRv2 ^
    -ne 100 ^
    -lr 0.00005 ^
    -b 4 ^
    -s ./checkpoints/overfit ^
    -cuda cuda:0
