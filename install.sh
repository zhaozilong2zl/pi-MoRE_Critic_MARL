#!/bin/bash

set -e

echo '>>> [1/9] Installing modified local dependency: vmas'
pip install -e ./dependencies/VectorizedMultiAgentSimulator-inactive_logic

echo '>>> [2/9] Installing modified local dependency: benchmarl'
pip install -e ./dependencies/BenchMARL-DS_MoE

echo '>>> [3/9] Installing torchrl 0.7.1'
pip install torchrl==0.7.1

echo '>>> [4/9] Installing wandb 0.20.0'
pip install wandb==0.20.0

echo '>>> [5/9] Installing tensordict 0.7.1'
pip install tensordict==0.7.1

echo '>>> [6/9] Installing numpy 1.26.4'
pip install numpy==1.26.4

echo '>>> [7/9] Installing matplotlib 3.3.4'
pip install matplotlib==3.3.4

echo '>>> [8/9] Installing seaborn 0.9.0'
pip install seaborn==0.9.0

echo '>>> [9/9] Installing pandas 1.4.4'
pip install pandas==1.4.4
