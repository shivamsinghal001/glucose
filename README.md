# SimGlucose Environment
This code was adapted for "Preventing Reward Hacking using Occupancy Measure Regularization". More details about how to run experiments can be found within our main repository and our paper. We have updated this environment so that the safe policy actions can be used if the user wants to generate a safe policy, and the true and proxy reward are calculated simultaneously for our evaluations.

This repository is based on the [code](https://github.com/aypan17/reward-misspecification/tree/main/glucose) of [Pan et al.](https://arxiv.org/abs/2201.03544). 

The original code release for the SimGlucose Environment (Deep Reinforcement Learning for Closed-Loop Blood Glucose Control, MLHC 2020) can be found [here](https://github.com/MLD3/RL4BG).

## Installation
Running 
```
pip install -r requirements.txt
```
from our main repository will install this package along with all of its depedencies. 
