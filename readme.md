# Quadrotor Motion Control Using Deep Reinforcement Learning

This repo contains code, Pytorch model and training data log of paper "Quadrotor Motion Control Using Deep Reinforcement Learning" submitted to Journal of Unmanned Vehicle Systems.

## Installation

Create a python virtual environment and then install the required package. Run the following commands in the terminal.
```bash
python3 -m venv uavppo
source uavppo/bin/activate
pip install -r requirements.txt
```

## Activate Training process

```bash
python learn.py --exp_name='Quad0' --seed=1
```

## The training log and data
Training has been conducted by the author. The results of training can be found in the folder ./data.

## Acknowledgement
The quadrotor simulator is adopted from [RLSchool](https://github.com/PaddlePaddle/RLSchool). The code in this project is inspired by [OpenAI Spinning Up in Deep RL](https://github.com/openai/spinningup).