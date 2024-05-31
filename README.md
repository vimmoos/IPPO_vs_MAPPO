# Deep Reinforcement Learning Part 2

This project provides a simple framework for IPPO and MAPPO experiments on the Waterworld environment from pettingzoo. It leverages the Tianshou library and integrates with Weights & Biases (WandB) for experiment tracking and visualization.

Folder Structure:
+ `agents/`: This folder contains the core components for defining and training agents:
  + `config.py`: Configuration classes for PPO and MAPPO algorithms.
  + `default_conf.py`: Default configuration settings.
  + `experiments.py`: Experiment configurations and setup.
  + `hooks.py`: Custom training hooks for saving checkpoints and logging.
  + `__main__.py`: Main script for running experiments from the command line.
  + `make.py`: Functions for creating PPO and MAPPO policies.
  + `run_exp.py`: Main function to run a single experiment.
  + `mappo/`: This subfolder contains the implementation of the MAPPO algorithm:
+ `sb3/`: Integration with Stable Baselines3 (SB3) library:
  + `adaptor.py`: Adapter to make the Waterworld environment compatible with SB3.
  + `wandb_run.py`: Functions for integrating WandB with SB3 experiments.
+ `wenv/`: Adaptation for the Waterworld environment:
  + `config.py`: Configuration for the Waterworld environment.
  + `default_envs.py`: Default environment configurations.
  + `registries.py`: Registry for managing different types of agents in the environment.

# Installation
Please create a new virtualenv preferably with Python>=3.10 and install the packages with:
```
pip install -r req.txt
```

# Experiments
To run the experiments first look at the available command line arguments with:
```
python -m dww.agents --help
```
Then to run a simple test run you can use the following command:
```
python -m dww.agents dww.agents.experiments:conf_test
```

To disable wandb login add the flag `--no-wlogger`
To change algorithm use the flag `--algorithm`. The available one are `ippo` and `mappo`
The available experiments are:
+ dww.agents.experiments:confs exploratory config
+ dww.agents.experiments:confs2 exploratory config
+ dww.agents.experiments:conf_final Final config used in the paper
+ dww.agents.experiments:conf_test testing

The first two experiments were exploratory to see how the agents performed in different situations. These exploratory results were used to decide the final parameter configurations.
