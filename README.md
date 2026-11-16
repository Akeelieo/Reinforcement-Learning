# Reinforcement Learning: From Tables to Neural Networks

Reinforcement learning (RL) is the branch of machine learning where an **agent** learns good behaviour by trial and error, guided only by rewards. There's no dataset of correct answers: the agent tries things, sees what pays off, and gradually works out a good strategy.

This repository has two connected parts. The first uses **tabular** methods, which store a value for every state in a table. The second moves to **deep RL**, where the table is replaced by a neural network so the agent can handle problems with too many states for a table.

## Part 1: Tabular RL on a GridWorld → [`tabular/`](tabular/README.md)

An agent learns to cross a slippery 5×5 grid to reach a goal while avoiding a trap. Three methods are compared:

- **Value iteration:** computes the optimal policy from a full model of the world (the baseline).
- **Q-learning (off-policy):** learns from experience alone and finds the bold, optimal route.
- **SARSA (on-policy):** learns from experience while accounting for its own exploration, so it plays more cautiously.

Q-learning matched the optimal policy in 19 of 23 states, compared with 16 for SARSA.

## Part 2: Deep RL on CartPole → [`deep_rl/`](deep_rl/README.md)

An agent learns to balance a pole on a moving cart. The state is continuous, so a table is no longer possible:

- **DQN:** Q-learning with a neural network, experience replay and a target network.
- **REINFORCE with baseline:** a policy-gradient method that learns the policy directly.

Over 5 seeds, REINFORCE reached a mean return of 474.7 (solved = 475) against 355.7 for DQN. An ablation showed that removing DQN's target network cut its return to 122.8.

## Repository structure

```
├── tabular/     GridWorld environment, value iteration, Q-learning, SARSA
├── deep_rl/     DQN and REINFORCE on CartPole-v1, plus the ablation study
└── requirements.txt
```

## Getting started

```bash
pip install -r requirements.txt
```

Then open the notebook in either folder. Each folder's README explains how to run it.

