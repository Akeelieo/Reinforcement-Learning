# Deep RL on CartPole: DQN vs REINFORCE

The tabular methods in the main folder store one value per square, which only works when there are few states. CartPole has a continuous state (cart position and velocity, pole angle and angular velocity), so a table is no longer possible. This part uses **neural networks** to approximate the values or the policy instead.

## The task

A pole is balanced on a cart that moves left and right. Each step the agent pushes the cart left or right and gets +1 reward while the pole stays upright. An episode ends when the pole falls, the cart leaves the track, or 500 steps pass, so the maximum return is 500. The task counts as **solved** at an average return of 475.

## The two methods

- **DQN (Deep Q-Network)** is a value-based method: Q-learning with a neural network in place of the Q-table. It uses an **experience replay buffer**, training on random past transitions rather than only the latest one, and a **target network**, a frozen copy of the Q-network updated every 100 steps that provides stable learning targets. It explores with ε-greedy, with ε decaying from 1.0 to 0.01.
- **REINFORCE with baseline** is a policy-based method: a network outputs action probabilities directly and is pushed towards actions that led to higher-than-expected returns. A second network estimates the state value V(s), which is used as a baseline to reduce variance.

Both use the same network shape (2 hidden layers of 128 units, ReLU, Adam, learning rate 0.001) and γ = 0.99.

## Results

5 seeds, 1,000 episodes each, ±1 standard deviation shading, 20-episode moving average.

![DQN vs REINFORCE](figures/dqn_vs_reinforce.png)

| Method | Final mean return |
|---|---:|
| REINFORCE (with baseline) | 474.7 |
| DQN (with target network) | 355.7 |

**REINFORCE learned much faster**, passing 400 by about episode 300 and finishing just below the solved threshold, but its results varied more between seeds. **DQN learned more slowly but more consistently.** It has to fill its replay buffer and decay ε before learning picks up, and its curve was still rising at 1,000 episodes.

### Ablation: removing DQN's target network

![DQN target network ablation](figures/dqn_target_network_ablation.png)

| Variant | Final mean return |
|---|---:|
| DQN with target network | 355.7 |
| DQN without target network | 122.8 |

Without the target network, the learning target `r + γ max Q(s′, a′)` is computed from the same network that is being updated, so it shifts after every gradient step. This is an example of the **deadly triad**: off-policy learning, bootstrapping and function approximation combined can be unstable. Freezing the target for 100 steps gives the network a fixed target to learn towards.

## Files

| File | What it does |
|---|---|
| `dqn_starter.py` | DQN agent, replay buffer, training loop |
| `policy_gradient_starter.py` | REINFORCE (with/without baseline) and one-step A2C |
| `deep_rl_compare.py` | Command-line script comparing the methods |
| `cartpole_figures.ipynb` | Runs the experiments, caches results and makes the figures above |

The three `.py` files are starter code supplied by the ST962 *Introduction to Deep Reinforcement Learning* module (University of Warwick), used unchanged. The notebook, experiment design, ablation and analysis are my own.

## Running it

```bash
pip install -r ../requirements.txt
jupyter notebook cartpole_figures.ipynb
```

Training takes a while on a CPU (3 configurations × 5 seeds × 1,000 episodes). Results are saved to `results/cartpole_returns.npz`, so re-running the notebook afterwards only redraws the plots. Exact numbers can differ slightly between machines and library versions.

Command-line alternative:

```bash
python deep_rl_compare.py --dqn-ablation
```
