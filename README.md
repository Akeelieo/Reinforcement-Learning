# Tabular RL on a 5×5 GridWorld

Value iteration, Q-learning, and SARSA on a stochastic GridWorld (start `(0,0)`, goal `(4,4)` reward +1, trap `(3,2)` reward −1, 10% random action slip, γ = 0.95).

## Requirements & how to run

Needs `gridworld.py` and `tabular_rl_starter.py` in the same folder as the notebook.

```bash
pip install numpy matplotlib
jupyter notebook Untitled1.ipynb   # run all cells top to bottom
```

## How the code works

- **`gridworld.py`** — the environment (`GridWorld`) plus helpers: `evaluate_policy` (Monte Carlo evaluation of a greedy policy), `compare_policies`, and the plotting functions.
- **`tabular_rl_starter.py`** — the algorithms: `value_iteration`, `q_learning`, `sarsa`, and the hyperparameter study scaffold.

The notebook: (1) builds the environment and runs value iteration to get v* and π*; (2) runs Q-learning for 5,000 episodes (α = 0.1, ε = 0.1, seed 0) and plots the smoothed learning curve; (3) runs SARSA with identical settings and compares; (4) evaluates each greedy policy over 1,000 episodes and plots all three policies side by side; (5) runs the ε × α hyperparameter grid.

## Theory

**Value iteration** repeatedly applies the Bellman optimality backup until the sup-norm change falls below θ = 10⁻⁸, then extracts the greedy policy:

$$v_{k+1}(s) = \max_a \sum_{s',r} p(s',r \mid s,a)\,\big[r + \gamma v_k(s')\big], \qquad \pi^{\ast}(s) = \arg\max_a q^{\ast}(s,a)$$

**Q-learning** (off-policy TD control) follows an ε-greedy behaviour policy but backs up the greedy action, so it learns q* directly:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big]$$

**SARSA** (on-policy TD control) backs up the action a′ actually taken by the ε-greedy policy, so it learns the best policy *under exploration* and is more cautious near the trap:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma Q(s',a') - Q(s,a)\big]$$

## Results

![Policy comparison](policy_comparison(2).png)

| Policy | Mean Return | Agreement with π* |
|---|---:|---:|
| Optimal π* | 0.6693 | — |
| Q-learning | 0.6565 | 19/23 (82.6%) |
| SARSA | 0.6508 | 16/23 (69.6%) |

Both learned policies fall slightly short of optimal because fixed ε = 0.1 means convergence to the best ε-soft policy rather than π*.
