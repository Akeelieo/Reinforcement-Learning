
"""Starter code for the tabular RL coursework exercises."""

from __future__ import annotations

import itertools
import numpy as np
import matplotlib.pyplot as plt

from gridworld import (
    GridWorld,
    compare_policies,
    evaluate_policy,
    plot_learning_curves,
    plot_policy,
    plot_value_function,
)


def greedy_action(Q: np.ndarray, state: int, rng: np.random.Generator) -> int:
    """Choose a greedy action, breaking ties uniformly at random."""
    row = Q[state]
    best_actions = np.flatnonzero(row == row.max())
    return int(rng.choice(best_actions))


def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    """Epsilon-greedy action selection with random tie-breaking."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return greedy_action(Q, state, rng)


def extract_policy_from_values(env: GridWorld, V: np.ndarray) -> np.ndarray:
    """Extract a greedy policy from a value function."""
    q_values = env.R + env.gamma * (env.P @ V)
    policy = np.argmax(q_values, axis=1)
    return policy.astype(int)


def extract_policy_from_q(Q: np.ndarray) -> np.ndarray:
    """Greedy policy from a state-action value table."""
    return np.argmax(Q, axis=1).astype(int)


def value_iteration(env: GridWorld, theta: float = 1e-8, max_iters: int = 10_000):
    """
    Compute the optimal value function and policy by value iteration.

    Returns
    -------
    V : np.ndarray, shape (n_states,)
    policy : np.ndarray, shape (n_states,)
    n_iters : int
    """
    V = np.zeros(env.n_states, dtype=float)

    for iteration in range(1, max_iters + 1):
        q_values = env.R + env.gamma * (env.P @ V)
        V_new = q_values.max(axis=1)
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        if delta < theta:
            break

    q_values = env.R + env.gamma * (env.P @ V)
    policy = np.argmax(q_values, axis=1).astype(int)
    return V, policy, iteration


def q_learning(
    env: GridWorld,
    n_episodes: int = 5_000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    seed: int = 0,
    max_steps: int = 500,
):
    """
    Tabular Q-learning with an epsilon-greedy behaviour policy.

    Update:
        Q(s, a) <- Q(s, a) + alpha [r + gamma max_a' Q(s', a') - Q(s, a)]
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    episode_returns = []

    for ep in range(n_episodes):
        state = env.reset(seed=seed + ep)
        discounted_return = 0.0
        discount = 1.0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, done, _ = env.step(action)

            target = reward if done else reward + env.gamma * np.max(Q[next_state])
            td_error = target - Q[state, action]
            Q[state, action] += alpha * td_error

            discounted_return += discount * reward
            discount *= env.gamma
            state = next_state
            if done:
                break

        episode_returns.append(discounted_return)

    policy = extract_policy_from_q(Q)
    return Q, policy, np.asarray(episode_returns, dtype=float)


def sarsa(
    env: GridWorld,
    n_episodes: int = 5_000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    seed: int = 0,
    max_steps: int = 500,
):
    """
    On-policy tabular SARSA with epsilon-greedy control.

    Update:
        Q(s, a) <- Q(s, a) + alpha [r + gamma Q(s', a') - Q(s, a)]
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    episode_returns = []

    for ep in range(n_episodes):
        state = env.reset(seed=seed + ep)
        action = epsilon_greedy(Q, state, epsilon, rng)
        discounted_return = 0.0
        discount = 1.0

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            discounted_return += discount * reward
            discount *= env.gamma

            if done:
                target = reward
                Q[state, action] += alpha * (target - Q[state, action])
                break

            next_action = epsilon_greedy(Q, next_state, epsilon, rng)
            target = reward + env.gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (target - Q[state, action])

            state, action = next_state, next_action

        episode_returns.append(discounted_return)

    policy = extract_policy_from_q(Q)
    return Q, policy, np.asarray(episode_returns, dtype=float)


def run_hyperparameter_study(
    env_factory,
    epsilons=(0.01, 0.1, 0.3),
    alphas=(0.01, 0.1, 0.5),
    n_runs: int = 10,
    n_episodes: int = 2_000,
):
    """
    Run a grid of Q-learning experiments over (epsilon, alpha) values.

    Parameters
    ----------
    env_factory : callable
        Zero-argument function returning a fresh GridWorld instance.
    """
    results = {}
    for alpha, epsilon in itertools.product(alphas, epsilons):
        runs = []
        for seed in range(n_runs):
            env = env_factory()
            _, _, returns = q_learning(
                env,
                n_episodes=n_episodes,
                alpha=alpha,
                epsilon=epsilon,
                seed=seed,
            )
            runs.append(returns)
        results[(alpha, epsilon)] = np.asarray(runs, dtype=float)
    return results


def plot_hyperparameter_study(results, alphas=(0.01, 0.1, 0.5), epsilons=(0.01, 0.1, 0.3), window: int = 50):
    """Plot a grid of Q-learning learning curves for different hyperparameters."""
    fig, axes = plt.subplots(len(alphas), len(epsilons), figsize=(13, 10), sharex=True, sharey=True)

    for i, alpha in enumerate(alphas):
        for j, epsilon in enumerate(epsilons):
            ax = axes[i, j]
            returns = np.asarray(results[(alpha, epsilon)], dtype=float)
            mean = returns.mean(axis=0)
            std = returns.std(axis=0)

            if window > 1 and len(mean) >= window:
                kernel = np.ones(window) / window
                mean = np.convolve(mean, kernel, mode="valid")
                std = np.convolve(std, kernel, mode="valid")
                x = np.arange(len(mean))
            else:
                x = np.arange(len(mean))

            ax.plot(x, mean)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            ax.set_title(rf"$\alpha={alpha},\ \varepsilon={epsilon}$")
            ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Episode")
    for ax in axes[:, 0]:
        ax.set_ylabel("Discounted return")

    fig.suptitle("Q-learning hyperparameter study", y=0.98)
    fig.tight_layout()
    return fig, axes


def summarise_policy_performance(env: GridWorld, policy: np.ndarray, label: str, seed: int = 0):
    """Print the discounted Monte Carlo performance of a fixed policy."""
    returns = evaluate_policy(env, policy, n_episodes=1_000, seed=seed)
    print(f"{label}: mean discounted return = {returns.mean():.3f}, std = {returns.std():.3f}")
    return returns


def main():
    env = GridWorld(seed=0)
    print("GridWorld:", env.describe_layout())

    # --------------------------------------------------------------
    # Value iteration
    # --------------------------------------------------------------
    V_opt, pi_opt, n_iters = value_iteration(env)
    print(f"Value iteration converged in {n_iters} iterations.")
    summarise_policy_performance(env, pi_opt, "Optimal policy", seed=123)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_value_function(env, V_opt, title="Optimal value function", ax=axes[0])
    plot_policy(env, pi_opt, title="Optimal policy", ax=axes[1])
    fig.tight_layout()
    fig.savefig("gridworld_value_iteration.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Saved: gridworld_value_iteration.png")

    # --------------------------------------------------------------
    # Q-learning
    # --------------------------------------------------------------
    env_q = GridWorld(seed=0)
    Q_q, pi_q, returns_q = q_learning(env_q, n_episodes=5_000, alpha=0.1, epsilon=0.1, seed=0)
    compare_policies(env_q, pi_q, pi_opt, "Q-learning", "Optimal")
    summarise_policy_performance(env_q, pi_q, "Q-learning policy", seed=321)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_learning_curves({"Q-learning": returns_q}, window=50, title="Q-learning on GridWorld", ax=axes[0])
    plot_policy(env_q, pi_q, title="Q-learning policy", ax=axes[1])
    fig.tight_layout()
    fig.savefig("gridworld_q_learning.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Saved: gridworld_q_learning.png")

    # --------------------------------------------------------------
    # SARSA
    # --------------------------------------------------------------
    env_s = GridWorld(seed=0)
    Q_s, pi_s, returns_s = sarsa(env_s, n_episodes=5_000, alpha=0.1, epsilon=0.1, seed=0)
    compare_policies(env_s, pi_s, pi_opt, "SARSA", "Optimal")
    summarise_policy_performance(env_s, pi_s, "SARSA policy", seed=999)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_learning_curves({"Q-learning": returns_q, "SARSA": returns_s}, window=50, title="Q-learning vs SARSA", ax=axes[0])
    plot_policy(env_s, pi_s, title="SARSA policy", ax=axes[1])
    fig.tight_layout()
    fig.savefig("gridworld_q_vs_sarsa.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Saved: gridworld_q_vs_sarsa.png")

    # --------------------------------------------------------------
    # Hyperparameter study
    # --------------------------------------------------------------
    results = run_hyperparameter_study(lambda: GridWorld(seed=0), n_runs=10, n_episodes=2_000)
    fig, _ = plot_hyperparameter_study(results)
    fig.savefig("gridworld_hyperparameter_study.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Saved: gridworld_hyperparameter_study.png")


if __name__ == "__main__":
    main()
