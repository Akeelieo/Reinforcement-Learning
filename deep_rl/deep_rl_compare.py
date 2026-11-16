
"""Compare the supplied deep-RL methods under the same episode budget."""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import numpy as np

from dqn_starter import run_multiple_seeds as run_dqn_multiple_seeds
from policy_gradient_starter import (
    run_multiple_seeds as run_pg_multiple_seeds,
    train_a2c,
    train_reinforce,
)


def plot_results(returns_dict, title: str = "Deep RL comparison", window: int = 20):
    """Plot one or more smoothed learning curves with mean ± std shading."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (label, returns) in enumerate(returns_dict.items()):
        returns = np.asarray(returns, dtype=float)
        if returns.ndim == 1:
            returns = returns[np.newaxis, :]

        mean = returns.mean(axis=0)
        std = returns.std(axis=0)

        if window > 1 and len(mean) >= window:
            kernel = np.ones(window) / window
            mean = np.convolve(mean, kernel, mode="valid")
            std = np.convolve(std, kernel, mode="valid")
            x = np.arange(len(mean))
        else:
            x = np.arange(len(mean))

        color = colors[i % len(colors)]
        ax.plot(x, mean, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--policy-algo", choices=["reinforce", "a2c"], default="reinforce")
    parser.add_argument("--dqn-ablation", action="store_true")
    parser.add_argument("--reinforce-ablation", action="store_true")
    args = parser.parse_args()

    results = {}

    print(f"Running DQN on {args.env}")
    dqn_returns = run_dqn_multiple_seeds(
        args.env,
        n_seeds=args.seeds,
        n_episodes=args.episodes,
        use_target_network=True,
    )
    results["DQN"] = dqn_returns

    if args.dqn_ablation:
        print("\nAblation: DQN without target network")
        dqn_no_target = run_dqn_multiple_seeds(
            args.env,
            n_seeds=args.seeds,
            n_episodes=args.episodes,
            use_target_network=False,
        )
        results["DQN (no target network)"] = dqn_no_target

    if args.policy_algo == "reinforce":
        print(f"\nRunning REINFORCE with baseline on {args.env}")
        reinforce_returns = run_pg_multiple_seeds(
            train_reinforce,
            n_seeds=args.seeds,
            env_name=args.env,
            n_episodes=args.episodes,
            use_baseline=True,
        )
        results["REINFORCE (with baseline)"] = reinforce_returns

        if args.reinforce_ablation:
            print("\nAblation: REINFORCE without baseline")
            reinforce_no_base = run_pg_multiple_seeds(
                train_reinforce,
                n_seeds=args.seeds,
                env_name=args.env,
                n_episodes=args.episodes,
                use_baseline=False,
            )
            results["REINFORCE (no baseline)"] = reinforce_no_base

    else:
        print(f"\nRunning A2C on {args.env}")
        a2c_returns = run_pg_multiple_seeds(
            train_a2c,
            n_seeds=args.seeds,
            env_name=args.env,
            n_episodes=args.episodes,
        )
        results["A2C"] = a2c_returns

    fig, _ = plot_results(results, title=f"Deep RL comparison - {args.env}")
    plt.savefig("deep_rl_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Saved: deep_rl_comparison.png")
