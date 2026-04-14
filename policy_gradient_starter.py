
"""Starter policy-gradient code for the long coursework."""

from __future__ import annotations

import argparse
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Maps states to action logits."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNetwork(nn.Module):
    """Maps states to scalar value estimates V(s)."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def compute_discounted_returns(rewards, gamma: float) -> torch.Tensor:
    """
    Compute the raw discounted returns G_t = sum_{k>=0} gamma^k r_{t+k+1}.
    """
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = float(reward) + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def train_reinforce(
    env_name: str,
    n_episodes: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    use_baseline: bool = True,
    seed: int = 0,
):
    """
    REINFORCE with an optional state-value baseline.

    Policy update:
        theta <- theta + alpha * sum_t (G_t - b(S_t)) grad log pi_theta(A_t | S_t)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name)
    if not hasattr(env.observation_space, "shape") or env.observation_space.shape is None:
        raise ValueError("This script expects a vector observation space.")
    if not hasattr(env.action_space, "n"):
        raise ValueError("This script expects a discrete action space.")

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_net = PolicyNetwork(state_dim, n_actions).to(device)
    policy_opt = optim.Adam(policy_net.parameters(), lr=lr)

    value_net = None
    value_opt = None
    if use_baseline:
        value_net = ValueNetwork(state_dim).to(device)
        value_opt = optim.Adam(value_net.parameters(), lr=lr)

    episode_returns = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        log_probs = []
        rewards = []
        states = []
        done = False

        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy_net(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()

            log_probs.append(dist.log_prob(action).squeeze())
            states.append(state_t)

            state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            rewards.append(float(reward))

        returns = compute_discounted_returns(rewards, gamma).to(device)
        episode_returns.append(sum(rewards))

        log_probs_t = torch.stack(log_probs)
        states_t = torch.cat(states, dim=0)

        if use_baseline:
            assert value_net is not None and value_opt is not None
            values = value_net(states_t)
            advantages = returns - values.detach()
            policy_loss = -(advantages * log_probs_t).mean()

            value_loss = nn.functional.mse_loss(values, returns)
            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()
        else:
            policy_loss = -(returns * log_probs_t).mean()

        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        if (ep + 1) % 100 == 0:
            mean_100 = np.mean(episode_returns[-100:])
            label = "with baseline" if use_baseline else "no baseline"
            print(f"[REINFORCE {label}] Ep {ep + 1:4d}  Mean(100): {mean_100:6.1f}")

    env.close()
    return np.asarray(episode_returns, dtype=float)


def train_a2c(
    env_name: str,
    n_episodes: int = 1000,
    gamma: float = 0.99,
    lr_actor: float = 1e-3,
    lr_critic: float = 1e-3,
    seed: int = 0,
):
    """
    One-step Advantage Actor-Critic.

    Actor update:
        theta <- theta + alpha * A_t grad log pi_theta(A_t | S_t)
    Critic target:
        r + gamma * V(s')
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name)
    if not hasattr(env.observation_space, "shape") or env.observation_space.shape is None:
        raise ValueError("This script expects a vector observation space.")
    if not hasattr(env.action_space, "n"):
        raise ValueError("This script expects a discrete action space.")

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    actor = PolicyNetwork(state_dim, n_actions).to(device)
    critic = ValueNetwork(state_dim).to(device)
    actor_opt = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_opt = optim.Adam(critic.parameters(), lr=lr_critic)

    episode_returns = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_return = 0.0
        done = False

        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            logits = actor(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).squeeze()

            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            total_return += float(reward)

            next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            value_s = critic(state_t).squeeze()
            with torch.no_grad():
                value_next = critic(next_state_t).squeeze()
                td_target = torch.tensor(float(reward), dtype=torch.float32, device=device)
                if not done:
                    td_target = td_target + gamma * value_next

            advantage = td_target - value_s
            actor_loss = -log_prob * advantage.detach()
            critic_loss = nn.functional.mse_loss(value_s, td_target.detach())

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            state = next_state

        episode_returns.append(total_return)

        if (ep + 1) % 100 == 0:
            mean_100 = np.mean(episode_returns[-100:])
            print(f"[A2C] Ep {ep + 1:4d}  Mean(100): {mean_100:6.1f}")

    env.close()
    return np.asarray(episode_returns, dtype=float)


def plot_results(returns_dict, title: str = "Policy Gradient Training", window: int = 20):
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


def run_multiple_seeds(train_fn, n_seeds: int = 5, **kwargs):
    """Run a training function over several seeds and stack the returns."""
    all_returns = []
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        returns = train_fn(seed=seed, **kwargs)
        all_returns.append(returns)
    return np.asarray(all_returns, dtype=float)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--algo", type=str, default="reinforce", choices=["reinforce", "a2c"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="For REINFORCE only: compare with versus without the state-value baseline.",
    )
    args = parser.parse_args()

    results = {}

    if args.algo == "reinforce":
        print(f"Training REINFORCE (with baseline) on {args.env}")
        returns_with = run_multiple_seeds(
            train_reinforce,
            n_seeds=args.seeds,
            env_name=args.env,
            n_episodes=args.episodes,
            use_baseline=True,
        )
        results["REINFORCE (with baseline)"] = returns_with

        if args.ablation:
            print(f"\nAblation: REINFORCE without baseline on {args.env}")
            returns_without = run_multiple_seeds(
                train_reinforce,
                n_seeds=args.seeds,
                env_name=args.env,
                n_episodes=args.episodes,
                use_baseline=False,
            )
            results["REINFORCE (no baseline)"] = returns_without

    else:
        print(f"Training A2C on {args.env}")
        returns_a2c = run_multiple_seeds(
            train_a2c,
            n_seeds=args.seeds,
            env_name=args.env,
            n_episodes=args.episodes,
        )
        results["A2C"] = returns_a2c

    fig, _ = plot_results(results, title=f"{args.algo.upper()} - {args.env}")
    plt.savefig("policy_gradient_results.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Saved: policy_gradient_results.png")
