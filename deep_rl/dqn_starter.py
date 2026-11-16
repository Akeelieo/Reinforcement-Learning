
"""Starter DQN code for the long coursework."""

from __future__ import annotations

import argparse
import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Maps states to Q-values for all actions."""

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


class DQNAgent:
    """DQN agent with experience replay and an optional target network."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        use_target_network: bool = True,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_target_network = use_target_network
        self.device = device
        self.update_count = 0

        self.q_net = QNetwork(state_dim, n_actions, hidden_dim).to(device)
        self.target_net = QNetwork(state_dim, n_actions, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randrange(self.n_actions)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self):
        """
        Sample a minibatch and perform one gradient update.

        Returns
        -------
        float | None
            The scalar loss, or None if the replay buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_target_network:
                next_q_values = self.target_net(next_states).max(dim=1).values
            else:
                next_q_values = self.q_net(next_states).max(dim=1).values
            target = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.use_target_network and self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


def train(
    env_name: str,
    n_episodes: int = 1000,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    gamma: float = 0.99,
    buffer_capacity: int = 10_000,
    batch_size: int = 64,
    target_update_freq: int = 100,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    use_target_network: bool = True,
    seed: int = 0,
):
    """
    Train DQN on a Gymnasium environment with vector observations and discrete actions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name)
    env.action_space.seed(seed)

    if not hasattr(env.observation_space, "shape") or env.observation_space.shape is None:
        raise ValueError("This script expects a vector observation space.")
    if not hasattr(env.action_space, "n"):
        raise ValueError("This script expects a discrete action space.")

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DQNAgent(
        state_dim,
        n_actions,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        use_target_network=use_target_network,
        device=device,
    )

    episode_returns = []
    losses = []
    epsilon = epsilon_start

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_return = 0.0
        done = False

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            total_return += reward
            state = next_state

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_returns.append(total_return)

        if (ep + 1) % 100 == 0:
            mean_100 = np.mean(episode_returns[-100:])
            print(
                f"[DQN {'target' if use_target_network else 'no-target'}] "
                f"Ep {ep + 1:4d}  Return: {total_return:6.1f}  "
                f"Mean(100): {mean_100:6.1f}  ε: {epsilon:.3f}"
            )

    env.close()
    return np.asarray(episode_returns, dtype=float)


def run_multiple_seeds(env_name: str, n_seeds: int = 5, **kwargs):
    """Run DQN over multiple seeds and stack the returns."""
    all_returns = []
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        returns = train(env_name, seed=seed, **kwargs)
        all_returns.append(returns)
    return np.asarray(all_returns, dtype=float)


def plot_results(returns_dict, title: str = "DQN Training", window: int = 20):
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
    parser.add_argument("--ablation", action="store_true", help="Also run without the target network.")
    args = parser.parse_args()

    print(f"Training DQN on {args.env}")
    returns_with = run_multiple_seeds(
        args.env,
        n_seeds=args.seeds,
        n_episodes=args.episodes,
        use_target_network=True,
    )
    results = {"DQN (with target network)": returns_with}

    if args.ablation:
        print("\nAblation: DQN without target network")
        returns_without = run_multiple_seeds(
            args.env,
            n_seeds=args.seeds,
            n_episodes=args.episodes,
            use_target_network=False,
        )
        results["DQN (no target network)"] = returns_without

    fig, _ = plot_results(results, title=f"DQN - {args.env}")
    plt.savefig("dqn_results.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Saved: dqn_results.png")
