
"""GridWorld environment used for the tabular RL coursework."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class GridWorld:
    """A stochastic 5x5 gridworld with one goal and one trap."""

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
    DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        goal: tuple[int, int] = (4, 4),
        trap: tuple[int, int] = (3, 2),
        goal_reward: float = 1.0,
        trap_reward: float = -1.0,
        step_reward: float = 0.0,
        noise: float = 0.1,
        gamma: float = 0.95,
        start: tuple[int, int] = (0, 0),
        seed: int | None = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.n_actions = 4

        self.goal = goal
        self.trap = trap
        self.start = start
        self.terminal_states = {goal, trap}

        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.step_reward = step_reward
        self.noise = noise
        self.gamma = gamma

        self.rng = np.random.default_rng(seed)
        self.state: int | None = None

        self.P = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=float)
        self.R = np.zeros((self.n_states, self.n_actions), dtype=float)
        self._build_model()

    def _rc_to_s(self, row: int, col: int) -> int:
        return row * self.cols + col

    def _s_to_rc(self, s: int) -> tuple[int, int]:
        return divmod(s, self.cols)

    def _move(self, row: int, col: int, action: int) -> tuple[int, int]:
        dr, dc = self.DELTAS[action]
        nr, nc = row + dr, col + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return nr, nc
        return row, col

    def _build_model(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                s = self._rc_to_s(r, c)

                if (r, c) in self.terminal_states:
                    for a in range(self.n_actions):
                        self.P[s, a, s] = 1.0
                        self.R[s, a] = 0.0
                    continue

                for a in range(self.n_actions):
                    for actual_a in range(self.n_actions):
                        if actual_a == a:
                            prob = (1.0 - self.noise) + self.noise / self.n_actions
                        else:
                            prob = self.noise / self.n_actions
                        nr, nc = self._move(r, c, actual_a)
                        ns = self._rc_to_s(nr, nc)
                        self.P[s, a, ns] += prob

                    for ns in range(self.n_states):
                        nr, nc = self._s_to_rc(ns)
                        if (nr, nc) == self.goal:
                            self.R[s, a] += self.P[s, a, ns] * self.goal_reward
                        elif (nr, nc) == self.trap:
                            self.R[s, a] += self.P[s, a, ns] * self.trap_reward
                        else:
                            self.R[s, a] += self.P[s, a, ns] * self.step_reward

        if not np.allclose(self.P.sum(axis=2), 1.0):
            raise ValueError("Transition probabilities do not sum to 1.")

    def describe_layout(self) -> str:
        """Return a short string describing the default coordinates."""
        return (
            f"Start={self.start}, Goal={self.goal}, Trap={self.trap}, "
            f"noise={self.noise}, gamma={self.gamma}"
        )

    def reset(self, seed: int | None = None) -> int:
        """Reset the environment to the start state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self._rc_to_s(*self.start)
        return self.state

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """Take one step in the environment."""
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if action not in range(self.n_actions):
            raise ValueError(f"Action must be in {{0,1,2,3}}, got {action}.")

        probs = self.P[self.state, action]
        next_state = int(self.rng.choice(self.n_states, p=probs))
        nr, nc = self._s_to_rc(next_state)

        if (nr, nc) == self.goal:
            reward = self.goal_reward
        elif (nr, nc) == self.trap:
            reward = self.trap_reward
        else:
            reward = self.step_reward

        done = (nr, nc) in self.terminal_states
        self.state = next_state
        return next_state, reward, done, {"row": nr, "col": nc}


def plot_value_function(env: GridWorld, V: np.ndarray, title: str = "Value Function", ax=None):
    """Plot a heatmap of a state-value function."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    grid = V.reshape(env.rows, env.cols)
    norm = Normalize(vmin=float(grid.min()), vmax=float(grid.max()))
    im = ax.imshow(grid, cmap="RdYlGn", norm=norm)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for r in range(env.rows):
        for c in range(env.cols):
            s = env._rc_to_s(r, c)
            ax.text(c, r, f"{V[s]:.2f}", ha="center", va="center", fontsize=8, color="black")

    gr, gc = env.goal
    tr, tc = env.trap
    sr, sc = env.start
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(tc, tr, "X", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=12, fontweight="bold", color="navy")

    ax.set_title(title)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax


def plot_policy(env: GridWorld, policy: np.ndarray, title: str = "Policy", ax=None):
    """Plot a policy as arrows on the grid."""
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(-0.5, env.rows - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for r in range(env.rows):
        for c in range(env.cols):
            s = env._rc_to_s(r, c)
            if (r, c) == env.goal:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="mediumseagreen"))
                ax.text(c, r, "G", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
            elif (r, c) == env.trap:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="tomato"))
                ax.text(c, r, "X", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
            elif (r, c) == env.start:
                ax.text(c, r - 0.18, "S", ha="center", va="center", fontsize=10, fontweight="bold", color="navy")
                ax.text(c, r + 0.18, arrows[int(policy[s])], ha="center", va="center", fontsize=18)
            else:
                ax.text(c, r, arrows[int(policy[s])], ha="center", va="center", fontsize=18)

    ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title)
    return ax


def plot_learning_curves(returns_dict, window: int = 50, title: str = "Learning Curves", ax=None):
    """
    Plot learning curves for one or more algorithms.

    ``returns_dict`` maps a label to either:
        - a 1D array of shape (episodes,)
        - a 2D array of shape (runs, episodes)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

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
    ax.set_ylabel("Discounted return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def compare_policies(env: GridWorld, pi1: np.ndarray, pi2: np.ndarray, label1: str = "π1", label2: str = "π2") -> float:
    """Return and print the state-wise agreement rate between two policies."""
    non_terminal = [s for s in range(env.n_states) if env._s_to_rc(s) not in env.terminal_states]
    agree = sum(int(pi1[s] == pi2[s]) for s in non_terminal)
    pct = 100.0 * agree / len(non_terminal)
    print(f"Policy agreement ({label1} vs {label2}): {agree}/{len(non_terminal)} states ({pct:.1f}%)")
    return pct


def evaluate_policy(env: GridWorld, policy: np.ndarray, n_episodes: int = 1000, max_steps: int = 500, seed: int | None = None):
    """Estimate the discounted return distribution of a fixed policy by Monte Carlo simulation."""
    returns = []
    base_seed = seed if seed is not None else None

    for ep in range(n_episodes):
        episode_seed = None if base_seed is None else base_seed + ep
        state = env.reset(seed=episode_seed)
        discounted_return = 0.0
        discount = 1.0

        for _ in range(max_steps):
            action = int(policy[state])
            state, reward, done, _ = env.step(action)
            discounted_return += discount * reward
            discount *= env.gamma
            if done:
                break

        returns.append(discounted_return)

    return np.asarray(returns, dtype=float)


if __name__ == "__main__":
    env = GridWorld(seed=0)
    print("Environment summary:", env.describe_layout())
    print(f"States: {env.n_states}, Actions: {env.n_actions}")
    print(f"Transition probabilities sum to 1: {np.allclose(env.P.sum(axis=2), 1.0)}")

    state = env.reset(seed=0)
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < 100:
        action = int(env.rng.integers(env.n_actions))
        state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
    print(f"Random episode: {steps} steps, total reward = {total_reward:.2f}")

    V_demo = np.linspace(-0.4, 0.9, env.n_states)
    pi_demo = np.zeros(env.n_states, dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_value_function(env, V_demo, title="Demo value function", ax=axes[0])
    plot_policy(env, pi_demo, title="Demo policy", ax=axes[1])
    plt.tight_layout()

    # Avoid warnings in headless environments used for automated checks.
    if "agg" in plt.get_backend().lower():
        fig.savefig("gridworld_demo.png", dpi=140, bbox_inches="tight")
        print("Saved: gridworld_demo.png")
    else:
        plt.show()
