# Teaching a Robot to Cross a Grid (Reinforcement Learning)

This project is a small experiment in **reinforcement learning (RL)** — the branch of machine learning where an "agent" learns good behaviour by trial and error, guided only by rewards. There is no dataset of correct answers; the agent tries things, sees what pays off, and gradually works out a good strategy.

## The problem: a grid world

Picture a 5×5 chessboard. A little agent starts in the **top-left corner** and wants to reach the **goal** in the bottom-right, which pays a reward of **+1**. One square is a **trap** that pays **−1**. Reaching either the goal or the trap ends the game.

Each turn the agent picks a direction — up, down, left, or right — and moves one square. Two wrinkles make it interesting:

- **The floor is slippery.** 10% of the time the agent slips and moves in a random direction instead of the one it chose. So a good strategy has to be robust, not just lucky.
- **Future rewards count for slightly less than immediate ones.** This is set by a "discount factor" `γ = 0.95` — reaching the goal sooner is worth a little more than reaching it later.

The agent's job is to learn a **policy**: a rule that says, for every square, which direction to move.

## The three methods compared

Each method learns the same thing — a good policy — but in a different way.

**1. Value iteration.** This is the "cheating" baseline: it assumes the agent already knows the full rules of the world (the map, the slip probability, where the reward is). Given that, it can calculate the mathematically optimal policy directly. We use it to know what "perfect play" looks like, so we can judge the other two.

**2. Q-learning.** This method *doesn't* know the rules in advance — it has to learn by playing thousands of games. It keeps a table of scores, one per (square, direction) pair, estimating how good each move is. After every step it nudges those scores towards what it just experienced. Q-learning is *optimistic*: it always assumes it will play perfectly from the next square onward, so it tends to learn the bold, optimal route.

**3. SARSA.** Almost identical to Q-learning, with one difference: it updates its scores based on the move it *actually* makes next — including the occasional random exploratory move — rather than assuming perfect play. Because it "knows" it sometimes acts randomly, it learns to play *cautiously*, giving the trap a wider berth. Think of it as a nervous driver who leaves extra room, versus Q-learning's confident one.

Both Q-learning and SARSA need to **explore** to learn. They do this with an "ε-greedy" rule: most of the time they take their current best-guess move, but 10% of the time (ε = 0.1) they move randomly, to discover options they'd otherwise never try.

## A little of the maths

Every method rests on the same core idea — the value of a move equals the reward you get now, plus the (discounted) value of wherever you end up next.

**Value iteration** applies this repeatedly until the numbers stop changing, then picks the best move in each square:

$$v_{k+1}(s) = \max_a \sum_{s',r} p(s',r \mid s,a)\,\big[r + \gamma v_k(s')\big]$$

Here `s` is a square, `a` is a direction, `r` is the reward, and `s'` is the next square. It literally averages over every possible outcome of a move (including the 10% slips) and keeps the best.

**Q-learning** updates its score `Q(s,a)` for the move it took, pulling it towards the reward plus the *best* score available next. `α = 0.1` is the learning rate — how big each nudge is:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big]$$

**SARSA** is the same but uses the score of the move it *actually* takes next, `a'`, rather than the best one:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma Q(s',a') - Q(s,a)\big]$$

That single swap — `max` versus the action actually taken — is the entire difference between "optimistic" and "cautious" learning.

## Results

The picture below shows the policy each method learned, as arrows pointing the chosen direction in each square. Left is the optimal policy (from value iteration); the middle and right are what Q-learning and SARSA discovered on their own. **X** is the trap, **G** the goal.

![Policy comparison](policy_comparison(2).png)

| Policy | Average score per game | How often it matches the optimal policy |
|---|---:|---:|
| Optimal (value iteration) | 0.6693 | — |
| Q-learning | 0.6565 | 19 of 23 squares (82.6%) |
| SARSA | 0.6508 | 16 of 23 squares (69.6%) |

**What this tells us:** Q-learning ends up very close to the optimal route, while SARSA plays it safer and so matches the optimum less often and scores slightly lower. Neither reaches a perfect score, because both keep making 10% random moves throughout — they converge to the best *cautious-explorer* policy rather than to flawless play. This trade-off (Q-learning bold, SARSA careful) is the classic textbook illustration of off-policy versus on-policy learning.

## Running it yourself

You need two files in the same folder as the notebook — **`gridworld.py`** (the grid environment and plotting helpers) and **`tabular_rl_starter.py`** (the three algorithms) — plus Python with NumPy and Matplotlib.

```bash
pip install numpy matplotlib
jupyter notebook Untitled1.ipynb   # then run all cells from top to bottom
```

The notebook runs value iteration first, then Q-learning and SARSA (5,000 practice games each), then plots the comparison above.
