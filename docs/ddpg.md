# DDPG: Ornstein–Uhlenbeck (OU) Exploration Noise

## Overview
This note summarizes the OU process and how it is used for action exploration in DDPG.

## OU Process
The OU process generates temporally correlated noise, which can be useful for smooth control in continuous action spaces. In continuous time:

**OU is a mean‑reverting Wiener process**. First, it supply noise accumulaiton like Wiener process.Second it pull back the accumulated drift so that "bound" the process somehow. The second point make it suitable for RL exploration. Furthermore, OU noise is more moother than random walk.

$$
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t
$$

- $\theta$ controls the mean-reversion speed.
- $\mu$ is the long-term mean (often 0 for exploration noise).
- $\sigma$ controls the noise scale.
- $W_t$ is standard Brownian motion.

Discrete-time Update
With time step $\Delta t$ (one env step is treated as $\Delta t = 1$ in this repo):

$$
X_{t+1} = X_t + \theta(\mu - X_t)\Delta t + \sigma \sqrt{\Delta t}\,\epsilon_t,
\quad \epsilon_t \sim \mathcal{N}(0, I)
$$

DDPG Usage
DDPG applies OU noise to the deterministic actor output:

$$
a_t = \pi_\phi(s_t) + X_t
$$

Then actions are clipped to the environment action bounds. The OU state $X_t$
is carried across steps during rollout, so the noise is temporally correlated.

Implementation Notes (this repo)
- Exploration type: `ou_noise`.
- OU parameters in config:
  - `ou_theta` corresponds to $\theta$.
  - `exploration_noise` corresponds to $\sigma$.
- $\Delta t$ is treated as 1 per env step for consistency with normal noise.
- The OU state is initialized to zero each rollout and updated per step.
