from flax import struct
from jax import numpy as jnp


@struct.dataclass
class RunningMeanStdState:
    mean: jnp.ndarray | float
    var: jnp.ndarray | float
    count: jnp.ndarray | float
    epsilon: float


def update_rms(state: RunningMeanStdState, batch):
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]

    mean, var, count = state.mean, state.var, state.count
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return RunningMeanStdState(
        mean=new_mean,
        var=new_var,
        count=new_count,
        epsilon=state.epsilon,
    )


def normalize_with_rms(x, state: RunningMeanStdState):
    return (x - state.mean) / jnp.sqrt(state.var + state.epsilon)


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4):
        """Tracks the mean, variance and count of values."""
        self.epsilon = epsilon
    
    def init(self, x: jnp.array):
        state = RunningMeanStdState(
            mean = jnp.zeros_like(x),
            var = jnp.ones_like(x),
            count = self.epsilon,
            epsilon = self.epsilon,
        )
        return state

    def update(self, state: RunningMeanStdState, batch):
        """Updates the mean, var and count from a batch of samples."""
        return update_rms(state, batch)
    
    def normalize(self, x, state: RunningMeanStdState):
        return normalize_with_rms(x, state)
