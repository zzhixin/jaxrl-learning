#%%
import flashbax as fbx
from jax import numpy as jnp, random
import jax


# First define hyper-parameters of the buffer.
max_length = 32 # Maximum length of buffer (max number of experiences stored within the state).
min_length = 8 # Minimum number of experiences saved in the buffer state before we can sample.
sample_batch_size = 4 # Batch size of experience data sampled from the buffer.

add_sequences = False # Will we be adding data in sequences to the buffer?
add_batch_size = 6    # Will we be adding data in batches to the buffer? 
                      # It is possible to add data in both sequences and batches. 
                      # If adding data in batches, what is the batch size that is being added each time?

# Instantiate the flat buffer, which is a Dataclass of pure functions.
buffer = fbx.make_flat_buffer(max_length, min_length, sample_batch_size, add_sequences, add_batch_size)

def create_fake_timestep(key, obs_shape, action_shape):
    key0, key1, key2, key3, key4, key5 = random.split(key, 6)
    obs = random.normal(key0, obs_shape)
    action = random.normal(key1, action_shape)
    rew = random.normal(key2, ())
    next_obs = random.normal(key3, obs_shape)
    ter = random.normal(key4, ())
    tru = random.normal(key5, ())
    return {
        "obs": obs,
        "action": action, 
        "rew": rew,
        "ter": ter, 
        "tru": tru,
    }


fake_timestep = create_fake_timestep(random.key(0), obs_shape=(3, ), action_shape=(2, ))
state = buffer.init(fake_timestep)
for key, value in state.experience.items():
    print(f"{key}: {value.shape}")
    assert value.shape[0] == add_batch_size
    assert value.shape[1] == max_length//add_batch_size


print(f"current index: {state.current_index}")
keys = random.split(random.key(0), add_batch_size)
fake_batch_timesteps = jax.vmap(create_fake_timestep, in_axes=(0, None, None))(keys, (3, ), (2, ))
state = buffer.add(state, fake_batch_timesteps)
print(f"current index: {state.current_index}")
assert not buffer.can_sample(state)
state = buffer.add(state, fake_batch_timesteps)
print(f"current index: {state.current_index}")
assert buffer.can_sample(state)

for i in range(10):
    state = buffer.add(state, fake_batch_timesteps)
    print(f"current index: {state.current_index}")

sampled_batch = buffer.sample(state, random.key(0))
print(sampled_batch.experience.first.keys()) # prints dict_keys(['obs', 'reward'])
print(sampled_batch.experience.second.keys()) # prints dict_keys(['obs', 'reward'])
print(sampled_batch.experience.first['obs'].shape) # prints (4,) = (sample_batch_size, *fake_timestep['reward'].shape)
print(sampled_batch.experience.second['obs'].shape) # prints (4,) = (sample_batch_size, *fake_timestep['reward'].shape)

