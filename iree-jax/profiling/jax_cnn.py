import jax
import jax.numpy as jnp
import flax.linen as nn

class CNN(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x

# model = CNN()
# batch = jnp.ones((32, 64, 64, 10))  # (N, H, W, C) format
# variables = model.init(jax.random.PRNGKey(0), batch)
# output = model.apply(variables, batch)

model = CNN()
batch = jnp.ones((32, 64, 64, 10))  # (N, H, W, C) format

@jax.jit
def run_cnn(batch):
  variables = model.init(jax.random.PRNGKey(0), batch)
  return model.apply(variables, batch)

output = run_cnn(batch)
print(output)
