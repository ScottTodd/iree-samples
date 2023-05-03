from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x

model = MLP([12, 8, 4])
batch = jnp.ones((32, 10))

@jax.jit
def run_mlp(batch):
  variables = model.init(jax.random.PRNGKey(0), batch)
  return model.apply(variables, batch)

output = run_mlp(batch)
print(output)
