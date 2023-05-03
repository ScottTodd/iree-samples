import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000)

selu_jit = jax.jit(selu)

# Warm up
selu_jit(x).block_until_ready()
# %timeit selu_jit(x).block_until_ready()
