# In riemannian_score_sde/models/score_network.py

import haiku as hk
import jax.numpy as jnp
# You might need to import jax.nn for activations if not already imported
import jax.nn

# Original MLP class would be here:
# class MLP(hk.Module): ...

# Add these classes:
class GaussianFourierProjection(hk.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30., name=None):
    super().__init__(name=name)
    self.W = hk.get_parameter("W", shape=(embed_dim // 2,), dtype=jnp.float32, init=hk.initializers.RandomNormal(scale))
  
  def __call__(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Dense(hk.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, output_dim, name=None):
    super().__init__(name=name)
    self.dense = hk.Linear(output_dim)
  
  def __call__(self, x):
    return jnp.expand_dims(jnp.expand_dims(self.dense(x), -1), -1)


class SPDUnetScoreNetwork(hk.Module):
  """A time-dependent score-based model built upon U-Net architecture for SPD matrices."""
  def __init__(self, manifold_dim, channels=[32, 64], embed_dim=256, activation=jax.nn.swish, name=None):
    super().__init__(name=name)
    self.manifold_dim = manifold_dim
    self.channels = channels
    self.embed_dim = embed_dim
    self.act = activation 

    self.embed = hk.Sequential([
        GaussianFourierProjection(embed_dim=embed_dim, name='gfp_embed'),
        hk.Linear(embed_dim, name='linear_embed')
    ])

    self.conv1 = hk.Conv2D(channels[0], kernel_shape=2, stride=1, with_bias=False, name='conv1')
    self.dense1 = Dense(channels[0], name='dense1')
    self.gnorm1 = hk.GroupNorm(4, create_scale=True, create_offset=True, name='gnorm1')

    self.conv2 = hk.Conv2D(channels[1], kernel_shape=2, stride=1, with_bias=False, name='conv2')
    self.dense2 = Dense(channels[1], name='dense2')
    self.gnorm2 = hk.GroupNorm(32, create_scale=True, create_offset=True, name='gnorm2')

    self.tconv2 = hk.Conv2DTranspose(channels[0], kernel_shape=2, stride=1, with_bias=False, name='tconv2')
    self.dense7 = Dense(channels[0], name='dense7')
    self.tgnorm2 = hk.GroupNorm(32, create_scale=True, create_offset=True, name='tgnorm2')

    self.tconv1 = hk.Conv2DTranspose(1, kernel_shape=2, stride=1, name='tconv1')


  def __call__(self, x, t):
    # x: (batch_size, manifold_dim, manifold_dim)
    # t: (batch_size,)

    x_reshaped = jnp.expand_dims(x, axis=1) # (batch, 1, dim, dim)

    embed = self.act(self.embed(t))

    h1 = self.conv1(x_reshaped)
    h1 += self.dense1(embed)
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)

    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)

    h = self.tconv2(h2)
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)

    h = jnp.concatenate([h, h1], axis=1) # Skip connection
    
    score_output = self.tconv1(h)
    score = jnp.squeeze(score_output, axis=1)
    
    score = (score + jnp.swapaxes(score, -1, -2)) / 2.0
    
    return score