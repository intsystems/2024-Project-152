# %%
%load_ext autoreload
%autoreload 2

import os
os.environ["GEOMSTATS_BACKEND"] = 'jax'

import jax
import numpy as np
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt

from score_sde.utils import GlobalRNG

g_rng = GlobalRNG()

# %% [markdown]
# ## Define an SDE class

# %%
from score_sde.sde import SDE, VPSDE, RSDE, ProbabilityFlowODE

# %% [markdown]
# 

# %% [markdown]
# ## Define rollout sampler for an SDE

# %%
from score_sde.sampling import get_pc_sampler

# %%
def plot_2dsde(x_hist):
    fig, axes = plt.subplots(1,3, sharey=False, figsize=(9,3))

    axes[0].set_aspect('equal', share=False)
    n = 4
    for i in range(n):
        i = int(i * x_hist.shape[1] / n)
        sns.kdeplot(x=x_hist[:,i,0], y=x_hist[:,i,1], ax=axes[0])
    sns.kdeplot(x=x_hist[:,-1,0], y=x_hist[:,-1,1], ax=axes[0])

    axes[1].plot(x_hist.mean(axis=0)[..., 0], label='x')
    axes[1].plot(x_hist.mean(axis=0)[..., 1], label='y')
    axes[1].legend()

    axes[2].plot(x_hist.std(axis=0)[..., 0], label='x')
    axes[2].plot(x_hist.std(axis=0)[..., 1], label='y')
    axes[2].legend()
    
    axes[0].set_title("KDE estimate")
    axes[1].set_title("Mean")
    axes[2].set_title("Std")

    plt.tight_layout()

    return fig, axes

# %%
N=1000
# Define forward diffusion to be Ornstein Uhlenbeck process
# forward_sde = VPSDE(tf=10, beta_0=1, beta_f=1)
forward_sde = VPSDE(tf=1, beta_0=0.01, beta_f=8)
def rescale_t(t):
    return 0.5 * t ** 2 * (forward_sde.beta_f - forward_sde.beta_0) + t * forward_sde.beta_0
# Sampler of forward process via Euler-Maruyama discretisation
sampler = get_pc_sampler(forward_sde, N=N, predictor='EulerMaruyamaPredictor', return_hist=True)

# Target distribution is 2D Gaussian with mean=10 and std=1
mean, var = 10, 1
from score_sde.datasets import GaussianMixture
dataset = GaussianMixture((2000,), next(g_rng), means=[jnp.array([mean,mean])], stds=[jnp.array([var,var])], weights=[1.0])
dataset.means.shape
x_init = next(dataset)

x_sample, x_hist, _ = sampler(jax.random.PRNGKey(0), x_init)
fig, axes = plot_2dsde(x_hist.swapaxes(0, 1))
t_ = jnp.linspace(forward_sde.t0, forward_sde.tf,num=N, endpoint=True)
# axes[1].plot(mean * jnp.exp(-0.5 * t_), label='$\mu \exp(-t/2)$')
axes[1].plot(mean * jnp.exp(-0.5 * rescale_t(t_)), label='$\mu \exp(-s/2)$')
axes[1].legend()
axes[2].legend()

# plt.scatter(x_init[..., 0], x_init[..., 1])

# %%
# This score is ONLY accurate for beta(t) = 1
def score_fn(x, t):
    """
    X_t = e^{-t}X_0 + \sqrt{1-e^{-2t}}Z with X_0 ~ p_0=N(a,1) and  Z ~ N(0,1)
    so X_t ~ N(ae^{-t},1)
    hence \nabla log p(X_t) = x - a*e^{-t}
    """
    # return -(x - mean * jnp.exp(-t[..., None]/2))
    return -(x - mean * jnp.exp(-0.5*rescale_t(t)[..., None]))

reverse_sde = forward_sde.reverse(score_fn) # reverse process

reverse_sampler = get_pc_sampler(reverse_sde, N=N, predictor='EulerMaruyamaPredictor', return_hist=True)

# Samples from limiting distribution ~ N(0, 1)
x_init = reverse_sde.sde.sample_limiting_distribution(jax.random.PRNGKey(0), (2000,2))

x_sample, x_hist, _ = reverse_sampler(jax.random.PRNGKey(0), x_init)
fig, axes = plot_2dsde(x_hist.swapaxes(0, 1))
t_ = jnp.linspace(reverse_sde.t0, reverse_sde.tf,num=N, endpoint=True)

# axes[1].plot(mean * jnp.exp(-0.5 * t_), label='$\mu \exp(-t/2)$')
axes[1].plot(mean * jnp.exp(-0.5 * rescale_t(t_)), label='$\mu \exp(-s/2)$')
axes[1].legend()
axes[2].legend()


# %%
from score_sde.likelihood import get_likelihood_fn

x_init = mean + var * forward_sde.sample_limiting_distribution(jax.random.PRNGKey(0), (2000,2))

likelihood_fn = get_likelihood_fn(forward_sde, score_fn, hutchinson_type='Gaussian', bits_per_dimension=False)
logl, z, _ = likelihood_fn(next(g_rng), x_init)

plt.scatter(x_init[..., 0], x_init[..., 1], c = jnp.exp(logl))
plt.scatter(z[..., 0], z[..., 1], c = jnp.exp(forward_sde.limiting_distribution_logp(z)))
plt.gca().set_aspect('equal')

# %%
from score_sde.models import AmbientGenerator, Concat
from score_sde.utils import TrainState
from score_sde.losses import get_ema_loss_step_fn, get_dsm_loss_fn, get_ism_loss_fn
from score_sde.models import SDEPushForward
import geomstats as gs
import haiku as hk
import optax

manifold = gs.geometry.euclidean.Euclidean(2)

# score network
def score_model(x, t):
    score = AmbientGenerator(dict(
        _target_ = "score_sde.models.Concat",
        hidden_shapes = [512,512,512], 
        act='sin'
    ),
    dict(_target_ = "riemannian_score_sde.models.NoneEmbedding"),
    2, manifold)
    return score(x, t)

score_model = hk.transform_with_state(score_model)

x = next(dataset)
params, state = score_model.init(rng=next(g_rng), x=x, t=0)

# Optmiser + scheduler
warmup_steps = 100
steps = 10000
schedule_fn = optax.join_schedules([
        optax.linear_schedule(
            init_value= 0.0,
            end_value= 1.0,
            transition_steps = warmup_steps
        ),
        optax.cosine_decay_schedule(
            init_value= 1.0,
            decay_steps=steps - warmup_steps,
            alpha= 0.0
        )
    ],
  boundaries=[warmup_steps]
)

optimiser = optax.chain(
    optax.adam(learning_rate=2e-4, b1=0.9,b2=0.999, eps=1e-8), optax.scale_by_schedule(schedule_fn)
)
opt_state = optimiser.init(params)

next_rng = next(g_rng)
train_state = TrainState(
    opt_state=opt_state,
    model_state=state,
    step=0,
    params=params,
    ema_rate=0.9999,
    params_ema=params,
    rng=next_rng,
)

# Loss function
pushforward = SDEPushForward(manifold, forward_sde)
loss = get_dsm_loss_fn(pushforward, model=score_model, eps=1e-3, train=True)

train_step_fn = get_ema_loss_step_fn(
    loss,
    optimizer=optimiser,
    train=True,
)

train_step_fn = jax.jit(train_step_fn)

# Training
losses = np.zeros(steps)
rng = next(g_rng)
for i in range(steps):
    batch = {"data": next(dataset)}
    rng, next_rng = jax.random.split(rng)
    (rng, train_state), loss = train_step_fn((next_rng, train_state), batch)
    losses[i] = loss
    if i % 100 == 0:
        print(f"{i:4d}: {loss:.3f}")

fig, axis = plt.subplots(1, 1, figsize=(15,5))
axis.plot(losses)
axis.set_xlabel("number of iterations")
axis.set_ylabel("loss")

# %%
from score_sde.models import get_score_fn

nn_score_fn = get_score_fn(forward_sde, score_model, train_state.params_ema, train_state.model_state)

# %%
x_t = forward_sde.sample_limiting_distribution(next(g_rng), (2000,2)) #next(dataset)
x_0 = jnp.zeros_like(x_t)

forward_sde.marginal_prob(x_t, 10 * jnp.ones(x_t.shape[:-1]))

# %%
n = 5

fig, axs = plt.subplots(2, n, figsize=(n*3, 2*3))

x = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-2,12,10), jnp.linspace(-2,12,10)
    ), axis=-1
).reshape((1,-1,2))

for i in range(n):
    t = (i * forward_sde.tf)/(n-1) + 1e-3
    nn_score = nn_score_fn(x, t * jnp.ones(x.shape[:-1])[..., None])
    score = score_fn(x, t * jnp.ones(x.shape[:-1]))

    axs[0][i].quiver(x[..., 0], x[..., 1], score[..., 0], score[..., 1])
    axs[1][i].quiver(x[..., 0], x[..., 1], nn_score[..., 0], nn_score[..., 1])

# %%
reverse_sde = forward_sde.reverse(jax.jit(nn_score_fn))
N = 1000
reverse_sampler = jax.jit(get_pc_sampler(reverse_sde, N=N, predictor='EulerMaruyamaPredictor', return_hist=True))

x_init = reverse_sde.sde.sample_limiting_distribution(jax.random.PRNGKey(0), (2000,2))

x_sample, x_hist, _ = reverse_sampler(jax.random.PRNGKey(0), x_init)

fig, axes = plot_2dsde(x_hist.swapaxes(0, 1))
t_ = jnp.linspace(reverse_sde.t0, reverse_sde.tf,num=N, endpoint=True)
axes[1].plot(mean * jnp.exp(-0.5 * t_), label='$\mu \exp(-t/2)$')
axes[1].legend()
axes[2].legend()

# %%
from score_sde.likelihood import get_likelihood_fn

x_init = mean + var * forward_sde.sample_limiting_distribution(jax.random.PRNGKey(0), (2000,2))

likelihood_fn = get_likelihood_fn(forward_sde, nn_score_fn, hutchinson_type='Gaussian', bits_per_dimension=False)
logl, z, N = likelihood_fn(next(g_rng), x_init)

plt.scatter(x_init[..., 0], x_init[..., 1], c = jnp.exp(logl))
plt.scatter(z[..., 0], z[..., 1], c = jnp.exp(forward_sde.limiting_distribution_logp(z)))
plt.gca().set_aspect('equal')


