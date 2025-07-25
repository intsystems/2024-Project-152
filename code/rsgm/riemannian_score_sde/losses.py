"""All functions related to loss computation and optimization.
"""

from typing import Tuple
from geomstats.backend import zeros_like, expand_dims, mean, log
import jax.random as random

from score_sde.utils import batch_mul, ParametrisedScoreFunction
from score_sde.models import SDEPushForward, MoserFlow, div_noise, get_riemannian_div_fn

def loss_base(rng, batch, sde, transform, model, params, states, train, score_manipulation_fn):
    """Base function for loss calculation to be used by specific loss functions."""
    score_fn = sde.reparametrise_score_fn(model, params, states, train, True)
    # print(f"DEBUG: In loss_base - batch keys: {batch.keys()}")
    # print(f"DEBUG: In loss_base - type(batch['data']): {type(batch['data'])}")
    # print(f"DEBUG: In loss_base - batch['data'].shape (if JAX array): {batch['data'].shape if hasattr(batch['data'], 'shape') else 'N/A'}")
    # print(f"DEBUG: In loss_base - transform type: {type(transform)}")
    # print(batch["data"].shape)
    # print(batch["data"])
    # from score_sde.models.transform import Id
    # transform = Id()
    # print(transform.inv(0))
    y_0, context = transform.inv(batch["data"]), batch["context"]
    t = random.uniform(rng, (y_0.shape[0],), minval=sde.t0 + 1e-3, maxval=sde.tf)
    return score_fn, y_0, context, t

def get_dsm_loss_fn(
    pushforward: SDEPushForward,
    model: ParametrisedScoreFunction,
    train: bool = True,
    like_w: bool = True,
    eps: float = 1e-3,
    s_zero=True,
    **kwargs
):
    sde = pushforward.sde

    def loss_fn(
        rng, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        rng, step_rng = random.split(rng)
        score_fn, y_0, context, t = loss_base(step_rng, batch, sde, pushforward.transform, model, params, states, train, lambda x: x)

        rng, step_rng = random.split(rng)
        # sample p(y_t | y_0)
        # compute $\nabla \log p(y_t | y_0)$
        if s_zero:  # l_{t|0}
            y_t = sde.marginal_sample(step_rng, y_0, t)
            if "n_max" in kwargs and kwargs["n_max"] <= -1:
                get_logp_grad = lambda y_0, y_t, t: sde.varhadan_exp(
                    y_0, y_t, zeros_like(t), t
                )[1]
            else:
                get_logp_grad = lambda y_0, y_t, t: sde.grad_marginal_log_prob(
                    y_0, y_t, t, **kwargs
                )[1]
            logp_grad = get_logp_grad(y_0, y_t, t)
            std = expand_dims(sde.marginal_prob(zeros_like(y_t), t)[1], -1)
        else:  # l_{t|s}
            y_t, y_hist, timesteps = sde.marginal_sample(
                step_rng, y_0, t, return_hist=True
            )
            y_s = y_hist[-2]
            delta_t, logp_grad = sde.varhadan_exp(y_s, y_t, timesteps[-2], timesteps[-1])
            delta_t = t  # NOTE: works better?
            std = expand_dims(sde.marginal_prob(zeros_like(y_t), delta_t)[1], -1)

        # compute approximate score at y_t
        score, new_model_state = score_fn(y_t, t, context, rng=step_rng)
        score = score.reshape(y_t.shape)

        if not like_w:
            score = batch_mul(std, score)
            logp_grad = batch_mul(std, logp_grad)
            losses = sde.manifold.metric.squared_norm(score - logp_grad, y_t)
        else:
            # compute $E_{p{y_0}}[|| s_\theta(y_t, t) - \nabla \log p(y_t | y_0)||^2]$
            g2 = sde.coefficients(zeros_like(y_0), t)[1] ** 2
            losses = sde.manifold.metric.squared_norm(score - logp_grad, y_t) * g2

        loss = mean(losses)
        return loss, new_model_state

    return loss_fn

def get_ism_loss_fn(
    pushforward: SDEPushForward,
    model: ParametrisedScoreFunction,
    train: bool,
    like_w: bool = True,
    hutchinson_type="Rademacher",
    eps: float = 1e-3,
):
    sde = pushforward.sde

    def loss_fn(
        rng, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        rng, step_rng = random.split(rng)
        score_fn, y_0, context, t = loss_base(step_rng, batch, sde, pushforward.transform, model, params, states, train, lambda x: x)

        rng, step_rng = random.split(rng)
        y_t = sde.marginal_sample(step_rng, y_0, t)
        score, new_model_state = score_fn(y_t, t, context, rng=step_rng)
        score = score.reshape(y_t.shape)

        # ISM loss
        rng, step_rng = random.split(rng)
        epsilon = div_noise(step_rng, y_0.shape, hutchinson_type)
        drift_fn = lambda y, t, context: score_fn(y, t, context, rng=step_rng)[0]
        div_fn = get_riemannian_div_fn(drift_fn, hutchinson_type, sde.manifold)
        div_score = div_fn(y_t, t, context, epsilon)
        sq_norm_score = sde.manifold.metric.squared_norm(score, y_t)
        losses = 0.5 * sq_norm_score + div_score

        if like_w:
            g2 = sde.beta_schedule.beta_t(t)
            losses = losses * g2

        loss = mean(losses)
        return loss, new_model_state

    return loss_fn


def get_moser_loss_fn(
    pushforward: MoserFlow,
    model: ParametrisedScoreFunction,
    alpha_m: float,
    alpha_p: float,
    K: int,
    hutchinson_type: str,
    eps: float,
    **kwargs
):
    def loss_fn(
        rng, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        y_0, context = pushforward.transform.inv(batch["data"]), batch["context"]
        model_w_dicts = (model, params, states)

        # log probability term
        rng, step_rng = random.split(rng)
        mu_plus = pushforward.mu_plus(
            y_0, context, model_w_dicts, hutchinson_type, step_rng
        )
        log_prob = mean(log(mu_plus))

        # regularization term
        rng, step_rng = random.split(rng)
        ys = pushforward.base.sample(step_rng, (K,))
        prior_prob = pushforward.nu(ys)

        rng, step_rng = random.split(rng)
        mu_minus = pushforward.mu_minus(
            ys, context, model_w_dicts, hutchinson_type, step_rng
        )
        volume_m = mean(batch_mul(mu_minus, 1 / prior_prob), axis=0)
        penalty = alpha_m * volume_m  # + alpha_p * volume_p

        loss = -log_prob + penalty

        return loss, states

    return loss_fn
