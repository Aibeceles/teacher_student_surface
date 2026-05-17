"""Sobolev + energy losses for the JAX student.

The arithmetic head is supervised against ``f_H``'s value, gradient, and
Hessian (Sobolev triple).  The energy head is supervised against ``f_M``'s
value plus a binary positive-definiteness label at lattice nodes.

All losses are pure functions consuming a flat ``batch`` dict (as produced by
:func:`dataset.select`).  They are designed to be wrapped by
``jax.value_and_grad(..., has_aux=True)``; auxiliary metrics are returned in a
plain dict so the training loop can log without re-running.

Conventions:

- ``xy`` is a ``(B, 2)`` batch of normalised input coordinates.
- ``V``, ``GX``, ``GY``, ``Hxx``, ``Hxy``, ``Hyy``, ``V_M`` are ``(B,)``
  normalised teacher targets (see :mod:`dataset`).
- ``is_node`` and ``is_pd`` are ``(B,)`` boolean masks that activate the BCE
  term only on lattice samples.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .model import Student, f_arith, f_energy


@dataclass(frozen=True)
class LossWeights:
    """Coefficients in front of the individual loss terms.

    Defaults turn the Hessian distillation term off (``hess=0.0``).  Reason: in
    the Birkhoff-Hermite construction the polynomial Hessian at lattice nodes
    is not constrained to small magnitudes and can be very large at high node
    counts.  Distilling those values teaches nothing about arithmetic and
    destabilises training.  Set ``hess > 0`` only when you have explicitly
    capped Hessian targets (e.g. small ``K`` and small ``hess_clip``).

    ``hess_reg`` is a *self*-regularisation term: it penalises
    ``mean(||H g_theta||_F^2)`` of the **student's own** Hessian, with no
    teacher target involved.  This borrows the curvature-shaping spirit of
    ``f_M`` without contaminating the gradient field, smoothing the student
    between lattice nodes.  Typical scale: ``1e-3`` for SIREN trunks,
    ``1e-4`` for GELU trunks.

    ``pd_pos_weight`` rebalances the BCE term in :func:`energy_loss` to handle
    the lattice-vs-not class imbalance (e.g. 144/5371 -> ``pos_weight ~= 37``).

    The Czarnecki et al. (2017) Sobolev-training recipe (decreasing weight
    with derivative order) is recovered by setting e.g.
    ``LossWeights(value=1.0, grad=0.1, hess=0.01)``.
    """

    value: float = 1.0
    grad: float = 0.1
    hess: float = 0.0
    hess_reg: float = 0.0
    energy_value: float = 0.5
    energy_pd: float = 0.1
    pd_pos_weight: float = 1.0


def _per_point_value_grad_hess(
    student: Student, xy: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (value, grad (2,), hess (2, 2)) for the arithmetic head at one point.

    ``f_arith`` returns a scalar so ``jax.grad`` and ``jax.hessian`` give the
    natural shapes without ``jacrev`` gymnastics.
    """
    f = lambda inp: f_arith(student, inp)  # noqa: E731  (intentional closure)
    val = f(xy)
    g = jax.grad(f)(xy)
    h = jax.hessian(f)(xy)
    return val, g, h


_batched_value_grad_hess = jax.vmap(_per_point_value_grad_hess, in_axes=(None, 0))


def _per_point_energy(
    student: Student, xy: jnp.ndarray
) -> jnp.ndarray:
    """Return ``(2,)``: (energy_value, pd_logit) at one point."""
    return f_energy(student, xy)


_batched_energy = jax.vmap(_per_point_energy, in_axes=(None, 0))


def sobolev_loss(
    student: Student,
    batch: dict[str, jnp.ndarray],
    weights: LossWeights = LossWeights(),
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """MSE on (value, grad, hess) of the arithmetic head against ``f_H``.

    Returns ``(loss, aux)`` where ``aux`` carries the per-component MSE for
    diagnostics.
    """
    xy = batch["xy"]
    pred_v, pred_g, pred_h = _batched_value_grad_hess(student, xy)

    target_v = batch["V"]
    target_g = jnp.stack([batch["GX"], batch["GY"]], axis=-1)             # (B, 2)
    target_h = jnp.stack(
        [
            jnp.stack([batch["Hxx"], batch["Hxy"]], axis=-1),
            jnp.stack([batch["Hxy"], batch["Hyy"]], axis=-1),
        ],
        axis=-2,
    )                                                                       # (B, 2, 2)

    val_mse = jnp.mean((pred_v - target_v) ** 2)
    grad_mse = jnp.mean(jnp.sum((pred_g - target_g) ** 2, axis=-1))
    hess_mse = jnp.mean(jnp.sum((pred_h - target_h) ** 2, axis=(-2, -1)))
    # Self-regularisation: Frobenius norm of the student's own Hessian.
    # Reuses ``pred_h`` so it costs nothing extra beyond the existing
    # ``jax.hessian`` call.
    hess_reg = jnp.mean(jnp.sum(pred_h ** 2, axis=(-2, -1)))

    loss = (
        weights.value * val_mse
        + weights.grad * grad_mse
        + weights.hess * hess_mse
        + weights.hess_reg * hess_reg
    )
    aux = {
        "sobolev_value_mse": val_mse,
        "sobolev_grad_mse": grad_mse,
        "sobolev_hess_mse": hess_mse,
        "sobolev_hess_reg": hess_reg,
    }
    return loss, aux


def _binary_cross_entropy_with_logits(
    logits: jnp.ndarray, targets: jnp.ndarray
) -> jnp.ndarray:
    """Numerically stable per-element BCE."""
    log_p = jax.nn.log_sigmoid(logits)
    log_1_p = jax.nn.log_sigmoid(-logits)
    return -(targets * log_p + (1.0 - targets) * log_1_p)


def energy_loss(
    student: Student,
    batch: dict[str, jnp.ndarray],
    weights: LossWeights = LossWeights(),
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """MSE on ``f_M`` plus weighted BCE on ``is_pd``.

    The BCE term applies to the **whole batch** (not only lattice nodes) so the
    classifier learns ``is at a lattice node?`` as a binary geometric task.
    Positive samples (``is_pd == True``) are upweighted by
    ``weights.pd_pos_weight`` to compensate for class imbalance.  The
    normalisation divides by ``pos_weight * n_pos + n_neg`` so the term has a
    consistent scale regardless of imbalance.
    """
    xy = batch["xy"]
    pred = _batched_energy(student, xy)            # (B, 2)
    pred_v = pred[:, 0]
    pred_logit = pred[:, 1]

    energy_mse = jnp.mean((pred_v - batch["V_M"]) ** 2)

    pd_target = batch["is_pd"].astype(pred_logit.dtype)
    bce = _binary_cross_entropy_with_logits(pred_logit, pd_target)
    pos_w = jnp.asarray(weights.pd_pos_weight, dtype=pred_logit.dtype)
    sample_w = pd_target * pos_w + (1.0 - pd_target)
    weighted_bce = bce * sample_w
    norm = pos_w * jnp.sum(pd_target) + jnp.sum(1.0 - pd_target)
    pd_loss = jnp.sum(weighted_bce) / jnp.maximum(norm, 1.0)

    loss = weights.energy_value * energy_mse + weights.energy_pd * pd_loss
    aux = {
        "energy_value_mse": energy_mse,
        "energy_pd_bce": pd_loss,
        "energy_pd_n_pos": jnp.sum(pd_target),
    }
    return loss, aux


def total_loss(
    student: Student,
    batch: dict[str, jnp.ndarray],
    weights: LossWeights = LossWeights(),
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Sobolev (head A) + energy (head B), summed.

    The aux dict contains both component dicts plus ``total_loss`` for logging.
    """
    sob, sob_aux = sobolev_loss(student, batch, weights)
    eng, eng_aux = energy_loss(student, batch, weights)
    loss = sob + eng
    aux = {**sob_aux, **eng_aux, "total_loss": loss, "sobolev_loss": sob, "energy_loss": eng}
    return loss, aux


def value_and_grad_total(
    weights: LossWeights = LossWeights(),
) -> Callable[[Student, dict[str, jnp.ndarray]],
              tuple[tuple[jnp.ndarray, dict[str, jnp.ndarray]], Student]]:
    """Return ``jax.value_and_grad(total_loss, has_aux=True)`` bound to ``weights``.

    Convenience wrapper so the training loop does not need to recreate the
    closure on every step (which would defeat ``jit`` cache reuse).
    """
    def _wrapped(student: Student, batch: dict[str, jnp.ndarray]):
        return total_loss(student, batch, weights)

    return jax.value_and_grad(_wrapped, has_aux=True)


__all__ = [
    "LossWeights",
    "energy_loss",
    "sobolev_loss",
    "total_loss",
    "value_and_grad_total",
]
