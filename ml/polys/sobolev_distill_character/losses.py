"""Sobolev + energy losses for the character JAX student.

The arithmetic head returns ``(re, im)`` per point.  Per-channel value /
gradient / Hessian are obtained by index-selecting the ``c``-th component
before differentiation, then stacking the two channels.

Resulting shapes per batch ``B``:

- ``pred_v``: ``(B, 2)`` -- (re, im) values
- ``pred_g``: ``(B, 2, 2)`` -- ``pred_g[b, c, d] = d/dx_d (re_or_im_c)(xy_b)``
- ``pred_h``: ``(B, 2, 2, 2)`` -- second derivatives of the c-th component

The Sobolev loss is the sum of per-channel MSEs across the 12 real teacher
fields (2 value, 4 grad, 6 Hessian).  Energy and PD losses are unchanged
from :mod:`sobolev_distill.losses` (the energy field ``V_M`` is real).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .model import CharacterStudent, f_arith_character, f_energy_character


@dataclass(frozen=True)
class LossWeights:
    """Coefficients in front of the individual loss terms.

    Same knobs as :class:`sobolev_distill.losses.LossWeights`, applied
    identically (the Sobolev terms aggregate over both channels before
    weighting), plus a character-specific ``unit_circle`` term.

    ``unit_circle`` adds a magnitude-matching penalty
    ``mean((|pred|^2 - |target|^2)^2)`` that pushes the head's output to
    track the teacher's modulus.  At lattice nodes the teacher lives on
    the unit circle (in raw coords), so this term provides the angular-
    decoder pressure that the bare per-component MSE lacks.  Recommended
    range ``0.1 - 1.0`` (tune via the workbook).

    ``axis`` adds an auxiliary linear-probe loss
    ``mean(||axis_probe(trunk(xy)) - axis_target||^2)`` that pushes the
    trunk to expose linearly decodable per-axis Fourier features
    ``(cos 2 pi x_raw/p, sin 2 pi x_raw/p, cos 2 pi y_raw/p,
    sin 2 pi y_raw/p)``.  Active only when the student carries a
    non-``None`` ``axis_probe``; otherwise the term is silently skipped.
    Recommended range ``0.1 - 1.0``.

    Registered as a JAX pytree (all 9 floats are data leaves) so callers
    can wrap each field as a ``jnp`` scalar and thread the dataclass
    through ``jax.jit`` as a *traced* runtime argument.  This is what
    :func:`train_student_character_scheduled` does to swap weights per
    epoch without retriggering compilation.  Constructing weights with
    plain Python floats (the default) keeps the original static-folded
    behaviour intact.
    """

    value: float = 1.0
    grad: float = 0.1
    hess: float = 0.0
    hess_reg: float = 0.0
    unit_circle: float = 0.0
    axis: float = 0.0
    energy_value: float = 0.5
    energy_pd: float = 0.1
    pd_pos_weight: float = 1.0


jax.tree_util.register_dataclass(
    LossWeights,
    data_fields=[
        "value",
        "grad",
        "hess",
        "hess_reg",
        "unit_circle",
        "axis",
        "energy_value",
        "energy_pd",
        "pd_pos_weight",
    ],
    meta_fields=[],
)


def _per_point_per_channel_value_grad_hess(
    student: CharacterStudent, xy: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (value (2,), grad (2, 2), hess (2, 2, 2)) at one point."""

    def channel(c: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
        return lambda inp: f_arith_character(student, inp)[c]

    val = f_arith_character(student, xy)
    g0 = jax.grad(channel(0))(xy)
    g1 = jax.grad(channel(1))(xy)
    h0 = jax.hessian(channel(0))(xy)
    h1 = jax.hessian(channel(1))(xy)
    g = jnp.stack([g0, g1], axis=0)            # (2, 2)
    h = jnp.stack([h0, h1], axis=0)            # (2, 2, 2)
    return val, g, h


_batched_value_grad_hess = jax.vmap(
    _per_point_per_channel_value_grad_hess, in_axes=(None, 0)
)


_batched_energy = jax.vmap(
    lambda student, xy: f_energy_character(student, xy),
    in_axes=(None, 0),
)


def sobolev_loss_character(
    student: CharacterStudent,
    batch: dict[str, jnp.ndarray],
    weights: LossWeights = LossWeights(),
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """MSE on (value, grad, hess) summed over Re and Im channels."""
    xy = batch["xy"]
    pred_v, pred_g, pred_h = _batched_value_grad_hess(student, xy)
    # pred_v: (B, 2);  pred_g: (B, 2, 2);  pred_h: (B, 2, 2, 2)

    target_v = jnp.stack([batch["V_re"], batch["V_im"]], axis=-1)              # (B, 2)
    target_g = jnp.stack(
        [
            jnp.stack([batch["GX_re"], batch["GY_re"]], axis=-1),
            jnp.stack([batch["GX_im"], batch["GY_im"]], axis=-1),
        ],
        axis=-2,
    )                                                                            # (B, 2, 2)
    target_h_re = jnp.stack(
        [
            jnp.stack([batch["Hxx_re"], batch["Hxy_re"]], axis=-1),
            jnp.stack([batch["Hxy_re"], batch["Hyy_re"]], axis=-1),
        ],
        axis=-2,
    )                                                                            # (B, 2, 2)
    target_h_im = jnp.stack(
        [
            jnp.stack([batch["Hxx_im"], batch["Hxy_im"]], axis=-1),
            jnp.stack([batch["Hxy_im"], batch["Hyy_im"]], axis=-1),
        ],
        axis=-2,
    )
    target_h = jnp.stack([target_h_re, target_h_im], axis=-3)                    # (B, 2, 2, 2)

    val_mse = jnp.mean((pred_v - target_v) ** 2)
    grad_mse = jnp.mean(jnp.sum((pred_g - target_g) ** 2, axis=(-2, -1)))
    hess_mse = jnp.mean(jnp.sum((pred_h - target_h) ** 2, axis=(-3, -2, -1)))
    hess_reg = jnp.mean(jnp.sum(pred_h ** 2, axis=(-3, -2, -1)))
    # Magnitude-matching: |pred|^2 vs |target|^2 in normalised coords.  At
    # lattice nodes the teacher lies on the unit circle in raw coords, so
    # this term aligns the head's output magnitude with the unit circle
    # without competing with the angular content of val_mse / grad_mse.
    pred_mag_sq = jnp.sum(pred_v ** 2, axis=-1)
    target_mag_sq = jnp.sum(target_v ** 2, axis=-1)
    unit_circle_pen = jnp.mean((pred_mag_sq - target_mag_sq) ** 2)

    # Auxiliary axis loss: linear probe from trunk(xy) to the four
    # canonical axis Fourier targets.  Skipped (zero) when no axis probe
    # is attached; this keeps the JIT graph shape-stable across variants.
    # ``getattr`` supports duck-typed students (e.g. notebook-only variants)
    # that omit ``axis_probe`` entirely.
    axis_probe = getattr(student, "axis_probe", None)
    if axis_probe is not None and "axis_target" in batch:
        trunk_fn = jax.vmap(lambda x: student.trunk(x))
        H = trunk_fn(xy)
        probe_fn = jax.vmap(axis_probe)
        axis_pred = probe_fn(H)
        axis_target = batch["axis_target"]
        axis_mse = jnp.mean(jnp.sum((axis_pred - axis_target) ** 2, axis=-1))
    else:
        axis_mse = jnp.asarray(0.0, dtype=val_mse.dtype)

    loss = (
        weights.value * val_mse
        + weights.grad * grad_mse
        + weights.hess * hess_mse
        + weights.hess_reg * hess_reg
        + weights.unit_circle * unit_circle_pen
        + weights.axis * axis_mse
    )
    aux = {
        "sobolev_value_mse": val_mse,
        "sobolev_grad_mse": grad_mse,
        "sobolev_hess_mse": hess_mse,
        "sobolev_hess_reg": hess_reg,
        "sobolev_unit_circle_pen": unit_circle_pen,
        "sobolev_axis_mse": axis_mse,
    }
    return loss, aux


def _binary_cross_entropy_with_logits(
    logits: jnp.ndarray, targets: jnp.ndarray
) -> jnp.ndarray:
    log_p = jax.nn.log_sigmoid(logits)
    log_1_p = jax.nn.log_sigmoid(-logits)
    return -(targets * log_p + (1.0 - targets) * log_1_p)


def energy_loss_character(
    student: CharacterStudent,
    batch: dict[str, jnp.ndarray],
    weights: LossWeights = LossWeights(),
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """MSE on ``V_M`` plus weighted BCE on ``is_pd`` (real-valued, identical to the real path)."""
    xy = batch["xy"]
    pred = _batched_energy(student, xy)        # (B, 2)
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


def total_loss_character(
    student: CharacterStudent,
    batch: dict[str, jnp.ndarray],
    weights: LossWeights = LossWeights(),
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    sob, sob_aux = sobolev_loss_character(student, batch, weights)
    eng, eng_aux = energy_loss_character(student, batch, weights)
    loss = sob + eng
    aux = {**sob_aux, **eng_aux, "total_loss": loss, "sobolev_loss": sob, "energy_loss": eng}
    return loss, aux


def value_and_grad_total_character(
    weights: LossWeights = LossWeights(),
) -> Callable[[CharacterStudent, dict[str, jnp.ndarray]],
              tuple[tuple[jnp.ndarray, dict[str, jnp.ndarray]], CharacterStudent]]:
    def _wrapped(student: CharacterStudent, batch: dict[str, jnp.ndarray]):
        return total_loss_character(student, batch, weights)

    return jax.value_and_grad(_wrapped, has_aux=True)


def value_and_grad_total_character_dynamic() -> Callable[
    [CharacterStudent, dict[str, jnp.ndarray], LossWeights],
    tuple[tuple[jnp.ndarray, dict[str, jnp.ndarray]], CharacterStudent],
]:
    """Variant of :func:`value_and_grad_total_character` that takes ``weights``
    as a runtime (traced) argument instead of capturing it in a closure.

    Pair with :func:`~sobolev_distill_character.train.make_step_character_dynamic`
    when the loss weights need to change per step (e.g. linear ramp).
    Wrapping the ``LossWeights`` fields as ``jnp`` scalars before calling
    keeps a single JIT cache hit across all weight values.
    """

    def _wrapped(
        student: CharacterStudent,
        batch: dict[str, jnp.ndarray],
        weights: LossWeights,
    ):
        return total_loss_character(student, batch, weights)

    return jax.value_and_grad(_wrapped, argnums=0, has_aux=True)


__all__ = [
    "LossWeights",
    "energy_loss_character",
    "sobolev_loss_character",
    "total_loss_character",
    "value_and_grad_total_character",
    "value_and_grad_total_character_dynamic",
]
