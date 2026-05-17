"""Optax-driven training loop and diagnostics for the Sobolev student."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .dataset import SobolevDataset, sample_minibatch, select
from .losses import LossWeights, value_and_grad_total
from .model import Student, f_arith, f_arith_batched, f_energy_batched


@dataclass
class TrainConfig:
    """Hyperparameters for :func:`train_student`."""

    epochs: int = 200
    batch_size: int = 1024
    lr_init: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lattice_frac: float = 0.25
    chebyshev_frac: float = 0.5
    weights: LossWeights = field(default_factory=LossWeights)
    log_every: int = 20
    seed: int = 0


@dataclass
class DiagnosticsReport:
    """Snapshot of evaluation metrics on the full mesh."""

    node_value_mse: float
    node_grad_mse: float
    node_grad_angle_deg: float
    offlattice_value_mse: float
    energy_value_mse: float
    energy_pd_auroc: float

    def as_dict(self) -> dict[str, float]:
        return {
            "node_value_mse": self.node_value_mse,
            "node_grad_mse": self.node_grad_mse,
            "node_grad_angle_deg": self.node_grad_angle_deg,
            "offlattice_value_mse": self.offlattice_value_mse,
            "energy_value_mse": self.energy_value_mse,
            "energy_pd_auroc": self.energy_pd_auroc,
        }


def make_step(
    optimizer: optax.GradientTransformation,
    weights: LossWeights | None = None,
) -> Callable[[Student, optax.OptState, dict[str, jnp.ndarray]],
              tuple[Student, optax.OptState, jnp.ndarray, dict[str, jnp.ndarray]]]:
    """Return a ``jit``-compiled training step.

    The step closure captures ``optimizer`` and the loss ``weights`` so the
    JIT cache stays warm across epochs.  It does **not** carry the dataset; the
    caller passes a freshly-sliced ``batch`` dict (see :func:`dataset.select`)
    on every call.
    """
    weights = weights or LossWeights()
    grad_fn = value_and_grad_total(weights)

    @eqx.filter_jit
    def _step(student: Student, opt_state: optax.OptState, batch: dict[str, jnp.ndarray]):
        (loss, aux), grads = grad_fn(student, batch)
        updates, opt_state = optimizer.update(
            grads, opt_state, params=eqx.filter(student, eqx.is_inexact_array)
        )
        student = eqx.apply_updates(student, updates)
        return student, opt_state, loss, aux

    return _step


def _build_optimizer(cfg: TrainConfig, n_steps: int) -> optax.GradientTransformation:
    schedule = optax.cosine_decay_schedule(
        init_value=cfg.lr_init,
        decay_steps=max(1, n_steps),
        alpha=max(cfg.lr_min / max(cfg.lr_init, 1e-30), 1e-9),
    )
    chain = []
    if cfg.grad_clip > 0:
        chain.append(optax.clip_by_global_norm(cfg.grad_clip))
    if cfg.weight_decay > 0:
        chain.append(optax.adamw(schedule, weight_decay=cfg.weight_decay))
    else:
        chain.append(optax.adam(schedule))
    return optax.chain(*chain)


def evaluate_diagnostics(
    student: Student,
    dataset: SobolevDataset,
) -> DiagnosticsReport:
    """Compute mesh-wide diagnostics in normalised coordinates.

    All errors are reported in **normalised** units; convert back via
    ``dataset.norm`` to compare against raw node values if required.
    """
    xy = dataset.xy
    is_node = np.asarray(dataset.is_node)
    is_pd = np.asarray(dataset.is_pd)

    pred_v = np.asarray(f_arith_batched(student, xy))
    target_v = np.asarray(dataset.V)

    mse_all = (pred_v - target_v) ** 2
    node_value_mse = float(mse_all[is_node].mean()) if is_node.any() else float("nan")
    offlattice_value_mse = (
        float(mse_all[~is_node].mean()) if (~is_node).any() else float("nan")
    )

    # Gradient evaluation only at lattice nodes (where the slope target is the
    # most semantically meaningful Birkhoff signal).
    if is_node.any():
        grad_fn = jax.vmap(jax.grad(lambda inp: f_arith(student, inp)))
        pred_g = np.asarray(grad_fn(xy[is_node]))
        target_g = np.stack(
            [np.asarray(dataset.GX)[is_node], np.asarray(dataset.GY)[is_node]],
            axis=-1,
        )
        node_grad_mse = float(((pred_g - target_g) ** 2).sum(axis=-1).mean())
        cos = (
            (pred_g * target_g).sum(axis=-1)
            / (
                np.linalg.norm(pred_g, axis=-1) * np.linalg.norm(target_g, axis=-1)
                + 1e-12
            )
        )
        cos = np.clip(cos, -1.0, 1.0)
        node_grad_angle_deg = float(np.degrees(np.arccos(cos)).mean())
    else:
        node_grad_mse = float("nan")
        node_grad_angle_deg = float("nan")

    pred_e = np.asarray(f_energy_batched(student, xy))
    pred_em = pred_e[:, 0]
    pred_logit = pred_e[:, 1]
    energy_value_mse = float(((pred_em - np.asarray(dataset.V_M)) ** 2).mean())

    # AUROC for PD vs non-PD over the full mesh (treats off-lattice as
    # negatives; lattice nodes that are PD are positives).  Manhattan-sweep
    # AUROC implementation avoids a sklearn dependency.
    energy_pd_auroc = _auroc(pred_logit, is_pd.astype(float))

    return DiagnosticsReport(
        node_value_mse=node_value_mse,
        node_grad_mse=node_grad_mse,
        node_grad_angle_deg=node_grad_angle_deg,
        offlattice_value_mse=offlattice_value_mse,
        energy_value_mse=energy_value_mse,
        energy_pd_auroc=energy_pd_auroc,
    )


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U / pairwise-AUROC, NaN if degenerate.

    Uses the standard rank-based formula:

    ``AUROC = (sum_of_positive_ranks - n_pos * (n_pos + 1) / 2) /
              (n_pos * n_neg)``.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    n_pos = float(labels.sum())
    n_neg = float(labels.size) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    pos_ranks_sum = float(ranks[labels > 0.5].sum())
    auroc = (pos_ranks_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auroc)


def train_student(
    student: Student,
    dataset: SobolevDataset,
    cfg: TrainConfig | None = None,
    *,
    log_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> tuple[Student, list[dict[str, float]]]:
    """Train ``student`` on ``dataset`` with the configured Sobolev + energy loss.

    Returns the trained student and a per-log-step history of metrics.  The
    optional ``log_callback`` receives ``(epoch, metrics_dict)`` after every
    ``cfg.log_every`` epochs (useful for live notebook plots).

    Two-phase value-grad warmup recipe
    ----------------------------------
    For Birkhoff teachers with non-trivial slope data the gradient term tends
    to dominate the loss early and pulls the value head off-target.  A clean
    fix is to call ``train_student`` twice with different :class:`LossWeights`:

    >>> from dataclasses import replace
    >>> warmup = replace(cfg, epochs=300,
    ...     weights=replace(cfg.weights, grad=0.0, hess_reg=0.0))
    >>> student, h1 = train_student(student, dataset, warmup)
    >>> full = replace(cfg, epochs=800)            # restore full Sobolev
    >>> student, h2 = train_student(student, dataset, full)
    >>> history = h1 + h2

    The first phase fits the value field, the second phase aligns gradients
    (and Hessian regularisation) without disturbing the value baseline.  This
    is materially more robust than a single run with all weights active from
    epoch 0.
    """
    cfg = cfg or TrainConfig()
    n_steps = max(1, cfg.epochs)
    optimizer = _build_optimizer(cfg, n_steps)
    opt_state = optimizer.init(eqx.filter(student, eqx.is_inexact_array))
    step_fn = make_step(optimizer, cfg.weights)

    key = jax.random.PRNGKey(cfg.seed)
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        key, sub_key = jax.random.split(key)
        idx = sample_minibatch(
            sub_key,
            dataset,
            cfg.batch_size,
            lattice_frac=cfg.lattice_frac,
            chebyshev_frac=cfg.chebyshev_frac,
        )
        batch = select(dataset, idx)
        student, opt_state, loss, aux = step_fn(student, opt_state, batch)

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            metrics = {
                "epoch": epoch,
                "loss": float(loss),
                **{k: float(v) for k, v in aux.items() if v.ndim == 0},
            }
            history.append(metrics)
            if log_callback is not None:
                log_callback(epoch, metrics)

    return student, history


__all__ = [
    "DiagnosticsReport",
    "TrainConfig",
    "evaluate_diagnostics",
    "make_step",
    "train_student",
]
