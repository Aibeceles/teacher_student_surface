"""Optax training loop and diagnostics for the character Sobolev student.

Mirrors :mod:`sobolev_distill.train` with two added diagnostics:

- ``modular_recovery_accuracy`` -- top-1 accuracy of the modular addition
  table at lattice nodes, decoded as
  ``round(p * atan2(im_pred, re_pred) / (2*pi)) mod p`` vs ``(i+j) mod p``.
- ``modular_recovery_accuracy_train`` / ``_holdout`` -- same metric on a
  deterministic 80/20 split of lattice nodes (evaluation-only; training
  still samples the full mesh unless you add an explicit holdout mask).
  If the model fits every pair equally, both can match and the gap stays
  ~0 even under memorisation; the split mainly catches uneven fit.
- ``unit_circle_residual`` -- mean ``||(re_pred, im_pred)||_2 - 1`` at
  lattice nodes (sub-test of the modular recovery; the prediction lying
  near the unit circle is necessary for argument extraction).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from sobolev_distill.train import _auroc

from .dataset import CharacterSobolevDataset, sample_minibatch, select_character
from .losses import (
    LossWeights,
    value_and_grad_total_character,
    value_and_grad_total_character_dynamic,
)
from .model import (
    CharacterStudent,
    f_arith_character_batched,
    f_energy_character_batched,
)


@dataclass
class CharacterTrainConfig:
    """Hyperparameters for :func:`train_student_character`."""

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
class CharacterDiagnosticsReport:
    """Snapshot of evaluation metrics on the full mesh."""

    node_value_mse: float
    node_grad_mse: float
    offlattice_value_mse: float
    energy_value_mse: float
    energy_pd_auroc: float
    modular_recovery_accuracy: float
    modular_recovery_accuracy_train: float
    modular_recovery_accuracy_holdout: float
    modular_acc_train_minus_holdout: float
    unit_circle_residual: float

    def as_dict(self) -> dict[str, float]:
        return {
            "node_value_mse": self.node_value_mse,
            "node_grad_mse": self.node_grad_mse,
            "offlattice_value_mse": self.offlattice_value_mse,
            "energy_value_mse": self.energy_value_mse,
            "energy_pd_auroc": self.energy_pd_auroc,
            "modular_recovery_accuracy": self.modular_recovery_accuracy,
            "modular_recovery_accuracy_train": self.modular_recovery_accuracy_train,
            "modular_recovery_accuracy_holdout": self.modular_recovery_accuracy_holdout,
            "modular_acc_train_minus_holdout": self.modular_acc_train_minus_holdout,
            "unit_circle_residual": self.unit_circle_residual,
        }


def make_step_character(
    optimizer: optax.GradientTransformation,
    weights: LossWeights | None = None,
) -> Callable[[CharacterStudent, optax.OptState, dict[str, jnp.ndarray]],
              tuple[CharacterStudent, optax.OptState, jnp.ndarray, dict[str, jnp.ndarray]]]:
    """JIT-compiled training step for the character student."""
    weights = weights or LossWeights()
    grad_fn = value_and_grad_total_character(weights)

    @eqx.filter_jit
    def _step(student: CharacterStudent, opt_state: optax.OptState, batch: dict[str, jnp.ndarray]):
        (loss, aux), grads = grad_fn(student, batch)
        updates, opt_state = optimizer.update(
            grads, opt_state, params=eqx.filter(student, eqx.is_inexact_array)
        )
        student = eqx.apply_updates(student, updates)
        return student, opt_state, loss, aux

    return _step


def make_step_character_dynamic(
    optimizer: optax.GradientTransformation,
) -> Callable[
    [CharacterStudent, optax.OptState, dict[str, jnp.ndarray], LossWeights],
    tuple[CharacterStudent, optax.OptState, jnp.ndarray, dict[str, jnp.ndarray]],
]:
    """JIT-compiled step that takes ``weights`` as a runtime (traced) arg.

    Pass a :class:`LossWeights` whose 9 fields are ``jnp`` scalars (e.g.
    via ``jax.tree.map(jnp.asarray, weights)``) so the JIT cache stays warm
    across changing weight values.  See
    :func:`train_student_character_scheduled` for the canonical caller.
    """
    grad_fn = value_and_grad_total_character_dynamic()

    @eqx.filter_jit
    def _step(
        student: CharacterStudent,
        opt_state: optax.OptState,
        batch: dict[str, jnp.ndarray],
        weights: LossWeights,
    ):
        (loss, aux), grads = grad_fn(student, batch, weights)
        updates, opt_state = optimizer.update(
            grads, opt_state, params=eqx.filter(student, eqx.is_inexact_array)
        )
        student = eqx.apply_updates(student, updates)
        return student, opt_state, loss, aux

    return _step


def _build_optimizer(cfg: CharacterTrainConfig, n_steps: int) -> optax.GradientTransformation:
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


def _denormalised_lattice_complex(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    lattice_indices: np.ndarray,
) -> np.ndarray:
    """Return ``(K_lat,)`` complex predictions in raw (un-normalised) units."""
    pred_norm = np.asarray(f_arith_character_batched(student, dataset.xy[lattice_indices]))
    re = pred_norm[:, 0] * dataset.norm.v_re_std + dataset.norm.v_re_mean
    im = pred_norm[:, 1] * dataset.norm.v_im_std + dataset.norm.v_im_mean
    return re + 1j * im


def _lattice_modular_truth_recovered(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    teacher: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lattice rows only, order of ``np.where(dataset.is_node)[0]``.

    Returns ``lat_idx, i_idx, j_idx, truth, recovered, complex_pred`` as
    ``(K,)`` int/float/complex arrays.
    """
    is_node = np.asarray(dataset.is_node)
    lat_idx = np.where(is_node)[0]
    p = int(dataset.modulus)
    nodes_x = np.asarray(teacher.nodes_x)
    nodes_y = np.asarray(teacher.nodes_y)
    xy_lat_raw = np.asarray(dataset.xy_raw)[is_node]
    i_idx = np.argmin(np.abs(xy_lat_raw[:, 0:1] - nodes_x[None, :]), axis=1)
    j_idx = np.argmin(np.abs(xy_lat_raw[:, 1:2] - nodes_y[None, :]), axis=1)
    complex_pred = _denormalised_lattice_complex(student, dataset, lat_idx)
    recovered = (np.round(p * np.angle(complex_pred) / (2.0 * math.pi)).astype(int)) % p
    truth = (i_idx.astype(int) + j_idx.astype(int)) % p
    return lat_idx, i_idx, j_idx, truth, recovered, complex_pred


def _split_lattice_train_holdout(
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    *,
    holdout_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 80/20-style split: sort by ``(i, j)``, last holdout slice.

    If ``N < 5``, holdout is empty (all train) so holdout metrics are ``nan``
    downstream.
    """
    n = int(i_idx.shape[0])
    train_m = np.ones(n, dtype=bool)
    hold_m = np.zeros(n, dtype=bool)
    if n < 5:
        return train_m, hold_m
    order = np.lexsort((j_idx.astype(np.int64), i_idx.astype(np.int64)))
    n_hold = int(np.ceil(float(holdout_frac) * n))
    n_hold = min(max(n_hold, 1), n - 1)
    hold_rows = set(order[-n_hold:].tolist())
    train_m = np.ones(n, dtype=bool)
    hold_m = np.zeros(n, dtype=bool)
    for r in hold_rows:
        train_m[r] = False
        hold_m[r] = True
    return train_m, hold_m


def evaluate_diagnostics_character(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    teacher: Any | None = None,
) -> CharacterDiagnosticsReport:
    """Mesh diagnostics in normalised coords plus modular-recovery accuracy.

    ``teacher`` (a :class:`CharacterMeshTeacher`) is needed only for the
    lattice index map ``(i_idx, j_idx)`` used to compute the modular
    recovery target ``(i+j) mod p``.  When ``None``, the modular fields
    are returned as ``nan`` and the report still contains every other
    metric.
    """
    xy = dataset.xy
    is_node = np.asarray(dataset.is_node)
    is_pd = np.asarray(dataset.is_pd)

    pred = np.asarray(f_arith_character_batched(student, xy))      # (N, 2)
    target = np.stack([np.asarray(dataset.V_re), np.asarray(dataset.V_im)], axis=-1)
    mse_all = ((pred - target) ** 2).sum(axis=-1)
    node_value_mse = float(mse_all[is_node].mean()) if is_node.any() else float("nan")
    offlattice_value_mse = (
        float(mse_all[~is_node].mean()) if (~is_node).any() else float("nan")
    )

    if is_node.any():
        # Per-channel grad at lattice nodes: shape (K_lat, 2, 2)
        def _channel(c: int):
            return jax.vmap(jax.grad(lambda inp: pred_channel_fn(student, inp, c)))

        # We need a closure-free version of pred_channel_fn for vmap.  Define
        # inline against student/c via lambdas.
        from .model import f_arith_character

        def _grad_c0(s, x):
            return jax.grad(lambda inp: f_arith_character(s, inp)[0])(x)

        def _grad_c1(s, x):
            return jax.grad(lambda inp: f_arith_character(s, inp)[1])(x)

        g0 = np.asarray(jax.vmap(_grad_c0, in_axes=(None, 0))(student, xy[is_node]))
        g1 = np.asarray(jax.vmap(_grad_c1, in_axes=(None, 0))(student, xy[is_node]))
        target_g0 = np.stack(
            [np.asarray(dataset.GX_re)[is_node], np.asarray(dataset.GY_re)[is_node]],
            axis=-1,
        )
        target_g1 = np.stack(
            [np.asarray(dataset.GX_im)[is_node], np.asarray(dataset.GY_im)[is_node]],
            axis=-1,
        )
        node_grad_mse = float(
            ((g0 - target_g0) ** 2).sum(axis=-1).mean()
            + ((g1 - target_g1) ** 2).sum(axis=-1).mean()
        )
    else:
        node_grad_mse = float("nan")

    pred_e = np.asarray(f_energy_character_batched(student, xy))
    pred_em = pred_e[:, 0]
    pred_logit = pred_e[:, 1]
    energy_value_mse = float(((pred_em - np.asarray(dataset.V_M)) ** 2).mean())
    energy_pd_auroc = _auroc(pred_logit, is_pd.astype(float))

    modular_recovery_accuracy = float("nan")
    modular_recovery_accuracy_train = float("nan")
    modular_recovery_accuracy_holdout = float("nan")
    modular_acc_train_minus_holdout = float("nan")
    unit_circle_residual = float("nan")

    if teacher is not None and is_node.any():
        _, i_idx, j_idx, truth, recovered, complex_pred = _lattice_modular_truth_recovered(
            student, dataset, teacher
        )
        modular_recovery_accuracy = float((recovered == truth).mean())
        unit_circle_residual = float(np.abs(np.abs(complex_pred) - 1.0).mean())

        train_m, hold_m = _split_lattice_train_holdout(i_idx, j_idx)
        if hold_m.any():
            modular_recovery_accuracy_train = float((recovered[train_m] == truth[train_m]).mean())
            modular_recovery_accuracy_holdout = float((recovered[hold_m] == truth[hold_m]).mean())
            modular_acc_train_minus_holdout = (
                modular_recovery_accuracy_train - modular_recovery_accuracy_holdout
            )

    return CharacterDiagnosticsReport(
        node_value_mse=node_value_mse,
        node_grad_mse=node_grad_mse,
        offlattice_value_mse=offlattice_value_mse,
        energy_value_mse=energy_value_mse,
        energy_pd_auroc=energy_pd_auroc,
        modular_recovery_accuracy=modular_recovery_accuracy,
        modular_recovery_accuracy_train=modular_recovery_accuracy_train,
        modular_recovery_accuracy_holdout=modular_recovery_accuracy_holdout,
        modular_acc_train_minus_holdout=modular_acc_train_minus_holdout,
        unit_circle_residual=unit_circle_residual,
    )


# Placeholder; not actually called.  Helps type-check the closure pattern above.
def pred_channel_fn(student: CharacterStudent, xy: jnp.ndarray, c: int) -> jnp.ndarray:  # pragma: no cover
    from .model import f_arith_character

    return f_arith_character(student, xy)[c]


def train_student_character(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    cfg: CharacterTrainConfig | None = None,
    *,
    log_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> tuple[CharacterStudent, list[dict[str, float]]]:
    """Train ``student`` on ``dataset`` with the character Sobolev + energy loss."""
    cfg = cfg or CharacterTrainConfig()
    n_steps = max(1, cfg.epochs)
    optimizer = _build_optimizer(cfg, n_steps)
    opt_state = optimizer.init(eqx.filter(student, eqx.is_inexact_array))
    step_fn = make_step_character(optimizer, cfg.weights)

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
        batch = select_character(dataset, idx)
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


@dataclass(frozen=True)
class LinearRampSchedule:
    """Per-epoch ``LossWeights`` schedule that linearly ramps one field.

    For ``epoch < ramp_epochs`` the field is interpolated from ``start`` to
    ``end``; for ``epoch >= ramp_epochs`` the field is held at ``end``.
    All other fields stay at their ``base`` values.

    Example
    -------
    >>> base = LossWeights(value=1.0, grad=0.0, unit_circle=0.5, axis=1.0)
    >>> sched = LinearRampSchedule(base, "grad", 0.0, 0.05, 200)
    >>> sched(0).grad, sched(100).grad, sched(200).grad, sched(2000).grad
    (0.0, 0.025, 0.05, 0.05)
    """

    base: LossWeights
    field: str
    start: float
    end: float
    ramp_epochs: int

    def __call__(self, epoch: int) -> LossWeights:
        if self.ramp_epochs <= 0 or epoch >= self.ramp_epochs:
            t = 1.0
        else:
            t = float(epoch) / float(self.ramp_epochs)
        v = float(self.start) + (float(self.end) - float(self.start)) * t
        return replace(self.base, **{self.field: v})


def _wrap_weights_jnp(weights: LossWeights) -> LossWeights:
    """Convert all 9 ``LossWeights`` fields to ``jnp`` scalars.

    The pytree registration on :class:`LossWeights` then makes these
    leaves *traced* through ``eqx.filter_jit`` rather than constant-folded,
    which keeps the JIT cache warm when weights change between steps.
    """
    return jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), weights)


def train_student_character_scheduled(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    cfg: CharacterTrainConfig | None = None,
    schedule: Callable[[int], LossWeights] | None = None,
    *,
    log_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> tuple[CharacterStudent, list[dict[str, float]]]:
    """Single-phase training with a per-epoch ``LossWeights`` schedule.

    Mirrors :func:`train_student_character` (same optimizer, same
    minibatch sampler, same logging cadence) but replaces the static
    ``cfg.weights`` with ``schedule(epoch)`` on every step.  Weights are
    wrapped as ``jnp`` scalars and fed through one JIT-compiled step
    closure (:func:`make_step_character_dynamic`), so changing weights
    does *not* trigger recompilation.

    Parameters
    ----------
    schedule
        Callable mapping ``epoch -> LossWeights``.  When ``None`` the
        constant ``cfg.weights`` schedule is used (equivalent to
        :func:`train_student_character` modulo the dynamic-step closure).
    """
    cfg = cfg or CharacterTrainConfig()
    n_steps = max(1, cfg.epochs)
    optimizer = _build_optimizer(cfg, n_steps)
    opt_state = optimizer.init(eqx.filter(student, eqx.is_inexact_array))
    step_fn = make_step_character_dynamic(optimizer)

    if schedule is None:
        const_weights = cfg.weights
        schedule = lambda _epoch, _w=const_weights: _w  # noqa: E731

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
        batch = select_character(dataset, idx)
        weights_t = _wrap_weights_jnp(schedule(epoch))
        student, opt_state, loss, aux = step_fn(student, opt_state, batch, weights_t)

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


def train_student_character_scheduled_with_checkpoints(
    student: CharacterStudent,
    dataset: CharacterSobolevDataset,
    cfg: CharacterTrainConfig | None = None,
    schedule: Callable[[int], LossWeights] | None = None,
    *,
    snapshot_every: int = 50,
    log_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> tuple[
    CharacterStudent,
    list[dict[str, float]],
    list[tuple[int, CharacterStudent]],
]:
    """Scheduled training with periodic in-memory student snapshots.

    Mirrors :func:`train_student_character_scheduled` exactly (same JIT
    step, same minibatch sampler, same per-epoch logging cadence) and
    adds a third return value: a list of ``(epoch, student_pytree)``
    snapshots captured at ``epoch % snapshot_every == 0`` plus the final
    epoch.

    The student is stored as a deep copy via :func:`jax.tree.map` so
    downstream callers can mutate / discard it freely.  This is intended
    for ``EPOCHS_TOTAL <= 5000`` with snapshot dimensions on the order of
    the default ``CharacterStudentConfig`` (~few KB per snapshot).

    Parameters
    ----------
    snapshot_every
        Take a snapshot every ``snapshot_every`` epochs.  Must be ``>= 1``.
        ``snapshot_every == 1`` is allowed but produces one snapshot per
        epoch -- typically only meaningful for short tracing runs.
    """
    cfg = cfg or CharacterTrainConfig()
    if snapshot_every < 1:
        raise ValueError(f"snapshot_every must be >= 1, got {snapshot_every}")
    n_steps = max(1, cfg.epochs)
    optimizer = _build_optimizer(cfg, n_steps)
    opt_state = optimizer.init(eqx.filter(student, eqx.is_inexact_array))
    step_fn = make_step_character_dynamic(optimizer)

    if schedule is None:
        const_weights = cfg.weights
        schedule = lambda _epoch, _w=const_weights: _w  # noqa: E731

    key = jax.random.PRNGKey(cfg.seed)
    history: list[dict[str, float]] = []
    snapshots: list[tuple[int, CharacterStudent]] = []

    def _snapshot(epoch_: int, student_: CharacterStudent) -> None:
        frozen = jax.tree.map(
            lambda leaf: jnp.asarray(leaf) if eqx.is_inexact_array(leaf) else leaf,
            student_,
        )
        snapshots.append((int(epoch_), frozen))

    _snapshot(0, student)

    for epoch in range(cfg.epochs):
        key, sub_key = jax.random.split(key)
        idx = sample_minibatch(
            sub_key,
            dataset,
            cfg.batch_size,
            lattice_frac=cfg.lattice_frac,
            chebyshev_frac=cfg.chebyshev_frac,
        )
        batch = select_character(dataset, idx)
        weights_t = _wrap_weights_jnp(schedule(epoch))
        student, opt_state, loss, aux = step_fn(student, opt_state, batch, weights_t)

        is_last = epoch == cfg.epochs - 1
        if epoch % cfg.log_every == 0 or is_last:
            metrics = {
                "epoch": epoch,
                "loss": float(loss),
                **{k: float(v) for k, v in aux.items() if v.ndim == 0},
            }
            history.append(metrics)
            if log_callback is not None:
                log_callback(epoch, metrics)
        if epoch % snapshot_every == 0 or is_last:
            _snapshot(epoch, student)

    return student, history, snapshots


__all__ = [
    "CharacterDiagnosticsReport",
    "CharacterTrainConfig",
    "LinearRampSchedule",
    "evaluate_diagnostics_character",
    "make_step_character",
    "make_step_character_dynamic",
    "train_student_character",
    "train_student_character_scheduled",
    "train_student_character_scheduled_with_checkpoints",
]
