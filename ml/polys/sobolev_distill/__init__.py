"""Sobolev / Hermite distillation of the Birkhoff-Hermite arithmetic surface (JAX).

Trains a small JAX student with a shared trunk and two heads:

- ``Head A (arithmetic)`` is Sobolev-distilled from ``f_H`` (value + gradient + Hessian)
- ``Head B (energy / valid-input)`` is distilled from ``f_M`` (energy MSE + PD BCE)

Public surface:

- ``teacher.MeshTeacher`` and ``teacher.build_teacher_mesh`` materialize ``f_H`` and
  ``f_M`` mesh tensors via the existing CuPy einsums in
  :mod:`graphic_zero.hermite_barycentric_gpu`.
- ``dataset.SobolevDataset`` and ``dataset.build_dataset`` produce the frozen JAX
  arrays used as Sobolev supervision (``V, GX, GY, Hxx, Hxy, Hyy``) plus the
  ``f_M`` energy ``V_M`` and the ``is_pd`` lattice label.
- ``model.Student`` (Equinox) plus ``model.f_arith`` / ``model.f_energy`` pure
  function entrypoints suitable for ``jax.grad`` and ``jax.hessian``.
- ``losses.sobolev_loss``, ``losses.energy_loss``, ``losses.total_loss``.
- ``train.train_student`` plus ``train.evaluate_diagnostics``.
"""

from __future__ import annotations

from .dataset import SobolevDataset, build_dataset
from .losses import energy_loss, sobolev_loss, total_loss
from .model import (
    ArithmeticHead,
    EnergyHead,
    Student,
    StudentConfig,
    Trunk,
    f_arith,
    f_arith_batched,
    f_energy,
    f_energy_batched,
    make_student,
)
from .probes import (
    HessianAlignmentReport,
    LatentArrays,
    LinearProbeReport,
    PatchingReport,
    PCAReport,
    ProbeBundle,
    compute_latents,
    hessian_alignment,
    latent_pca,
    linear_probes,
    patching_probe,
    run_all_probes,
)
from .teacher import MeshTeacher, build_teacher_mesh
from .train import (
    DiagnosticsReport,
    TrainConfig,
    evaluate_diagnostics,
    make_step,
    train_student,
)

__all__ = [
    "ArithmeticHead",
    "DiagnosticsReport",
    "EnergyHead",
    "HessianAlignmentReport",
    "LatentArrays",
    "LinearProbeReport",
    "MeshTeacher",
    "PCAReport",
    "PatchingReport",
    "ProbeBundle",
    "SobolevDataset",
    "Student",
    "StudentConfig",
    "TrainConfig",
    "Trunk",
    "build_dataset",
    "build_teacher_mesh",
    "compute_latents",
    "energy_loss",
    "evaluate_diagnostics",
    "f_arith",
    "f_arith_batched",
    "f_energy",
    "f_energy_batched",
    "hessian_alignment",
    "latent_pca",
    "linear_probes",
    "make_step",
    "make_student",
    "patching_probe",
    "run_all_probes",
    "sobolev_loss",
    "total_loss",
    "train_student",
]
