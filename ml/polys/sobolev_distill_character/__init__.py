"""Character (Re, Im) sibling of :mod:`sobolev_distill`.

Mirrors the real Sobolev distillation pipeline against a complex-valued
character teacher ``f_H : R^2 -> C`` whose lattice value table is
``T_ij = zeta**((i+j) mod p)``.  The arithmetic head produces ``(Re, Im)``
predictions; the energy head uses the real-valued modulus energy
``V_M = (|f_H|**2 - 1)**2 + lam * Z_W`` from
:mod:`graphic_zero_character.character_birkhoff`.

Public surface mirrors :mod:`sobolev_distill` but with ``_character``
suffixes where the math differs.  ``Trunk``, ``EnergyHead``, and the
sampling helpers are reused from :mod:`sobolev_distill` directly.
"""

from __future__ import annotations

from .dataset import (
    CharacterNormalisation,
    CharacterSobolevDataset,
    build_character_dataset,
    sample_minibatch,
    select_character,
)
from .losses import (
    LossWeights,
    energy_loss_character,
    sobolev_loss_character,
    total_loss_character,
    value_and_grad_total_character_dynamic,
)
from .model import (
    AxisProbeHead,
    CharacterArithmeticHead,
    CharacterStudent,
    CharacterStudentConfig,
    CoordMeta,
    FactoredTrunk,
    FourierTrunk,
    f_arith_character,
    f_arith_character_batched,
    f_axis_probe_character,
    f_energy_character,
    f_energy_character_batched,
    make_character_student,
)
from .mechinterp import (
    AblationReport,
    DFTReport,
    HelixReport,
    Surface2DReport,
    ablate_subspace_and_score,
    dft_trunk_along_axis,
    excluded_loss_at_freqs,
    fft2_neuron_surface,
    helix_pca,
)
from .probes import (
    CharacterHessianAlignmentReport,
    CharacterLatentArrays,
    CharacterPatchingReport,
    CharacterProbeBundle,
    LinearProbeReport,
    ModularRecoveryReport,
    PCAReport,
    compute_latents_character,
    hessian_alignment_character,
    latent_pca_character,
    linear_probes_character,
    modular_addition_recovery_probe,
    patching_probe_character,
    run_all_probes_character,
)
from .teacher import (
    CharacterMeshTeacher,
    build_character_teacher_mesh,
    build_character_teacher_mesh_periodic,
)
from .train import (
    CharacterDiagnosticsReport,
    CharacterTrainConfig,
    LinearRampSchedule,
    evaluate_diagnostics_character,
    make_step_character,
    make_step_character_dynamic,
    train_student_character,
    train_student_character_scheduled,
    train_student_character_scheduled_with_checkpoints,
)

__all__ = [
    "AblationReport",
    "AxisProbeHead",
    "CharacterArithmeticHead",
    "CharacterDiagnosticsReport",
    "CharacterHessianAlignmentReport",
    "CharacterLatentArrays",
    "CharacterMeshTeacher",
    "CharacterNormalisation",
    "CharacterPatchingReport",
    "CharacterProbeBundle",
    "CharacterSobolevDataset",
    "CharacterStudent",
    "CharacterStudentConfig",
    "CharacterTrainConfig",
    "CoordMeta",
    "DFTReport",
    "FactoredTrunk",
    "FourierTrunk",
    "HelixReport",
    "LinearProbeReport",
    "LinearRampSchedule",
    "LossWeights",
    "ModularRecoveryReport",
    "PCAReport",
    "Surface2DReport",
    "ablate_subspace_and_score",
    "build_character_dataset",
    "build_character_teacher_mesh",
    "build_character_teacher_mesh_periodic",
    "sample_minibatch",
    "compute_latents_character",
    "dft_trunk_along_axis",
    "energy_loss_character",
    "evaluate_diagnostics_character",
    "excluded_loss_at_freqs",
    "f_arith_character",
    "f_arith_character_batched",
    "f_axis_probe_character",
    "f_energy_character",
    "f_energy_character_batched",
    "fft2_neuron_surface",
    "helix_pca",
    "hessian_alignment_character",
    "latent_pca_character",
    "linear_probes_character",
    "make_character_student",
    "make_step_character",
    "make_step_character_dynamic",
    "modular_addition_recovery_probe",
    "patching_probe_character",
    "run_all_probes_character",
    "select_character",
    "sobolev_loss_character",
    "total_loss_character",
    "train_student_character",
    "train_student_character_scheduled",
    "train_student_character_scheduled_with_checkpoints",
    "value_and_grad_total_character_dynamic",
]
