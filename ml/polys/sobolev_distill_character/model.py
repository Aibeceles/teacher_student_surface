"""Equinox character student: shared trunk + (Re, Im) arithmetic head + energy head.

The arithmetic head returns a length-2 vector ``(re, im)`` so ``jax.grad`` /
``jax.hessian`` of each component yields the natural ``(2,)`` / ``(2, 2)``
shapes per channel.  The default trunk is reused unchanged from
:mod:`sobolev_distill.model.Trunk`; alternative ``FourierTrunk`` and
``FactoredTrunk`` modules implement explicit factorisation pressure.

Three trunk kinds are exposed via ``CharacterStudentConfig.trunk_kind``:

- ``"mlp"`` (default): the original MLP/SIREN trunk.
- ``"fourier"``: deterministic Fourier features at integer harmonics of
  ``2 pi / p`` projected to ``embed_dim``.  Axis probes are linearly trivial
  by construction.
- ``"factored"``: two 1D MLPs producing ``(axis_emb,)`` each, concatenated
  to ``(2 * axis_emb,)`` and projected to ``embed_dim``.  The downstream
  ``CharacterArithmeticHead`` (a small MLP, not a hardcoded complex
  multiply) supplies the off-lattice flexibility the cell-23 sanity
  baseline lacked.

An optional :class:`AxisProbeHead` adds the auxiliary 4-target axis loss
to any trunk (defaults to off).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from sobolev_distill.model import EnergyHead, Trunk, _activation


@dataclass(frozen=True)
class CoordMeta:
    """Affine constants needed to map normalised inputs back to raw coords.

    ``FourierTrunk`` consumes normalised ``xy in [-1, 1]`` from the dataset
    but evaluates a periodic basis in raw coordinates, so the trunk needs
    ``x_center, x_scale`` (and the same for y) plus the modulus ``p``.
    These mirror :class:`~sobolev_distill_character.dataset.CharacterNormalisation`
    fields one-to-one.
    """

    p: int
    x_center: float
    x_scale: float
    y_center: float
    y_scale: float

    @classmethod
    def from_dataset(cls, dataset) -> "CoordMeta":  # type: ignore[no-untyped-def]
        norm = dataset.norm
        return cls(
            p=int(dataset.modulus),
            x_center=float(norm.x_center),
            x_scale=float(norm.x_scale),
            y_center=float(norm.y_center),
            y_scale=float(norm.y_scale),
        )


class CharacterArithmeticHead(eqx.Module):
    """Embed -> ``(re, im)`` regression head.

    Same MLP shape as :class:`sobolev_distill.model.ArithmeticHead`, but
    with output dimension 2 instead of 1.
    """

    layers: list
    act_name: str = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        *,
        activation: str = "gelu",
        key: jax.Array,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(embed_dim, hidden_dim, key=k1),
            eqx.nn.Linear(hidden_dim, 2, key=k2),
        ]
        self.act_name = activation

    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        act = _activation(self.act_name)
        return self.layers[1](act(self.layers[0](h)))  # (2,)


class AxisProbeHead(eqx.Module):
    """Linear projection ``embed_dim -> 4`` onto axis targets.

    Output ordering matches ``select_character``'s ``axis_target``:
    ``(cos 2 pi x_raw/p, sin 2 pi x_raw/p, cos 2 pi y_raw/p,
    sin 2 pi y_raw/p)``.  The probe is a single linear layer so the loss
    explicitly measures *linear* decodability of axis features in the
    trunk output.
    """

    layer: eqx.nn.Linear

    def __init__(self, embed_dim: int, *, key: jax.Array) -> None:
        self.layer = eqx.nn.Linear(embed_dim, 4, key=key)

    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        return self.layer(h)


class FourierTrunk(eqx.Module):
    """Deterministic Fourier-feature trunk + linear projection.

    The basis is
    ``phi(x, y) = concat_k=1..K [cos(omega_k * x_raw), sin(omega_k * x_raw),
    cos(omega_k * y_raw), sin(omega_k * y_raw)]``
    with ``omega_k = 2 pi k / p``, giving a ``(4K,)`` feature.  A single
    ``eqx.nn.Linear(4K, embed_dim)`` projects to the canonical embed width
    so downstream heads, probes, and patching code see a familiar shape.

    Coordinate conversion: the trunk receives normalised
    ``xy in [-1, 1]`` and applies the affine
    ``x_raw = xy[0] * x_scale + x_center`` (same for y).  Coord meta is
    stored as static fields so the basis is JIT-friendly.
    """

    proj: eqx.nn.Linear
    K: int = eqx.field(static=True)
    p: int = eqx.field(static=True)
    x_center: float = eqx.field(static=True)
    x_scale: float = eqx.field(static=True)
    y_center: float = eqx.field(static=True)
    y_scale: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        embed_dim: int,
        K: int,
        coord_meta: CoordMeta,
        key: jax.Array,
    ) -> None:
        if K < 1:
            raise ValueError(f"FourierTrunk requires K >= 1, got {K}")
        self.K = int(K)
        self.p = int(coord_meta.p)
        self.x_center = float(coord_meta.x_center)
        self.x_scale = float(coord_meta.x_scale)
        self.y_center = float(coord_meta.y_center)
        self.y_scale = float(coord_meta.y_scale)
        self.proj = eqx.nn.Linear(4 * self.K, embed_dim, key=key)

    def features(self, xy: jnp.ndarray) -> jnp.ndarray:
        """Return the raw ``(4K,)`` Fourier features at one point."""
        x_raw = xy[0] * self.x_scale + self.x_center
        y_raw = xy[1] * self.y_scale + self.y_center
        omega = 2.0 * math.pi / float(self.p)
        ks = jnp.arange(1, self.K + 1, dtype=xy.dtype)
        ang_x = omega * ks * x_raw
        ang_y = omega * ks * y_raw
        return jnp.concatenate(
            [jnp.cos(ang_x), jnp.sin(ang_x), jnp.cos(ang_y), jnp.sin(ang_y)],
            axis=-1,
        )

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        return self.proj(self.features(xy))


class _AxisMLP(eqx.Module):
    """Tiny 1D -> ``axis_emb`` MLP for one coordinate axis."""

    layers: list
    act_name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        hidden: int,
        depth: int,
        out_dim: int,
        key: jax.Array,
        activation: str = "gelu",
    ) -> None:
        if depth < 1:
            raise ValueError("AxisMLP depth must be >= 1")
        ks = jax.random.split(key, depth + 1)
        layers: list = [eqx.nn.Linear(1, hidden, key=ks[0])]
        for k in ks[1:-1]:
            layers.append(eqx.nn.Linear(hidden, hidden, key=k))
        layers.append(eqx.nn.Linear(hidden, out_dim, key=ks[-1]))
        self.layers = layers
        self.act_name = activation

    def __call__(self, x_scalar: jnp.ndarray) -> jnp.ndarray:
        h = jnp.atleast_1d(x_scalar)
        act = _activation(self.act_name)
        for layer in self.layers[:-1]:
            h = act(layer(h))
        return self.layers[-1](h)


class FactoredTrunk(eqx.Module):
    """Two 1D MLPs concatenated, then projected to ``embed_dim``.

    Unlike the cell-23 sanity baseline, the head downstream is the
    standard :class:`CharacterArithmeticHead` MLP rather than a hardcoded
    complex multiply.  This keeps off-lattice Hermite fittable while the
    factored trunk exposes per-axis features that linearly decode to
    ``cos/sin 2 pi x/p``.
    """

    mlp_x: _AxisMLP
    mlp_y: _AxisMLP
    proj: eqx.nn.Linear
    axis_emb: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        embed_dim: int,
        axis_emb: int,
        axis_hidden: int,
        axis_depth: int,
        activation: str,
        key: jax.Array,
    ) -> None:
        k_x, k_y, k_p = jax.random.split(key, 3)
        self.mlp_x = _AxisMLP(
            hidden=axis_hidden,
            depth=axis_depth,
            out_dim=axis_emb,
            key=k_x,
            activation=activation,
        )
        self.mlp_y = _AxisMLP(
            hidden=axis_hidden,
            depth=axis_depth,
            out_dim=axis_emb,
            key=k_y,
            activation=activation,
        )
        self.proj = eqx.nn.Linear(2 * axis_emb, embed_dim, key=k_p)
        self.axis_emb = int(axis_emb)

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        ex = self.mlp_x(xy[0])
        ey = self.mlp_y(xy[1])
        return self.proj(jnp.concatenate([ex, ey], axis=-1))


class CharacterStudent(eqx.Module):
    """Shared trunk + ``CharacterArithmeticHead`` + ``EnergyHead``.

    ``trunk`` is one of :class:`Trunk` (default), :class:`FourierTrunk`,
    or :class:`FactoredTrunk`.  ``axis_probe`` is ``None`` unless
    ``CharacterStudentConfig.axis_probe == True``; when present, the
    auxiliary 4-target axis loss is enabled (gated on
    ``LossWeights.axis``).
    """

    trunk: eqx.Module
    head_a: CharacterArithmeticHead
    head_b: EnergyHead
    axis_probe: AxisProbeHead | None = None


@dataclass(frozen=True)
class CharacterStudentConfig:
    """Architecture knobs for :func:`make_character_student`.

    ``trunk_kind`` selects between the standard MLP/SIREN trunk
    (``"mlp"``), a fixed Fourier-feature trunk (``"fourier"``), or a
    factored two-axis trunk (``"factored"``).  ``coord_meta`` is required
    for ``"fourier"`` and is otherwise ignored.
    """

    in_dim: int = 2
    trunk_hidden: int = 64
    trunk_depth: int = 3
    embed_dim: int = 32
    head_hidden: int = 32
    activation: str = "gelu"
    omega_0: float = 30.0
    head_activation: str | None = None
    trunk_kind: Literal["mlp", "fourier", "factored"] = "mlp"
    fourier_K: int = 4
    axis_emb: int = 16
    axis_probe: bool = False


def make_character_student(
    key: jax.Array,
    cfg: CharacterStudentConfig | None = None,
    *,
    coord_meta: CoordMeta | None = None,
) -> CharacterStudent:
    """Build a :class:`CharacterStudent` given a config.

    ``coord_meta`` is required when ``cfg.trunk_kind == "fourier"`` because
    the Fourier basis is evaluated in raw coordinates.  For other trunk
    kinds it is ignored.
    """
    cfg = cfg or CharacterStudentConfig()
    k_trunk, k_head_a, k_head_b, k_probe = jax.random.split(key, 4)
    head_act = cfg.head_activation or (
        "gelu" if cfg.activation == "siren" else cfg.activation
    )

    trunk: eqx.Module
    if cfg.trunk_kind == "mlp":
        trunk = Trunk(
            in_dim=cfg.in_dim,
            hidden_dim=cfg.trunk_hidden,
            depth=cfg.trunk_depth,
            embed_dim=cfg.embed_dim,
            activation=cfg.activation,
            omega_0=cfg.omega_0,
            key=k_trunk,
        )
    elif cfg.trunk_kind == "fourier":
        if coord_meta is None:
            raise ValueError(
                "trunk_kind='fourier' requires coord_meta; pass "
                "CoordMeta.from_dataset(dataset) when calling "
                "make_character_student."
            )
        trunk = FourierTrunk(
            embed_dim=cfg.embed_dim,
            K=cfg.fourier_K,
            coord_meta=coord_meta,
            key=k_trunk,
        )
    elif cfg.trunk_kind == "factored":
        # Factored activation defaults: gelu axis MLPs unless the user
        # explicitly chose siren.  Reusing trunk_hidden/trunk_depth keeps
        # the per-axis capacity comparable to the MLP baseline.
        trunk = FactoredTrunk(
            embed_dim=cfg.embed_dim,
            axis_emb=cfg.axis_emb,
            axis_hidden=cfg.trunk_hidden,
            axis_depth=cfg.trunk_depth,
            activation=("gelu" if cfg.activation == "siren" else cfg.activation),
            key=k_trunk,
        )
    else:
        raise ValueError(
            f"Unknown trunk_kind={cfg.trunk_kind!r}; "
            "choose from {'mlp', 'fourier', 'factored'}"
        )

    head_a = CharacterArithmeticHead(
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.head_hidden,
        activation=head_act,
        key=k_head_a,
    )
    head_b = EnergyHead(
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.head_hidden,
        activation=head_act,
        key=k_head_b,
    )
    axis_probe = (
        AxisProbeHead(embed_dim=cfg.embed_dim, key=k_probe) if cfg.axis_probe else None
    )
    return CharacterStudent(
        trunk=trunk, head_a=head_a, head_b=head_b, axis_probe=axis_probe
    )


def f_arith_character(student: CharacterStudent, xy: jnp.ndarray) -> jnp.ndarray:
    """Length-2 ``(re, im)`` prediction at a single ``(2,)`` point."""
    h = student.trunk(xy)
    return student.head_a(h)


def f_energy_character(student: CharacterStudent, xy: jnp.ndarray) -> jnp.ndarray:
    """Energy head output at a single ``(2,)`` point: ``(energy, pd_logit)``."""
    h = student.trunk(xy)
    return student.head_b(h)


def f_axis_probe_character(
    student: CharacterStudent, xy: jnp.ndarray
) -> jnp.ndarray:
    """Length-4 axis-probe prediction at a single point.

    Raises if ``student.axis_probe is None``; callers are expected to gate
    on ``weights.axis > 0`` before invoking this path.
    """
    if student.axis_probe is None:
        raise ValueError(
            "f_axis_probe_character requires a student built with "
            "CharacterStudentConfig(axis_probe=True)."
        )
    h = student.trunk(xy)
    return student.axis_probe(h)


def _vmap_along_batch(
    fn: Callable[[CharacterStudent, jnp.ndarray], jnp.ndarray],
) -> Callable[[CharacterStudent, jnp.ndarray], jnp.ndarray]:
    return jax.vmap(fn, in_axes=(None, 0))


f_arith_character_batched = _vmap_along_batch(f_arith_character)
f_energy_character_batched = _vmap_along_batch(f_energy_character)


__all__ = [
    "AxisProbeHead",
    "CharacterArithmeticHead",
    "CharacterStudent",
    "CharacterStudentConfig",
    "CoordMeta",
    "FactoredTrunk",
    "FourierTrunk",
    "f_arith_character",
    "f_arith_character_batched",
    "f_axis_probe_character",
    "f_energy_character",
    "f_energy_character_batched",
    "make_character_student",
]
