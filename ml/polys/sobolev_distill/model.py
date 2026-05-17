"""Equinox model: shared trunk + arithmetic head + energy head.

Design constraints:

- The arithmetic and energy fields are returned by **separate scalar functions**
  (``f_arith`` / ``f_energy``).  This keeps ``jax.grad`` / ``jax.hessian`` calls
  trivial: the differentiated function takes a single ``(2,)`` input.
- Pure-function style with Equinox modules as plain pytrees.  No mutable state,
  no global RNG; all randomness flows through ``jax.random.PRNGKey``.
- Trunk activation defaults to ``gelu``; the heads remain linear so the
  arithmetic head can express arbitrary scalar fields without saturation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp


_ACTIVATIONS: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "silu": jax.nn.silu,
    # ``siren`` is handled specially in ``Trunk.__call__`` because it needs
    # access to the per-trunk ``omega_0`` factor.  Listed here for validation.
    "siren": lambda x: jnp.sin(x),
}


def _activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    try:
        return _ACTIVATIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown activation {name!r}; choose from {sorted(_ACTIVATIONS)}"
        ) from exc


def _siren_reinit_layer(
    layer: eqx.nn.Linear,
    in_dim: int,
    omega_0: float,
    is_first: bool,
    key: jax.Array,
) -> eqx.nn.Linear:
    """Replace ``layer``'s weight/bias with the SIREN distribution.

    First-layer rule: ``W ~ Uniform(-1/in, 1/in)`` (covers a full sine cycle
    over the typical input range when ``omega_0`` is large).
    Hidden layers: ``W ~ Uniform(-sqrt(6/in)/omega_0, sqrt(6/in)/omega_0)``
    (keeps post-activation variance ~ 1 through the network, per Sitzmann
    et al. 2020 supplementary B).
    Biases are zero.
    """
    k_w, k_b = jax.random.split(key)
    if is_first:
        bound = 1.0 / float(in_dim)
    else:
        bound = (6.0 / float(in_dim)) ** 0.5 / float(omega_0)
    new_w = jax.random.uniform(
        k_w, layer.weight.shape, minval=-bound, maxval=bound, dtype=layer.weight.dtype
    )
    new_b = jnp.zeros_like(layer.bias) if layer.bias is not None else None
    layer = eqx.tree_at(lambda l: l.weight, layer, new_w)
    if new_b is not None:
        layer = eqx.tree_at(lambda l: l.bias, layer, new_b)
    return layer


class Trunk(eqx.Module):
    """MLP mapping ``(2,) -> (embed_dim,)``.

    Equinox stores layers in a pytree so optimisers can walk parameters with
    ``jax.tree.map``.  ``act_name`` and ``omega_0`` are captured as static
    fields (no gradient).

    When ``activation == "siren"``, the trunk uses the Sitzmann et al. (2020)
    initialisation so the post-sin activations stay well-conditioned, and the
    forward pass applies ``sin(omega_0 * (W x + b))`` per hidden layer.  The
    output layer remains linear (no sin) so the readout can express any
    range without saturating.
    """

    layers: list
    act_name: str = eqx.field(static=True)
    omega_0: float = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        depth: int,
        embed_dim: int,
        *,
        activation: str = "gelu",
        omega_0: float = 30.0,
        key: jax.Array,
    ):
        if depth < 1:
            raise ValueError("Trunk depth must be >= 1")
        keys = jax.random.split(key, depth + 1)
        layers: list = []
        prev = in_dim
        for d in range(depth):
            layers.append(eqx.nn.Linear(prev, hidden_dim, key=keys[d]))
            prev = hidden_dim
        layers.append(eqx.nn.Linear(prev, embed_dim, key=keys[-1]))

        if activation == "siren":
            init_keys = jax.random.split(keys[-1], depth)  # only hidden layers
            for d in range(depth):
                layers[d] = _siren_reinit_layer(
                    layers[d],
                    in_dim=in_dim if d == 0 else hidden_dim,
                    omega_0=omega_0,
                    is_first=(d == 0),
                    key=init_keys[d],
                )
            # output layer keeps default Equinox init (no sin applied to it).

        self.layers = layers
        self.act_name = activation
        self.omega_0 = float(omega_0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        if self.act_name == "siren":
            for layer in self.layers[:-1]:
                h = jnp.sin(self.omega_0 * layer(h))
        else:
            act = _activation(self.act_name)
            for layer in self.layers[:-1]:
                h = act(layer(h))
        return self.layers[-1](h)


class ArithmeticHead(eqx.Module):
    """Embed -> scalar regression head."""

    layers: list
    act_name: str = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        *,
        activation: str = "gelu",
        key: jax.Array,
    ):
        k1, k2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(embed_dim, hidden_dim, key=k1),
            eqx.nn.Linear(hidden_dim, 1, key=k2),
        ]
        self.act_name = activation

    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        act = _activation(self.act_name)
        return self.layers[1](act(self.layers[0](h)))[0]  # scalar


class EnergyHead(eqx.Module):
    """Embed -> (energy_value, pd_logit).

    The ``energy_value`` is a regression target on normalised ``f_M``; the
    ``pd_logit`` is a binary classifier supervised by ``is_pd``.
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
    ):
        k1, k2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(embed_dim, hidden_dim, key=k1),
            eqx.nn.Linear(hidden_dim, 2, key=k2),
        ]
        self.act_name = activation

    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        act = _activation(self.act_name)
        return self.layers[1](act(self.layers[0](h)))  # (2,)


class Student(eqx.Module):
    """Shared trunk + ``ArithmeticHead`` + ``EnergyHead``."""

    trunk: Trunk
    head_a: ArithmeticHead
    head_b: EnergyHead


@dataclass(frozen=True)
class StudentConfig:
    """Architecture knobs for :func:`make_student`.

    ``omega_0`` is only consulted when ``activation == "siren"``.  Sitzmann et
    al. (2020) recommend ``30.0`` as a robust default for low-dimensional
    coordinate inputs; smaller values bias the network toward smoother
    fields, larger values toward higher-frequency content.
    """

    in_dim: int = 2
    trunk_hidden: int = 64
    trunk_depth: int = 3
    embed_dim: int = 32
    head_hidden: int = 32
    activation: str = "gelu"
    omega_0: float = 30.0
    head_activation: str | None = None


def make_student(key: jax.Array, cfg: StudentConfig | None = None) -> Student:
    cfg = cfg or StudentConfig()
    k_trunk, k_head_a, k_head_b = jax.random.split(key, 3)
    head_act = cfg.head_activation or (
        "gelu" if cfg.activation == "siren" else cfg.activation
    )
    trunk = Trunk(
        in_dim=cfg.in_dim,
        hidden_dim=cfg.trunk_hidden,
        depth=cfg.trunk_depth,
        embed_dim=cfg.embed_dim,
        activation=cfg.activation,
        omega_0=cfg.omega_0,
        key=k_trunk,
    )
    head_a = ArithmeticHead(
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
    return Student(trunk=trunk, head_a=head_a, head_b=head_b)


def f_arith(student: Student, xy: jnp.ndarray) -> jnp.ndarray:
    """Scalar arithmetic prediction at a single ``(2,)`` point.

    Pure function with no batch dim so ``jax.grad(f_arith, argnums=1)`` and
    ``jax.hessian(f_arith, argnums=1)`` produce gradient ``(2,)`` and Hessian
    ``(2, 2)`` directly.
    """
    h = student.trunk(xy)
    return student.head_a(h)


def f_energy(student: Student, xy: jnp.ndarray) -> jnp.ndarray:
    """Energy head output at a single ``(2,)`` point: ``(energy, pd_logit)``."""
    h = student.trunk(xy)
    return student.head_b(h)


def _vmap_along_batch(
    fn: Callable[[Student, jnp.ndarray], jnp.ndarray],
) -> Callable[[Student, jnp.ndarray], jnp.ndarray]:
    return jax.vmap(fn, in_axes=(None, 0))


f_arith_batched = _vmap_along_batch(f_arith)
f_energy_batched = _vmap_along_batch(f_energy)


__all__ = [
    "ArithmeticHead",
    "EnergyHead",
    "Student",
    "StudentConfig",
    "Trunk",
    "f_arith",
    "f_arith_batched",
    "f_energy",
    "f_energy_batched",
    "make_student",
]
