# 03 - Sobolev distillation

The functional-analytic content of the loss in [losses.py](../../../sobolev_distill_character/losses.py): why matching value, gradient, and Hessian is approximation in $H^2(T^2)$, why $H^2$ is uniform on the 2-torus, and why the unit-circle penalty is a topological constraint rather than a regulariser.

## Sections

1. The Sobolev space $H^k(T^2)$
2. Spectral characterisation via Pontryagin duality
3. Sobolev embedding in two dimensions
4. The distillation loss as $H^2$ approximation
5. The autodiff Hessian as the engine of the matching
6. The unit-circle penalty as Ginzburg-Landau energy
7. Topological consequence: preservation of the homotopy class
8. The auxiliary axis loss as low-frequency Sobolev preconditioning
9. Code: where this chapter is realised
10. Further reading

---

## 1. The Sobolev space $H^k(T^2)$

**Definition.** Let $k \in \mathbb{Z}_{\ge 0}$. The **Sobolev space of order $k$** on $T^2$ is

$$H^k(T^2) \;:=\; \bigl\{ f \in L^2(T^2) : \partial^\alpha f \in L^2(T^2) \text{ for all multi-indices } |\alpha| \le k \bigr\}$$

equipped with the norm

$$\|f\|_{H^k}^2 \;:=\; \sum_{|\alpha| \le k} \|\partial^\alpha f\|_{L^2}^2.$$

For $k = 2$ on $T^2$ the multi-indices $|\alpha| \le 2$ contribute the seven terms $f, \partial_x f, \partial_y f, \partial_x^2 f, \partial_x \partial_y f, \partial_y \partial_x f, \partial_y^2 f$ (with the two mixed derivatives equal by Schwarz; we count one of them with multiplicity in the Hessian section below). See Adams-Fournier (2003, Chapter 3) for the general definition.

## 2. Spectral characterisation

On $T^2$, Sobolev norms admit a clean Pontryagin-dual characterisation. Let $\hat f(n)$ for $n = (n_1, n_2) \in \mathbb{Z}^2$ be the Fourier coefficient

$$\hat f(n) \;:=\; \frac{1}{p^2} \int_{T^2} f(x, y) \, e^{-2\pi i (n_1 x + n_2 y) / p} \, dx \, dy.$$

**Proposition (spectral norm).** For $f \in H^k(T^2)$,

$$\|f\|_{H^k}^2 \;=\; \sum_{n \in \mathbb{Z}^2} (1 + |n|^2)^k \, |\hat f(n)|^2$$

up to a $k$-dependent constant. (Different conventions absorb the constant differently; we follow Adams-Fournier.)

**Proof.** Apply Parseval to each $\partial^\alpha f$ in the definition; differentiation in physical space corresponds to multiplication by $(2\pi i n_1)^{\alpha_1} (2\pi i n_2)^{\alpha_2} / p^{|\alpha|}$ in Fourier space. Sum the resulting weights over $|\alpha| \le k$ and rearrange. $\square$

**Consequence.** $H^k(T^2)$ is the closure of $C^\infty(T^2)$ under the spectral norm; functions whose Fourier coefficients decay faster than $|n|^{-k}$ are "smooth of order $k$".

## 3. Sobolev embedding in $d = 2$

**Theorem (Sobolev embedding).** Let $d := \dim T^2 = 2$. For $k > d / 2 = 1$,

$$H^k(T^2) \;\hookrightarrow\; C^0(T^2)$$

continuously: there exists a constant $C_k > 0$ such that

$$\|f\|_{C^0} \;\le\; C_k \, \|f\|_{H^k} \qquad \forall f \in H^k(T^2).$$

In particular, control of the $H^2$ norm gives uniform control of $f$ -- one cannot have small $H^2$ error and a large pointwise error simultaneously. See Adams-Fournier (2003, Theorem 4.12) for the full statement and proof.

**Why this matters.** Distillation in $L^2$ controls $f$ only on average; $L^2$-small errors can have arbitrarily large pointwise spikes. Distillation in $H^2$ controls $f$ uniformly; matching value plus gradient plus Hessian on a dense mesh therefore guarantees the student is close to the teacher at every point of $T^2$, not just on average.

For the modular-arithmetic problem this is the right notion of "close": the modular-accuracy decoder $\mathrm{round}(p \cdot \arg(f) / 2\pi) \mod p$ is sensitive to pointwise errors near argument boundaries, so $H^2$ is the natural target.

## 4. The distillation loss as $H^2$ approximation

The training loss in [losses.py:LossWeights, _per_point_per_channel_value_grad_hess](../../../sobolev_distill_character/losses.py) has the form

$$\mathcal{L}(\theta) \;=\; \mathbb{E}_{(x, y) \sim \mu}\bigl[ \alpha_0 \, |f_\theta - T|^2 + \alpha_1 \, \|\nabla f_\theta - \nabla T\|^2 + \alpha_2 \, \|\nabla^2 f_\theta - \nabla^2 T\|_F^2 \bigr] \;+\; \text{auxiliary terms.}$$

The expectation is taken over a sampling measure $\mu$ on $T^2$ supported on (a) the lattice $\Lambda$, (b) Chebyshev-weighted off-lattice mesh points, and (c) uniform off-lattice points. The norms are pointwise: $|\cdot|$ for the complex value, $\|\cdot\|_2$ for the $\mathbb{R}^2$-valued gradient, $\|\cdot\|_F$ for the $2 \times 2$ Hessian.

**Proposition.** The expectation is a Riemann-sum approximation of

$$\|f_\theta - T\|_{L^2(\mu)}^2 + \alpha_1 \|\nabla f_\theta - \nabla T\|_{L^2(\mu)}^2 + \alpha_2 \|\nabla^2 f_\theta - \nabla^2 T\|_{L^2(\mu)}^2,$$

which is the $H^2$ norm of $f_\theta - T$ with respect to the measure $\mu$. As $|\mu| \to \infty$ this converges to the true $H^2(T^2)$ approximation error.

**Proof sketch.** Linearity of expectation; pointwise norms aggregate via the same orthogonal decomposition as the $H^k$ semi-norms; the Riemann-sum interpretation follows from the standard convergence of empirical means to integrals against the measure $\mu$. The asymmetric weighting via $\alpha_1, \alpha_2$ rescales the $H^k$ norm by per-derivative weights, but the topology is unchanged. $\square$

So the project's loss is genuinely an $H^2$-approximation loss, not merely an $L^2$ loss with derivative regularisers.

## 5. The autodiff Hessian as the engine

The right-hand side targets $\nabla T, \nabla^2 T$ are computed in closed form (chapter 02). The student's $\nabla f_\theta, \nabla^2 f_\theta$ are computed by JAX autodiff:

```python
def channel(c):
    return lambda xy: f_arith_character(student, xy)[c]

g0 = jax.grad(channel(0))(xy)               # 2-vector
g1 = jax.grad(channel(1))(xy)               # 2-vector
h0 = jax.hessian(channel(0))(xy)            # 2x2 matrix
h1 = jax.hessian(channel(1))(xy)            # 2x2 matrix
```

(Reproduced from [losses.py:_per_point_per_channel_value_grad_hess](../../../sobolev_distill_character/losses.py).)

`jax.hessian` is a composition of forward- and reverse-mode autodiff; for the SIREN trunk's modest depth (3 layers, 64 hidden units in the canonical config) the cost per evaluation is $O(d^2)$ where $d = 2$. `jax.vmap` then batches across mesh points, and `eqx.filter_jit` compiles the whole closure once per shape.

This is the *implementation* leverage that makes $H^2$ distillation tractable. Without analytic Hessian targets and analytic-by-autodiff student Hessians, an $H^2$ loss would require finite-difference approximations whose noise would dominate the signal.

## 6. The unit-circle penalty

The teacher's image is $S^1 \subset \mathbb{R}^2$ (chapter 01). To enforce the same constraint on the student, [losses.py](../../../sobolev_distill_character/losses.py) includes the term

$$\mathcal{E}_{\mathrm{uc}}(f) \;:=\; \mathbb{E}_{(x, y)} \bigl[ (|f(x, y)|^2 - 1)^2 \bigr].$$

This is a **double-well potential** with minimum manifold

$$\bigl\{ z \in \mathbb{R}^2 : |z| = 1 \bigr\} \;=\; S^1.$$

The same expression appears in **Ginzburg-Landau theory** as the bulk potential for an order-parameter field $f : \Omega \to \mathbb{R}^2$ taking values near the unit circle (Ginzburg-Landau 1950, Bethuel-Brezis-Helein 1994). The mathematical content is identical: the field is encouraged to live on $S^1$, and excursions off $S^1$ are penalised quartically.

**Why quartic and not just $(|f| - 1)^2$.** The quartic form $(|f|^2 - 1)^2$ is smooth across $f = 0$, where $|f|$ has a corner. The quadratic form would create gradient-flow obstacles at the origin. This is a standard analytic convenience documented in Bethuel-Brezis-Helein (1994).

## 7. Topological consequence: homotopy preservation

Continuous deformations of a continuous map $T^2 \to S^1$ preserve its homotopy class in

$$[T^2, S^1] \;\cong\; H^1(T^2; \mathbb{Z}) \;\cong\; \mathbb{Z}^2$$

(chapter 01, section 6). The teacher sits at the generator $(1, 1)$.

**Proposition (homotopy preservation under unit-circle gradient flow).** Let $f_t : T^2 \to S^1$ be a one-parameter family with $\|f_t\|_{C^0(T^2)}$ bounded uniformly and $|f_t| = 1$ pointwise. Then the homotopy class $[f_t] \in [T^2, S^1]$ is constant in $t$.

**Proof.** A continuous family of continuous maps is a homotopy, which by definition preserves homotopy class. $\square$

**Consequence.** Once the student $f_\theta$ is in homotopy class $(1, 1)$, gradient flow under the unit-circle penalty cannot leave this class without paying an unbounded $\mathcal{E}_{\mathrm{uc}}$ cost (because passing between classes requires a continuous deformation through some configuration with $|f| \ne 1$). In practice the unit-circle penalty acts as a **soft topological barrier**: the student stays in the correct homotopy class throughout training.

This is the structural reason the helix-PCA wrap angles consistently come out as $w_i / 2\pi = \pm 1$ in the experimental notebooks. The trunk's representation has the right homotopy class because the loss prevents it from leaving.

## 8. The auxiliary axis loss as Sobolev preconditioning

In addition to the value / gradient / Hessian terms, [losses.py](../../../sobolev_distill_character/losses.py) defines an **axis-probe loss**:

$$\mathcal{L}_{\mathrm{axis}}(\theta) \;:=\; \mathbb{E}_{(x, y)} \bigl\| A_\theta(\mathrm{trunk}_\theta(x, y)) - (\cos \omega x, \sin \omega x, \cos \omega y, \sin \omega y) \bigr\|^2$$

where $\omega = 2\pi / p$ and $A_\theta$ is a learnable linear map (the *axis probe*) from the trunk's $D$-dimensional output to $\mathbb{R}^4$. The targets are the lowest non-trivial real-form Fourier features.

**Mathematical content.** The axis loss explicitly forces the trunk to make $\{\cos \omega x, \sin \omega x, \cos \omega y, \sin \omega y\}$ linearly decodable. These four functions are the real basis of the $|n| = 1$ subspace of $\mathcal{T}_p \otimes \mathcal{T}_p$.

**Effect on Sobolev approximation.** The axis loss is a low-frequency *curriculum* for the trunk: it enforces that the lowest-mode Fourier features appear in the linear span of the trunk's activations from the start of training. This biases the trunk toward representations whose Fourier content is concentrated on the modes that the teacher actually uses ($k = 1$ in our case; the character is one Fourier mode).

**Side effect on the per-neuron trig score.** Because the axis loss puts neurons on axial top modes $(0, 1)$ or $(1, 0)$ in the 2-D Fourier decomposition, the *per-neuron* sum-of-angles ratio (chapter 04, section 8) collapses on these neurons. This is a genuine measurement caveat documented in the experimental notebooks: the modulus_sweep configurations use `axis_probe = True` and the per-neuron `soa_median` reads `0.000` even though the population-level Fourier story is clean. Chapter 04 returns to this point.

## 9. Code anchors

| structure | code |
|---|---|
| `LossWeights` (the seven coefficients) | [losses.py:LossWeights](../../../sobolev_distill_character/losses.py) |
| Per-point value / grad / Hessian | [losses.py:_per_point_per_channel_value_grad_hess](../../../sobolev_distill_character/losses.py) |
| Unit-circle penalty | the `unit_circle` term in `LossWeights`; see the `value_grad_hess` block |
| Axis-probe target construction | `axis_target_cos_x, axis_target_sin_x, ...` in [dataset.py](../../../sobolev_distill_character/dataset.py) |
| Axis-probe head | `AxisProbeHead` in [model.py](../../../sobolev_distill_character/model.py) |
| Optimiser (Adam / AdamW with cosine decay) | [train.py:_build_optimizer](../../../sobolev_distill_character/train.py) |
| `LinearRampSchedule` for the gradient-loss weight | [train.py:LinearRampSchedule](../../../sobolev_distill_character/train.py) |

## 10. Further reading

- **R. A. Adams and J. J. F. Fournier**, *Sobolev Spaces*, 2nd ed., Academic Press, 2003. Chapter 3 (definition), Theorem 4.12 (embedding), Chapter 7 (compactness).
- **L. C. Evans**, *Partial Differential Equations*, 2nd ed., AMS, 2010. Chapter 5 for an alternative treatment of Sobolev embedding and a clean proof in $d \le 2$.
- **F. Bethuel, H. Brezis, and F. Helein**, *Ginzburg-Landau Vortices*, Birkhauser, 1994. The canonical analytic treatment of the $(|f|^2 - 1)^2$ potential in two dimensions.
- **L. N. Trefethen**, *Spectral Methods in MATLAB*, SIAM, 2000. Chapter 2 for Riemann-sum convergence on the torus; Chapter 4 for the role of derivatives in spectral approximation.

---

Cross-reference: chapter 02 supplies the analytic targets $T, \nabla T, \nabla^2 T$ used here. Chapter 04 explains why the band-limit-1 Fourier content of the teacher is the *only* signal a correctly-trained student can carry. Chapter 05 discusses the topological consequences in terms of the trunk's PC1-PC2 wrap angle.
