# 02 - Interpolation, cardinal bases, and RKHS

The approximation-theory backbone of the periodic-cardinal teacher: the trig-polynomial space $\mathcal{T}_p$ as an RKHS, the Dirichlet kernel as its reproducing kernel, the cardinal interpolation property, and what is structurally lost when the periodic interpolant is replaced by a polynomial Hermite-Birkhoff one.

## Sections

1. The trig-polynomial space $\mathcal{T}_p \subset L^2(T^1)$
2. The reproducing-kernel Hilbert space (RKHS) framework
3. The Dirichlet kernel as the reproducing kernel of $\mathcal{T}_p$
4. Cardinal property at equispaced nodes
5. Exact reconstruction of the character from $p$ samples
6. Closed-form first and second derivatives of the cardinal basis
7. Tensor product to $T^2$
8. The polynomial Hermite-Birkhoff alternative on $[0, p-1]$
9. Boundary-value mismatch with $T^2$ and Runge oscillation
10. Code: where this chapter is realised
11. Further reading

---

## 1. The trig-polynomial space $\mathcal{T}_p$

**Definition.** Set $K := \lfloor p / 2 \rfloor$. The **trig-polynomial space at modulus $p$** is

$$\mathcal{T}_p \;:=\; \mathrm{span}_\mathbb{C}\{e^{2\pi i k x / p} : k = -K, -K+1, \ldots, K\} \;\subset\; L^2(T^1).$$

For $p$ odd the basis has $p = 2K + 1$ elements; for $p$ even one usually adopts the convention that the Nyquist mode $k = p/2$ is included as $\cos(\pi x)$, also giving $\dim \mathcal{T}_p = p$.

**Proposition (eigenfunctions of the Laplacian).** Each basis function $e^{2\pi i k x / p}$ is a smooth eigenfunction of the Laplacian $-\partial_x^2$ on $T^1 = \mathbb{R}/p\mathbb{Z}$ with eigenvalue $(2\pi k / p)^2$. Together they span the eigenspace of the first $p$ eigenvalues.

**Proof.** Direct: $-\partial_x^2 e^{2\pi i k x / p} = (2\pi k / p)^2 e^{2\pi i k x / p}$. $\square$

So $\mathcal{T}_p$ is a finite-dimensional Hilbert space with the inherited $L^2$ inner product

$$\langle f, g \rangle_{\mathcal{T}_p} := \frac{1}{p} \int_0^p f(x) \overline{g(x)} \, dx,$$

and an orthonormal basis $\{e^{2\pi i k x / p}\}_{k = -K}^{K}$.

**Why this space.** The fundamental character $\chi(x) = e^{2\pi i x / p}$ is the basis element $k = 1$; it lies in $\mathcal{T}_p$ exactly. So any approximation theory that works on $\mathcal{T}_p$ recovers $\chi$ exactly (chapter 01 emphasised the topological content; this chapter explains why the periodic-cardinal teacher is *analytically* exact).

## 2. The RKHS framework

**Definition.** A **reproducing kernel Hilbert space (RKHS)** on a set $X$ is a Hilbert space $\mathcal{H} \subset \mathbb{C}^X$ such that point evaluation $\delta_x : f \mapsto f(x)$ is a bounded linear functional for every $x \in X$. By Riesz representation, there exists a unique $K_x \in \mathcal{H}$ with

$$f(x) = \langle f, K_x \rangle_\mathcal{H} \qquad \forall f \in \mathcal{H}.$$

The function $K(x, y) := \langle K_y, K_x \rangle_\mathcal{H} = K_y(x)$ is the **reproducing kernel** of $\mathcal{H}$.

Every finite-dimensional Hilbert space of functions on a set is automatically an RKHS, so $\mathcal{T}_p$ is an RKHS on $T^1$. The reproducing kernel is computed from any orthonormal basis $\{\phi_k\}$ by

$$K(x, y) = \sum_k \overline{\phi_k(y)} \, \phi_k(x).$$

See Berlinet-Thomas-Agnan (2004, Chapter 1) for the general framework.

## 3. The Dirichlet kernel

**Proposition.** The reproducing kernel of $\mathcal{T}_p$ is the **Dirichlet kernel**

$$D_p(x - y) \;=\; \sum_{k = -K}^{K} e^{2\pi i k (x - y) / p}.$$

For $p$ odd this admits the closed form

$$D_p(x) \;=\; \frac{\sin\bigl((K + \tfrac{1}{2}) \cdot 2\pi x / p\bigr)}{\sin(\pi x / p)} \qquad \text{for } x \notin p\mathbb{Z},$$

with $D_p(0) = p$ by L'Hopital. For $p$ even there is an analogous formula with a half-step shift; we treat both cases below by direct sum-of-exponentials.

**Proof.** Apply the formula $K(x, y) = \sum_k \overline{\phi_k(y)} \phi_k(x)$ with $\phi_k(x) = e^{2\pi i k x / p}$. The sum is the geometric series in $z := e^{2\pi i (x - y) / p}$:

$$\sum_{k = -K}^{K} z^k = z^{-K} \cdot \frac{z^{2K + 1} - 1}{z - 1} = \frac{z^{K + 1/2} - z^{-(K + 1/2)}}{z^{1/2} - z^{-1/2}} = \frac{\sin\bigl((K + 1/2) \cdot 2\pi (x - y) / p\bigr)}{\sin(\pi (x - y) / p)}.$$

$\square$

The Dirichlet kernel is real-valued and depends only on $x - y$ (since $\mathcal{T}_p$ is translation-invariant under $T^1$).

## 4. Cardinal property at equispaced nodes

**Proposition (cardinal property).** Let $x_j := j$ for $j = 0, 1, \ldots, p-1$ (the $p$ equispaced lattice nodes on $[0, p)$). Then

$$D_p(x_j - x_k) = p \cdot \delta_{jk}.$$

**Proof.** Substitute $x_j - x_k = j - k$ into $D_p$:

$$D_p(j - k) = \sum_{m = -K}^{K} e^{2\pi i m (j - k) / p}.$$

The integrand depends only on $m \bmod p$, and for $p$ odd the index set $\{-K, \ldots, K\}$ with $K = (p-1)/2$ is a complete residue system mod $p$ (so this sum equals $\sum_{m = 0}^{p-1} e^{2\pi i m (j - k) / p}$; for $p$ even a similar identification holds with the Nyquist mode included as $\cos(\pi(j-k))$). By the standard finite-Fourier identity

$$\sum_{m = 0}^{p - 1} e^{2\pi i m \ell / p} = p \cdot \delta_{\ell \equiv 0 \bmod p},$$

the sum gives $p$ if $j \equiv k \bmod p$ and $0$ otherwise. Since $j, k \in \{0, \ldots, p-1\}$, this is $p \cdot \delta_{jk}$. $\square$

The normalised cardinal basis is

$$\psi_j(x) \;:=\; \frac{1}{p} D_p(x - x_j),$$

so $\psi_j(x_k) = \delta_{jk}$. The map $\psi_j$ takes value $1$ at its node and $0$ at every other lattice node.

## 5. Exact reconstruction of the character

**Theorem (cardinal interpolation is exact on $\mathcal{T}_p$).** For every $f \in \mathcal{T}_p$ and every $x \in T^1$,

$$f(x) \;=\; \sum_{j = 0}^{p - 1} f(x_j) \, \psi_j(x).$$

**Proof.** Both sides are elements of $\mathcal{T}_p$ (the right-hand side is a linear combination of $D_p(\cdot - x_j) \in \mathcal{T}_p$). They agree at the $p$ lattice nodes by the cardinal property. Two elements of a $p$-dimensional space that agree at $p$ points (where the values determine the function -- a unisolvent set) must be equal. $\square$

**Corollary (the teacher is exact off-lattice).** The fundamental character $\chi(x) = e^{2\pi i x / p}$ lies in $\mathcal{T}_p$ as the $k = 1$ basis element. Therefore the cardinal interpolant of $\chi$ from its $p$ lattice values $\{\chi(0), \chi(1), \ldots, \chi(p-1)\} = \{1, \zeta, \zeta^2, \ldots, \zeta^{p-1}\}$ reproduces $\chi$ exactly on all of $T^1$.

This is the key analytical claim of the periodic-cardinal teacher. Off-lattice, the teacher's value, gradient, and Hessian are all *exact* (no Runge oscillation, no approximation error from finite $p$); the Sobolev distillation in chapter 03 matches analytical targets, not numerical approximations.

## 6. Closed-form derivatives of $\psi_j$

**Proposition.** The first derivative of the Dirichlet kernel at $x \notin p\mathbb{Z}$ is

$$D_p'(x) \;=\; \frac{2\pi}{p} \cdot \biggl[ \bigl(K + \tfrac{1}{2}\bigr) \cos\bigl((K + \tfrac{1}{2}) \cdot 2\pi x / p\bigr) \cdot \frac{1}{\sin(\pi x / p)} - \frac{\sin\bigl((K + \tfrac{1}{2}) \cdot 2\pi x / p\bigr) \cdot \cos(\pi x / p)}{2 \sin^2(\pi x / p)} \biggr].$$

The second derivative is obtained by another application of the quotient rule.

For numerical stability at $x \to 0$, the codebase uses term-by-term differentiation of the sum-of-exponentials form

$$D_p(x) = \sum_{k = -K}^{K} e^{2\pi i k x / p}, \qquad D_p'(x) = \frac{2\pi i}{p} \sum_{k = -K}^{K} k \cdot e^{2\pi i k x / p}, \qquad D_p''(x) = -\frac{4\pi^2}{p^2} \sum_{k = -K}^{K} k^2 \cdot e^{2\pi i k x / p}.$$

These three formulas are implemented as `periodic_cardinal`, `periodic_cardinal_prime`, and `periodic_cardinal_second` in [graphic_zero/surfaces_barycentric.py](../../../graphic_zero/surfaces_barycentric.py) (and their GPU variants in [hermite_barycentric_gpu.py](../../../graphic_zero/hermite_barycentric_gpu.py)).

The derivatives of the cardinal basis $\psi_j$ are then

$$\psi_j'(x) = \frac{1}{p} D_p'(x - x_j), \qquad \psi_j''(x) = \frac{1}{p} D_p''(x - x_j).$$

## 7. Tensor product to $T^2$

The teacher acts on $T^2$, not $T^1$. The cardinal basis lifts directly via tensor product:

$$\Psi_{j, k}(x, y) := \psi_j(x) \cdot \psi_k(y), \qquad (j, k) \in \{0, \ldots, p-1\}^2.$$

These $p^2$ functions span the bivariate trig-polynomial space $\mathcal{T}_p \otimes \mathcal{T}_p \subset L^2(T^2)$, and the cardinal property generalises:

$$\Psi_{j, k}(x_a, x_b) = \delta_{ja} \cdot \delta_{kb}.$$

For $f \in \mathcal{T}_p \otimes \mathcal{T}_p$, the bivariate cardinal interpolant is exact:

$$f(x, y) = \sum_{j, k = 0}^{p-1} f(x_j, x_k) \, \Psi_{j, k}(x, y).$$

The teacher $T(x, y) = e^{2\pi i (x + y) / p}$ factorises as $\chi(x) \cdot \chi(y)$, hence sits in the band-limited subspace spanned by $e^{2\pi i x / p} \cdot e^{2\pi i y / p}$ -- a single tensor-product Fourier mode in $\mathcal{T}_p \otimes \mathcal{T}_p$. The bivariate cardinal interpolant therefore reproduces $T$ exactly on all of $T^2$.

The bivariate gradient and Hessian come from differentiating one factor at a time:

$$\partial_x \Psi_{j, k} = \psi_j'(x) \, \psi_k(y), \quad \partial_y \Psi_{j, k} = \psi_j(x) \, \psi_k'(y), \quad \partial_x^2 \Psi_{j, k} = \psi_j''(x) \, \psi_k(y), \quad \partial_x \partial_y \Psi_{j, k} = \psi_j'(x) \, \psi_k'(y), \quad \partial_y^2 \Psi_{j, k} = \psi_j(x) \, \psi_k''(y).$$

In the codebase these assemble into the teacher fields `T_re`, `T_im`, `GX_re`, `GX_im`, `GY_re`, `GY_im`, `Hxx_re`, `Hxx_im`, `Hxy_re`, `Hxy_im`, `Hyy_re`, `Hyy_im` returned by [build_character_teacher_mesh_periodic](../../../sobolev_distill_character/teacher.py).

## 8. The polynomial Hermite-Birkhoff alternative

A different choice of basis -- not periodic -- comes from polynomial Hermite-Birkhoff interpolation on $[0, p-1]$. Two surfaces are constructed in the existing project:

- **$f_H$**: Hermite tensor-product interpolation matching values *and* first partial derivatives at every lattice node.
- **$f_M$**: Birkhoff / minima-enforcing interpolation matching values, with zero gradient and positive-definite Hessian at every lattice node.

The full construction with derivations is in `bivariate_hermite_birkhoff_math_walkthrough.md` (not shipped with this extract; lives in the parent `Aibeceles` monorepo). The chapter does not duplicate that material; it uses it as a building block.

The mathematical content of $f_H, f_M$ is *polynomial* approximation: each per-axis basis is a Lagrange / Hermite polynomial system on the closed interval $[v_0, v_5]$ (in the original axis-quadratic-table problem) or on $[0, p-1]$ (when adapted to modular arithmetic). These are $p$-dimensional spaces but they are *polynomial* spaces, not trig-polynomial spaces.

## 9. Boundary-value mismatch with $T^2$

**The structural difference.** The polynomial Hermite-Birkhoff teacher is defined on $[0, p-1]^2 \subset \mathbb{R}^2$, *not* on $T^2$. There is no built-in identification of the boundary $\{p-1\} \times [0, p-1]$ with $\{0\} \times [0, p-1]$ (and similarly for the $y$-axis). Three immediate consequences:

1. **Off-lattice $|f_H| \ne 1$.** The polynomial interpolant of $T(x, y) = e^{2\pi i (x + y) / p}$ matches the unit-modulus value at lattice nodes by construction but oscillates above and below $|f| = 1$ between nodes. This is a manifestation of *Runge oscillation* -- the standard failure mode of high-degree polynomial interpolation on equispaced nodes (Trefethen 2000, Chapter 5).
2. **No smooth wrap.** The natural torus identification $(p - 1, 0) \sim (0, 0)$ is not respected. The polynomial $f_H$ takes very different values on the two sides.
3. **Pollution of the Sobolev loss.** Since $|f_H| \ne 1$ off-lattice, the unit-circle penalty $(|f|^2 - 1)^2$ would punish the polynomial teacher's *own* off-lattice values. The teacher and the loss are inconsistent.

By contrast the periodic-cardinal teacher respects the torus identification automatically (chapter 01) and reproduces the character exactly off-lattice (section 5 above). The unit-modulus property holds everywhere on $T^2$ for free.

**Empirical confirmation.** The $\S 11$ delta table in [sobolev_student_character_periodic.ipynb](../sobolev/sobolev_student_character_periodic.ipynb) compares the two teachers under identical student / schedule / seed. The off-lattice value MSE collapses by roughly two orders of magnitude under the periodic teacher; the unit-circle residual moves from $\sim 0.065$ (polynomial) to the float floor (periodic).

## 10. Code anchors

| structure | code |
|---|---|
| Single-axis $\psi_j$ | `periodic_cardinal` in [graphic_zero/surfaces_barycentric.py](../../../graphic_zero/surfaces_barycentric.py) |
| First derivative $\psi_j'$ | `periodic_cardinal_prime` in same file |
| Second derivative $\psi_j''$ | `periodic_cardinal_second` in same file |
| Bivariate teacher $T$, $\nabla T$, $\nabla^2 T$ | [build_character_teacher_mesh_periodic](../../../sobolev_distill_character/teacher.py) |
| GPU kernels | [graphic_zero/hermite_barycentric_gpu.py](../../../graphic_zero/hermite_barycentric_gpu.py) |
| Tests of cardinality, single-mode reproduction, and PD certificate | [graphic_zero_character/tests/test_periodic_cardinal.py](../../../graphic_zero_character/tests/test_periodic_cardinal.py) and [sobolev_distill_character/tests/test_teacher_periodic.py](../../../sobolev_distill_character/tests/test_teacher_periodic.py) |
| Polynomial Hermite-Birkhoff alternative | derivation in `bivariate_hermite_birkhoff_math_walkthrough.md` (not shipped); code in [graphic_zero/surfaces_barycentric.py](../../../graphic_zero/surfaces_barycentric.py) and [sobolev_distill_character/teacher.py:build_character_teacher_mesh](../../../sobolev_distill_character/teacher.py) |

## 11. Further reading

- **L. N. Trefethen**, *Spectral Methods in MATLAB*, SIAM, 2000. Chapters on the Dirichlet kernel and trigonometric interpolation; Runge oscillation in Chapter 5 of *Approximation Theory and Approximation Practice* (SIAM, 2013).
- **A. Berlinet and C. Thomas-Agnan**, *Reproducing Kernel Hilbert Spaces in Probability and Statistics*, Springer, 2004. Chapter 1 for the general RKHS framework; Chapter 2 for trig-polynomial RKHS examples.
- **G. B. Folland**, *Real Analysis: Modern Techniques and Their Applications*, 2nd ed., Wiley, 1999. Sections on $L^2$ orthonormal bases on the torus.
- **J. P. Boyd**, *Chebyshev and Fourier Spectral Methods*, 2nd ed., Dover, 2001. Comprehensive comparison of Chebyshev (non-periodic) and Fourier (periodic) cardinal bases.

---

Cross-reference: chapter 03 uses these analytic teacher targets as the right-hand side of the Sobolev distillation loss. Chapter 04 uses the band-limit $|k| \le K$ structure of $\mathcal{T}_p$ as the support of the Pontryagin dual.
