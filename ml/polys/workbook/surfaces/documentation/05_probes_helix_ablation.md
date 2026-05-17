# 05 - Probes, helix construction, and causal ablation

The linear-algebra layer that turns the harmonic-analysis claims of chapter 04 into measurable interpretability statements about the trunk's representation. Linear probes via ridge regression read off Fourier features. SVD on lattice activations exhibits the helix manifold, whose winding number is a topological invariant set by the teacher's homotopy class. Causal subspace ablation closes the loop by testing whether the readout actually depends on the cos / sin axes.

## Sections

1. The trunk's lattice activations as a data matrix
2. Linear probes via ridge regression
3. Coefficient of determination $R^2$
4. SVD and PCA on lattice activations
5. The helix construction
6. The PC1-PC2 loop as the image of $T^1$ under the trunk
7. Wrap angle as winding number
8. Orthogonal subspace projection $P_\perp = I - V V^\top$
9. Causal subspace ablation and the three contrast subspaces
10. Code: where this chapter is realised
11. Further reading

---

## 1. The trunk's lattice activations as a data matrix

Throughout this chapter, the **trunk** is the SIREN-based map $\mathrm{trunk}_\theta : T^2 \to \mathbb{R}^D$ from input coordinates to a $D$-dimensional embedding (see [model.py:Trunk](../../../sobolev_distill_character/model.py); we use $D = 32$ in the canonical config and $D = 8, 16$ in the bottlenecked variants).

For a given trained student, evaluate the trunk on the lattice $\Lambda = (\mathbb{Z}/p\mathbb{Z})^2$ and stack the activations into a matrix:

$$H_{\mathrm{lat}} \in \mathbb{R}^{p^2 \times D}, \qquad (H_{\mathrm{lat}})_{(i, j), n} := \mathrm{trunk}_\theta(i, j)_n.$$

This matrix is the input to every probe in the chapter. It is constructed by [probes.py:compute_latents_character](../../../sobolev_distill_character/probes.py) and stored in `CharacterLatentArrays.H_lat`.

## 2. Linear probes via ridge regression

**Definition.** Let $A \in \mathbb{R}^{N \times D}$ and $y \in \mathbb{R}^N$. The **ridge regression problem** with regularisation $\lambda > 0$ is

$$w^* := \arg\min_{w \in \mathbb{R}^D} \|A w - y\|_2^2 + \lambda \|w\|_2^2.$$

The closed-form solution is $w^* = (A^\top A + \lambda I)^{-1} A^\top y$. The penalty $\lambda \|w\|^2$ is a Tikhonov stabiliser; $\lambda \to 0$ recovers ordinary least squares.

**Application to the trunk.** [probes.py:linear_probes_character](../../../sobolev_distill_character/probes.py) treats $A := H_{\mathrm{lat}}$ as the design matrix and runs ridge regression to predict each of seven canonical Fourier-feature targets:

$$y \in \bigl\{ \mathrm{Re}\, T(i, j), \mathrm{Im}\, T(i, j), \mathrm{Re}\, \zeta^i, \mathrm{Im}\, \zeta^i, \mathrm{Re}\, \zeta^j, \mathrm{Im}\, \zeta^j, (i + j) \bmod p \bigr\},$$

where $\zeta = e^{2\pi i / p}$ as in chapter 01. The first two test whether the trunk linearly carries the teacher target; the next four test per-axis Fourier features; the last is a sanity check that modular accuracy is recoverable.

The ridge constant defaults to $\lambda = 10^{-3}$ in the codebase, with a 80 / 20 train / test split on lattice rows for honest $R^2$ reporting.

**Why ridge instead of OLS.** With $D = 32$ and $N = p^2$, the design matrix is well-conditioned for $p \ge 6$. Ridge is a precaution against degenerate cases and a continuous interpolation between OLS and zero; the ridge constant is small enough that it does not bias the reported $R^2$ in the regime of interest. See Folland (1999, Section 6.6) for the equivalent of ridge regression as a best-approximation problem in $L^2$.

## 3. Coefficient of determination $R^2$

For the held-out set $(A_{\mathrm{test}}, y_{\mathrm{test}})$, the **coefficient of determination** is

$$R^2 := 1 - \frac{\|A_{\mathrm{test}} w^* - y_{\mathrm{test}}\|_2^2}{\|y_{\mathrm{test}} - \bar y_{\mathrm{test}}\|_2^2}.$$

$R^2 \in (-\infty, 1]$. Values $R^2 \approx 1$ indicate that the linear model captures essentially all the variance in $y_{\mathrm{test}}$; $R^2 = 0$ matches the constant-mean predictor; $R^2 < 0$ indicates that the linear model is worse than the constant.

The probes report `r2_test` and `r2_train` in `LinearProbeReport` (see [probes.py:linear_probes_character](../../../sobolev_distill_character/probes.py)). The standard reporting convention in the writeup is the test $R^2$ unless otherwise noted.

## 4. SVD and PCA on lattice activations

Let $\bar H := \frac{1}{p^2} \sum_{(i, j)} H_{\mathrm{lat}, (i, j)}$ be the centroid of the lattice activations. Centre:

$$H_c := H_{\mathrm{lat}} - \mathbf{1} \bar H^\top \in \mathbb{R}^{p^2 \times D}.$$

The **singular value decomposition** (SVD) is

$$H_c = U \Sigma V^\top, \qquad U \in \mathbb{R}^{p^2 \times r}, \; \Sigma \in \mathbb{R}^{r \times r}, \; V \in \mathbb{R}^{D \times r}, \; r = \min(p^2, D).$$

The columns of $V$ are the **principal components** in $\mathbb{R}^D$, ordered by decreasing singular value $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r \ge 0$. The **explained variance** of the $k$-th component is $\sigma_k^2 / (p^2 - 1)$ and the cumulative explained-variance ratio is the standard PCA scree plot.

The first two principal components $V_{:, 1}, V_{:, 2}$ define a 2-d subspace of $\mathbb{R}^D$. The **PC1-PC2 coordinates** of each lattice point are

$$\mathrm{coords}_{(i, j)} := H_c \cdot V_{:, 1:2} \in \mathbb{R}^{p^2 \times 2}.$$

This is the visual the experimental notebooks plot when they speak of "PC1 vs PC2 of trunk activations".

## 5. The helix construction

[mechinterp.py:helix_pca](../../../sobolev_distill_character/mechinterp.py) does *not* run PCA on the full $p^2 \times D$ matrix. Instead it averages along one axis first, exposing the per-axis structure:

**Step 1 (axis averaging).** Fix axis = `'i'`. For each $i \in \mathbb{Z}/p\mathbb{Z}$, average the trunk activations across $j$:

$$\bar H_i := \frac{1}{p} \sum_{j = 0}^{p - 1} H_{\mathrm{lat}, (i, j)} \in \mathbb{R}^D.$$

This collapses the $p^2 \times D$ matrix to a $p \times D$ matrix $\bar H_{\mathrm{axis}-i}$ indexed by axis-i coordinate.

**Step 2 (centre and SVD).** Centre $\bar H_{\mathrm{axis}-i}$ by subtracting its row-mean centroid; SVD; take PC1 and PC2.

**Step 3 (frequency fit).** For each candidate frequency $k = 1, 2, \ldots, \lfloor p / 2 \rfloor$, fit the parametric curve $i \mapsto (\cos(2\pi k i / p), \sin(2\pi k i / p))$ jointly into the PC1-PC2 coordinates by least squares (with an affine intercept), and report the joint $R^2$ across both PC axes:

$$R^2_k := 1 - \frac{\sum_{i} \| \mathrm{coords}_i - \mathrm{coords}_i^{\mathrm{fit}, k} \|_2^2}{\sum_{i} \| \mathrm{coords}_i - \overline{\mathrm{coords}} \|_2^2}.$$

The dominant frequency $k^* := \arg\max_k R^2_k$ is the helix's principal mode; $R^2_{k^*}$ is reported as `helix_r2_i`.

**Step 4 (wrap angle).** Compute the angle $\phi_i := \arg(\mathrm{coords}_i - \overline{\mathrm{coords}})$ of each PC1-PC2 sample relative to the centroid, and integrate the **cumulative signed angular sweep** around the loop, including the closing step $i = p - 1 \to i = 0$:

$$w := \sum_{i = 0}^{p - 1} \mathrm{wrap}(\phi_{i + 1 \bmod p} - \phi_i),$$

where $\mathrm{wrap}(\Delta\phi) := \mathrm{atan2}(\sin\Delta\phi, \cos\Delta\phi)$ takes $\Delta\phi$ to its principal value in $(-\pi, \pi]$. For a single-mode helix at frequency $k$, the wrap angle is $w = \pm 2\pi k$ (sign depending on the orientation of the SVD basis).

The full procedure is implemented in [mechinterp.py:helix_pca](../../../sobolev_distill_character/mechinterp.py) and returns a `HelixReport` containing the components, coords, explained-variance ratios, $R^2_k$ table, dominant $k^*$, and wrap angle.

## 6. The PC1-PC2 loop as the image of $T^1$ under the trunk

**Geometric interpretation.** The axis-averaged trunk $\bar H_{\mathrm{axis}-i}$ is a function of one variable $i \in \mathbb{Z}/p\mathbb{Z} \cong T^1$, taking values in $\mathbb{R}^D$. Composing with the projection $V_{:, 1:2}^\top : \mathbb{R}^D \to \mathbb{R}^2$ gives a map

$$\Phi_i : T^1 \longrightarrow \mathbb{R}^2, \qquad \Phi_i(s) = V_{:, 1:2}^\top (\bar H_{s} - \overline{\bar H}).$$

The image of $\Phi_i$ is a closed curve in $\mathbb{R}^2$ -- specifically, the PC1-PC2 trace plotted in the helix figures.

**Homotopy class.** Pick a point $z_0 \in \mathbb{R}^2$ that the loop encloses (the centroid is the natural choice). The map $\Phi_i$ then has a homotopy class in $\pi_1(\mathbb{R}^2 \setminus \{z_0\}) \cong \mathbb{Z}$, classified by the winding number $w / 2\pi$.

**Proposition (winding from the teacher's character).** If the trunk has internalised the character $\chi$ correctly, the dominant frequency $k^*$ should be $1$ and the winding number $w / 2\pi$ should be $\pm 1$.

**Justification.** The teacher's homotopy class on $T^2$ has winding $(1, 1)$ along the canonical generators (chapter 01, section 6). The axis-averaged trunk implements one of the two generators (varying $i$, averaging $j$); its winding number under the teacher's prescription is $1$. The PC1-PC2 projection preserves winding (it is a rank-2 linear map onto the dominant 2-d feature subspace). So the wrap angle should be $\pm 2\pi$ in absolute value.

The sign depends on the orientation of the SVD basis (PC1 and PC2 are unique up to simultaneous reflection); $|w / 2\pi| = 1$ is the orientation-invariant statement.

**Empirical confirmation.** The bottleneck cells `(embed_dim = 8, trunk_hidden = 16)` and `(8, 32)` at $p = 17$ in [grokking_capacity_sweep.ipynb](../sobolev/grokking/grokking_capacity_sweep.ipynb) come back with `helix_r2_i = +0.85, +0.86` and `wrap_i / 2π = +1, -1`. The trunk has the right topology.

## 7. Wrap angle as winding number

The wrap-angle formula in section 5 step 4 is the discrete version of the standard winding-number integral

$$w = \oint_\Phi d\arg = \int_0^1 \frac{(\Phi_2 \, \dot\Phi_1 - \Phi_1 \, \dot\Phi_2)}{\Phi_1^2 + \Phi_2^2} \, dt,$$

where $\Phi(t)$ is a continuous parametrisation of the loop with the centroid at the origin. For a single-mode helix at frequency $k$, $\Phi(t) = R(\cos 2\pi k t, \sin 2\pi k t)$ and the integral evaluates to $2\pi k$ exactly.

The discrete sum approximates this contour integral with $p$ samples and is exact when the loop is a closed polygon -- which is precisely the case here.

**Closed-loop sanity test.** A clean single-frequency helix passes three tests jointly:

1. `helix_r2_i ≥ 0.85` (dominant fit explains most variance);
2. `|wrap_i / 2π| = 1` to within numerical tolerance (closed loop, winding 1);
3. The dominant frequency $k^*$ matches the expected value (1 for the canonical character).

All three are reported by `helix_pca` and tabulated in the experimental notebooks. See Hatcher (2002, Section 1.1) for the topological background on winding numbers and $\pi_1(S^1) = \mathbb{Z}$.

## 8. Orthogonal subspace projection

**Definition.** Let $V = \{v_1, \ldots, v_r\} \subset \mathbb{R}^D$ be a set of $r$ orthonormal vectors, and assemble them into a matrix $V \in \mathbb{R}^{r \times D}$. The **orthogonal projection onto the complement of $\mathrm{span}(V)$** is

$$P_\perp \;:=\; I_D - V^\top V,$$

a symmetric idempotent of rank $D - r$. For any $h \in \mathbb{R}^D$, $P_\perp h$ is the component of $h$ orthogonal to every $v_k$.

If the input vectors are not orthonormal, [mechinterp.py:_orthonormalise](../../../sobolev_distill_character/mechinterp.py) applies a Gram-Schmidt (via QR) before constructing $P_\perp$. So every $V$ in the codebase is treated as if its rows had been orthonormalised, which is the convention assumed in the rest of this chapter.

The image of $P_\perp$ is the orthogonal complement $V^\perp \subset \mathbb{R}^D$.

## 9. Causal subspace ablation

[mechinterp.py:ablate_subspace_and_score](../../../sobolev_distill_character/mechinterp.py) replaces the trunk's output by its $V^\perp$ component, then re-evaluates modular accuracy through the rest of the network:

$$\mathrm{trunk}_\theta(x, y) \;\mapsto\; P_\perp \cdot \mathrm{trunk}_\theta(x, y), \qquad \mathrm{accuracy}_{\mathrm{abl}}(V) := \mathrm{modular\_accuracy}\bigl(\mathrm{head}_a(P_\perp \cdot \mathrm{trunk}_\theta)\bigr).$$

The **causal effect** of subspace $V$ is the drop $\mathrm{accuracy}_{\mathrm{baseline}} - \mathrm{accuracy}_{\mathrm{abl}}(V)$. A large drop means the readout depends on the subspace; a small drop means the subspace is causally irrelevant.

The framework follows the standard mechanistic-interpretability convention of Nanda et al. (2023) and the Anthropic line of work; the projection $P_\perp$ is the canonical operation. See `AblationReport` in [mechinterp.py](../../../sobolev_distill_character/mechinterp.py) for the data structure that records the comparison.

**The three contrast subspaces.** [manifold_and_ablation.ipynb](../sobolev/grokking/manifold_and_ablation.ipynb) section 5 compares three choices of $V$ for each variant:

1. **Readout cos / sin axes** -- $V$ is the rank-4 subspace spanned by the linear-probe coefficients for $\mathrm{Re}\,\zeta^i, \mathrm{Im}\,\zeta^i, \mathrm{Re}\,\zeta^j, \mathrm{Im}\,\zeta^j$.
   *Expectation*: modular accuracy collapses for variants whose mechanism really is the per-axis cos / sin features (`B_fourier`, `C_factored`, `A_axis_loss`).
2. **Random matched-norm** -- $V$ is a random Gaussian subspace with the same Frobenius norm and rank as the readout subspace.
   *Expectation*: modular accuracy stays near baseline; the randomness cannot hit the readout's cos / sin axes by chance.
3. **Top-2 PC of helix** -- $V$ is the rank-2 subspace spanned by the first two SVD components from the helix construction.
   *Expectation*: collapses iff the helix's PC1-PC2 plane overlaps the readout subspace; partial collapse indicates the helix encodes the same information as the readout but in different coordinates.

The **gap** between the readout column and the random column is the strongest causal evidence in this framework that the readout's cos / sin axes are *the* mechanism, not just a candidate description.

## 10. Code anchors

| structure | code |
|---|---|
| Latent matrix $H_{\mathrm{lat}}$ | [probes.py:compute_latents_character](../../../sobolev_distill_character/probes.py) returning `CharacterLatentArrays` |
| Linear probes (ridge) | [probes.py:linear_probes_character](../../../sobolev_distill_character/probes.py) |
| Per-target $R^2$ | `LinearProbeReport.r2` |
| SVD / PCA on axis-averaged activations | [mechinterp.py:helix_pca](../../../sobolev_distill_character/mechinterp.py), lines 594-685 |
| Wrap-angle computation | within `helix_pca`, lines 663-672 |
| Orthonormalisation (Gram-Schmidt via QR) | [mechinterp.py:_orthonormalise](../../../sobolev_distill_character/mechinterp.py), lines 714-731 |
| Causal ablation projection | [mechinterp.py:ablate_subspace_and_score](../../../sobolev_distill_character/mechinterp.py), lines 734+ |
| The three contrast subspaces in code | section 5 of [manifold_and_ablation.ipynb](../sobolev/grokking/manifold_and_ablation.ipynb) |

## 11. Further reading

- **G. B. Folland**, *Real Analysis: Modern Techniques and Their Applications*, 2nd ed., Wiley, 1999. Section 6.6 for the variational characterisation of orthogonal projections; the underlying $L^2$ best-approximation framework that the linear probes specialise.
- **A. Hatcher**, *Algebraic Topology*, Cambridge University Press, 2002. Section 1.1 for $\pi_1(S^1) = \mathbb{Z}$ and the winding number; Section 1.3 for covering spaces.
- **G. H. Golub and C. F. Van Loan**, *Matrix Computations*, 4th ed., JHU Press, 2013. SVD, ridge regression, Gram-Schmidt; canonical numerical-linear-algebra reference.
- **N. Nanda et al.**, *Progress measures for grokking via mechanistic interpretability*, ICLR 2023. Originator of the causal-subspace-ablation methodology used here.
- **Anthropic**, *Helices in Claude 3.5 Haiku for newline prediction*, Anthropic interpretability blog (2024). The `helix_pca` framework is consistent with the Anthropic team's helix-finding pipeline; the chapter's geometric framing in terms of $\pi_1$ matches their winding-number language.

---

This is the final chapter of the math documentation. The five chapters together establish: (01) the geometric setup on $T^2$, (02) the analytic interpolation framework on the lattice, (03) the functional-analytic loss, (04) the harmonic-analysis view of the resulting representations, and (05) the linear-algebra layer that turns those representations into measurable interpretability claims. Cross-references to the experimental notebooks throughout point to where each claim is verified empirically.
