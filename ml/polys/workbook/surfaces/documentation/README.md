# Mathematical documentation for `surfaces/`

Math-first piece of the `surfaces/` writeup. Five chapters covering the geometric, analytic, and algebraic background of the character / Sobolev / Fourier work, plus this index.

This folder is the *reference* layer of the documentation. Experiments live in the notebooks themselves; engineering details (JAX, Equinox, optax, SIREN architecture) live elsewhere.

## Reading order

```
README.md                                  -- this file
00_commentary.md                           -- the narrative companion (story layer)
01_torus_and_character.md                  -- the geometric setup
02_interpolation_cardinals_rkhs.md         -- approximation theory on T^1 and T^2
03_sobolev_distillation.md                 -- functional analysis of the loss
04_dft_pontryagin_trig_identity.md         -- harmonic analysis on T^2
05_probes_helix_ablation.md                -- linear algebra for interpretability
figures/                                   -- the nine PNG figures embedded in 00
_export_figures.py                         -- regeneration script for figures/
```

The five chapters 01-05 are sequential: each presupposes the definitions of those before it. A reader who only wants the trig-identity / Fourier-features story can jump from chapter 1 to chapter 4 directly.

## Companion: narrative expository

The math chapters above are the *reference* layer. The companion *narrative* layer is in [00_commentary.md](00_commentary.md) (with figures in [figures/](figures/) regenerated from [_export_figures.py](_export_figures.py)). Readers who want the story can read 00 alone; readers who want the proofs follow the back-references into chapters 01-05. The commentary is structured around the seven-probe checklist of the Welch Labs / Nanda video, with each probe's figure at the modulus that best demonstrates the claim (p = 113 for the Tier-1 reproductions, p = 17 for the bottleneck and gradient-field figures, p = 8 for the variants comparison).

## Mapping `surfaces/` subfolders to chapters

> **Note (this extract).** This repo is the minimum carve-out needed to render
> `00_commentary.md` and regenerate its figures. The folders marked *(not
> shipped)* below live in the parent `Aibeceles` monorepo only; their links
> here will 404. The `sobolev/` rows are the ones that *are* shipped (only the
> seven notebooks the expository references).

| `surfaces/` subfolder | primary chapters | also relevant | external cross-reference |
|---|---|---|---|
| `bivariate/` *(not shipped)* | 02 | 04 | `bivariate_hermite_birkhoff_math_walkthrough.md` *(not shipped)* |
| `hermite_birkhoff/` *(not shipped)* | 02 | -- | same walkthrough as above |
| `character/` *(not shipped)* | 01, 02, 04 | 03 | -- |
| `quadratics/` *(not shipped)* | 02 | -- | same Hermite-Birkhoff walkthrough |
| [`sobolev/`](../sobolev/) | 03 | 01, 02 | -- |
| [`sobolev/grokking/`](../sobolev/grokking/) | 04, 05 | 01, 03 | -- |
| this folder, narrative layer | [00_commentary.md](00_commentary.md) | all of 01-05 | -- |

The polynomial Hermite-Birkhoff teacher derivation (referenced as
`bivariate_hermite_birkhoff_math_walkthrough.md`) is not shipped with this
extract; this folder treats it as a building block and refers out for the full
construction. The corresponding code lives in
[graphic_zero/surfaces_barycentric.py](../../../graphic_zero/surfaces_barycentric.py)
and
[graphic_zero/hermite_barycentric_gpu.py](../../../graphic_zero/hermite_barycentric_gpu.py),
both of which are shipped.

## Notation glossary

| symbol | meaning | first defined in |
|---|---|---|
| $T^n$ | the $n$-torus $(\mathbb{R} / \mathbb{Z})^n$ (occasionally rescaled to $(\mathbb{R}/p\mathbb{Z})^n$ for the modular-arithmetic problem) | 01 |
| $S^1$ | the unit circle $\{z \in \mathbb{C} : |z| = 1\}$ | 01 |
| $\mathbb{Z}/p\mathbb{Z}$ | the cyclic group of order $p$ | 01 |
| $\zeta$ | the primitive $p$-th root of unity $e^{2\pi i / p}$ | 01 |
| $\chi$ | the fundamental character $\chi : \mathbb{Z}/p\mathbb{Z} \to S^1$, $\chi(s) = \zeta^s$ | 01 |
| $\Lambda$ | the lattice $(\mathbb{Z}/p\mathbb{Z})^2 \subset T^2$ | 01 |
| $T(x,y)$ | the teacher map $T^2 \to S^1$, $T(x,y) = \chi(x+y)$ | 01 |
| $\Gamma$ | the graph of $T$ as a 2-manifold in $T^2 \times \mathbb{R}^2$ | 01 |
| $\widehat{G}$ | the Pontryagin dual of an LCA group $G$ | 04 |
| $\mathcal{T}_p$ | the trig-polynomial space $\mathrm{span}\{e^{2\pi i k x} : -\lfloor p/2 \rfloor \le k \le \lfloor p/2 \rfloor\} \subset L^2(T^1)$ | 02 |
| $D_p$ | the Dirichlet kernel; reproducing kernel of $\mathcal{T}_p$ | 02 |
| $\psi_j$ | the cardinal basis function at node $x_j$ | 02 |
| $H^k(T^2)$ | the Sobolev space of order $k$ on the 2-torus | 03 |
| $\hat f(n)$ | the Fourier coefficient of $f$ at $n \in \widehat{T^2} = \mathbb{Z}^2$ | 04 |
| $\mathrm{cc}, \mathrm{ss}, \mathrm{cs}, \mathrm{sc}$ | the four real product channels of the 2-D DFT | 04 |
| $H_\mathrm{lat}$ | the trunk's lattice activations, a $(K_\mathrm{lat} \times D)$ matrix | 05 |
| $P_\perp$ | the orthogonal projection $I - V V^\top$ onto the complement of a subspace $V$ | 05 |
| $w$ | the wrap angle of a closed PC1-PC2 trace; $w / 2\pi$ is the winding number | 05 |

## References

Each chapter has its own `Further reading` section pointing at the standard texts. The combined list:

- **G. B. Folland**, *A Course in Abstract Harmonic Analysis*, 2nd ed., CRC Press, 2016. Pontryagin duality, characters of LCA groups.
- **D. Bump**, *Automorphic Forms and Representations*, Cambridge University Press, 1998. Characters of finite abelian groups (chapter 3).
- **R. A. Adams and J. J. F. Fournier**, *Sobolev Spaces*, 2nd ed., Academic Press, 2003. $H^k$ definitions, embedding theorems.
- **A. Berlinet and C. Thomas-Agnan**, *Reproducing Kernel Hilbert Spaces in Probability and Statistics*, Springer, 2004. RKHS framework, Dirichlet kernel as reproducing kernel.
- **A. Hatcher**, *Algebraic Topology*, Cambridge University Press, 2002. Fundamental group, winding number, fibrations.
- **L. N. Trefethen**, *Approximation Theory and Approximation Practice*, SIAM, 2013. Cardinal interpolation, spectral methods, Chebyshev points (extended periodic version in *Spectral Methods in MATLAB*, SIAM 2000).

## Editorial conventions

- LaTeX block math uses `$$ ... $$`; inline uses `$ ... $`.
- Code references use markdown links to the relevant file or function:
  [helix_pca](../../../sobolev_distill_character/mechinterp.py).
- Definitions and propositions are explicitly marked: **Definition.**, **Proposition.**, **Theorem.**, **Proof.** The proofs are short; longer derivations are sketched and cited.
- No emojis. Project-specific terminology (*trunk*, *probe*, *axis loss*, *teacher*) is defined where it first appears.

## Status

The chapters are written for a fixed snapshot of the codebase; numerical results from individual notebook runs do not appear here and are not subject to drift. The framework that *predicts* those results (and the structural reasons it works) is what these chapters cover.
