# 04 - DFT, Pontryagin duality, and the trig identity

The harmonic-analysis core of the mechinterp probes. Establishes Pontryagin duality on $T^2$, the discrete Fourier transform on the lattice $\Lambda$, the real-form parity decomposition into the four product channels $\{\mathrm{cc}, \mathrm{ss}, \mathrm{cs}, \mathrm{sc}\}$, and the derivation of the trig identity as the real form of $\chi(x + y) = \chi(x) \chi(y)$.

## Sections

1. Pontryagin duality on the torus
2. Restriction to the lattice and the discrete Fourier transform
3. The 1-D DFT along an axis
4. The 2-D FFT of a neuron surface
5. Real-form parity decomposition into four product channels
6. The trig identity as a real-form character identity
7. The sum-of-angles score as a per-neuron identity test
8. Excluded loss as Fourier-mode projection
9. Code: where this chapter is realised
10. Further reading

---

## 1. Pontryagin duality on the torus

**Definition.** Let $G$ be a locally compact abelian (LCA) group. The **Pontryagin dual** $\widehat{G}$ is the group of continuous group homomorphisms $\xi : G \to S^1$, equipped with pointwise multiplication and the compact-open topology.

**Examples (basic).**
- $\widehat{\mathbb{R}} = \mathbb{R}$, with the duality $\xi(x) = e^{2\pi i \xi x}$.
- $\widehat{T^1} = \mathbb{Z}$, with the duality $\xi(x) = e^{2\pi i \xi x / p}$ for $\xi \in \mathbb{Z}$ (using our $p$-rescaled convention).
- $\widehat{\mathbb{Z}/p\mathbb{Z}} = \mathbb{Z}/p\mathbb{Z}$, with the duality $\xi(s) = \zeta^{\xi s}$.

**Proposition.** $\widehat{T^2} = \mathbb{Z}^2$, with the pairing

$$\xi_n(x, y) = e^{2\pi i (n_1 x + n_2 y) / p}, \qquad n = (n_1, n_2) \in \mathbb{Z}^2.$$

The teacher $T$ is the single character $\xi_{(1, 1)}$:

$$T(x, y) = e^{2\pi i (x + y) / p} = \xi_{(1, 1)}(x, y).$$

So the teacher *is* a single point in $\widehat{T^2}$. Every Fourier-feature claim later in this chapter ultimately rests on this observation. See Folland (2016, Chapter 4) and Bump (1998, Chapter 3) for the general theory.

## 2. Restriction to the lattice

The lattice $\Lambda = (\mathbb{Z}/p\mathbb{Z})^2 \subset T^2$ has its own Pontryagin dual

$$\widehat{\Lambda} = (\mathbb{Z}/p\mathbb{Z})^2,$$

with the $p^2$ characters

$$\xi_k : \Lambda \to S^1, \qquad \xi_k(i, j) = e^{2\pi i (k_1 i + k_2 j) / p}, \qquad k = (k_1, k_2) \in \widehat{\Lambda}.$$

This is the natural setting for the **discrete Fourier transform**: for any function $f : \Lambda \to \mathbb{C}$,

$$\widehat{f}(k) := \sum_{(i, j) \in \Lambda} f(i, j) \, \overline{\xi_k(i, j)} = \sum_{i, j} f(i, j) \, e^{-2\pi i (k_1 i + k_2 j) / p}.$$

The DFT is the canonical isomorphism $L^2(\Lambda) \to L^2(\widehat{\Lambda})$. Its inverse is the same map with a sign change in the exponent (and a $1/p^2$ normalisation).

**Restriction to a single axis.** Fixing $j = 0$ (or any other constant) gives a 1-d slice. Its DFT is

$$\widehat{f|_{j = 0}}(k_1) = \sum_{i = 0}^{p - 1} f(i, 0) \, e^{-2\pi i k_1 i / p}, \qquad k_1 \in \mathbb{Z}/p\mathbb{Z}.$$

This is the operation that `dft_trunk_along_axis` in [mechinterp.py](../../../sobolev_distill_character/mechinterp.py) performs per neuron.

## 3. The 1-D DFT along an axis

For each trunk neuron $n \in \{0, \ldots, D-1\}$, [mechinterp.py:dft_trunk_along_axis](../../../sobolev_distill_character/mechinterp.py) evaluates the trunk activation as a function of one input axis (with the other axis fixed) and computes the $p$-point DFT. The resulting $(D, p)$ complex spectrum is summarised by

- the **dominant non-DC frequency** per neuron, $k_n^* := \arg\max_{k \in \{1, \ldots, \lfloor p/2 \rfloor\}} |\widehat{H_n}(k)|$;
- the **mode concentration**, $|\widehat{H_n}(k_n^*)|^2 / \sum_{k \ne 0} |\widehat{H_n}(k)|^2$;
- a **histogram** of dominant frequencies across the $D$ neurons.

For the modular-arithmetic problem we expect the histogram to spike at $k = 1$ -- the character's own Fourier mode -- and to be empty elsewhere. This is exactly the visualisation in [modulus_sweep.ipynb](../sobolev/grokking/modulus_sweep.ipynb) section 5: a few clean spikes among $\lfloor p/2 \rfloor$ candidate frequencies.

**Why $k_1 \in \{1, \ldots, \lfloor p/2 \rfloor\}$ and not $\{1, \ldots, p-1\}$.** For real-valued activations, $\widehat{f}(p - k) = \overline{\widehat{f}(k)}$, so $|\widehat{f}(p - k)| = |\widehat{f}(k)|$. The unique-up-to-conjugation modes are indexed by $k \in \{0, 1, \ldots, \lfloor p/2 \rfloor\}$. Excluding $k = 0$ (the DC component) leaves the $\lfloor p/2 \rfloor$ non-trivial modes.

## 4. The 2-D FFT of a neuron surface

For each neuron, [mechinterp.py:fft2_neuron_surface](../../../sobolev_distill_character/mechinterp.py) evaluates the trunk on the full $p \times p$ lattice $\Lambda$ and computes the 2-D DFT, producing a $(D, p, p)$ complex spectrum.

The key observation is that the 2-D DFT decomposes any $f : \Lambda \to \mathbb{R}$ into modes indexed by $(k_1, k_2) \in \widehat{\Lambda}$. The spectrum encodes the same information as the function but in the dual basis.

In our setting the trunk activation $H_n(i, j)$ is a real function on $\Lambda$. The DFT $\widehat{H_n}(k_1, k_2)$ is generally complex; the symmetry $\widehat{H_n}(-k_1, -k_2) = \overline{\widehat{H_n}(k_1, k_2)}$ holds. This is the conjugate-symmetry constraint we exploit next.

## 5. Real-form parity decomposition

For real-valued $f$, the conjugate symmetry $\widehat{f}(-k) = \overline{\widehat{f}(k)}$ allows a decomposition into purely real coefficients indexed by parity. For a single mode pair $(k_1, k_2)$ with $k_1, k_2 > 0$, write the contribution to $f$ as

$$\widehat{f}(k_1, k_2) e^{2\pi i (k_1 x + k_2 y)/p} + \widehat{f}(-k_1, -k_2) e^{-2\pi i (k_1 x + k_2 y)/p} + \widehat{f}(k_1, -k_2) e^{2\pi i (k_1 x - k_2 y)/p} + \widehat{f}(-k_1, k_2) e^{-2\pi i (k_1 x - k_2 y)/p}.$$

Pairing terms by conjugate symmetry and applying $e^{i\theta} + e^{-i\theta} = 2 \cos\theta$, $e^{i\theta} - e^{-i\theta} = 2i \sin\theta$, the four-mode contribution rewrites as

$$a_{++}(k_1, k_2) \cos\Bigl(\tfrac{2\pi k_1 x}{p}\Bigr) \cos\Bigl(\tfrac{2\pi k_2 y}{p}\Bigr) + a_{--}(k_1, k_2) \sin\Bigl(\tfrac{2\pi k_1 x}{p}\Bigr) \sin\Bigl(\tfrac{2\pi k_2 y}{p}\Bigr) + a_{+-}(k_1, k_2) \cos\sin + a_{-+}(k_1, k_2) \sin\cos$$

with real coefficients $a_{**}$ given by linear combinations of $\widehat{f}(\pm k_1, \pm k_2)$. We label these four channels

$$\mathrm{cc} := a_{++}, \quad \mathrm{ss} := a_{--}, \quad \mathrm{cs} := a_{+-}, \quad \mathrm{sc} := a_{-+},$$

corresponding to the parity of the cosine / sine factor in $x$ and $y$. The construction is exactly the one in [mechinterp.py:fft2_neuron_surface](../../../sobolev_distill_character/mechinterp.py) (`channel_energy["cos_cos"]`, `["sin_sin"]`, etc.), with the *energy* in each channel reported as $|a_{**}|^2$ aggregated over the neuron axis.

The four-channel decomposition is exhaustive: any real function on $\Lambda$ is a sum of $\mathrm{cc} + \mathrm{ss} + \mathrm{cs} + \mathrm{sc}$ terms across all positive-frequency mode pairs (the DC and Nyquist edges contribute trivially in either parity).

## 6. The trig identity as a real-form character identity

**Theorem (sum-of-angles).**

$$\cos\theta_+ \cos\theta_- - \sin\theta_+ \sin\theta_- \;=\; \cos(\theta_+ + \theta_-),$$

$$\sin\theta_+ \cos\theta_- + \cos\theta_+ \sin\theta_- \;=\; \sin(\theta_+ + \theta_-).$$

**Proof.** Apply the group homomorphism $\chi(\theta_+ + \theta_-) = \chi(\theta_+) \chi(\theta_-)$ at the real-form level: write $\chi(\theta) = \cos\theta + i \sin\theta$ for $\theta = 2\pi s / p$, then

$$\chi(\theta_+ + \theta_-) = \chi(\theta_+) \chi(\theta_-) = (\cos\theta_+ + i \sin\theta_+)(\cos\theta_- + i \sin\theta_-)$$

$$= (\cos\theta_+ \cos\theta_- - \sin\theta_+ \sin\theta_-) + i(\sin\theta_+ \cos\theta_- + \cos\theta_+ \sin\theta_-).$$

Equate real and imaginary parts. $\square$

**Why this matters mechanistically.** The teacher is $T(x, y) = \chi(x + y)$. Its real part is

$$\mathrm{Re}\,T(x, y) = \cos\bigl(2\pi(x + y) / p\bigr) = \mathrm{cc}(x, y) - \mathrm{ss}(x, y)$$

at $(k_1, k_2) = (1, 1)$. So *the trig-identity neuron contributes equal magnitude in the cc channel and the ss channel, with opposite signs*. This is the mechanistic content of "the network solves modular addition by computing $\cos x \cos y - \sin x \sin y$".

The imaginary part:

$$\mathrm{Im}\,T(x, y) = \sin\bigl(2\pi(x + y) / p\bigr) = \mathrm{cs}(x, y) + \mathrm{sc}(x, y),$$

contributes equally to the cs and sc channels.

## 7. The sum-of-angles score

[mechinterp.py:fft2_neuron_surface](../../../sobolev_distill_character/mechinterp.py) computes a per-neuron score quantifying how cleanly the cc and ss channels balance at the neuron's dominant mode pair.

**Definition.** Let $E_{**}(n)$ for $** \in \{\mathrm{cc}, \mathrm{ss}, \mathrm{cs}, \mathrm{sc}\}$ be neuron $n$'s energy in channel $**$ at its dominant non-DC mode pair $(k_1^*, k_2^*)_n$. The **sum-of-angles score** of neuron $n$ is

$$\mathrm{soa}(n) \;:=\; \frac{2 \sqrt{E_{\mathrm{cc}}(n) \cdot E_{\mathrm{ss}}(n)}}{E_{\mathrm{cc}}(n) + E_{\mathrm{ss}}(n)}.$$

This is the **geometric-to-arithmetic mean ratio** of $E_{\mathrm{cc}}$ and $E_{\mathrm{ss}}$. It is bounded $\mathrm{soa}(n) \in [0, 1]$ with equality $\mathrm{soa}(n) = 1$ if and only if $E_{\mathrm{cc}}(n) = E_{\mathrm{ss}}(n)$.

**Interpretation.**
- $\mathrm{soa}(n) = 1$: the neuron has equal magnitude in both real-form channels at its top mode -- a perfect *trig-identity neuron*.
- $\mathrm{soa}(n) = 0$: one of the channels is zero -- a pure *axis* neuron with $k_1^* = 0$ or $k_2^* = 0$ (so the missing-axis channel vanishes by definition).
- Intermediate values: partial balance.

**Caveat (axis-loss-on configurations).** When the auxiliary axis loss (chapter 03) is active, the trunk is explicitly trained to expose per-axis Fourier features. The dominant mode of most neurons is then $(0, 1)$ or $(1, 0)$ rather than $(1, 1)$, and $\mathrm{soa}$ collapses to 0 for those neurons. This is observed in [fourier_decomp.ipynb](../sobolev/grokking/fourier_decomp.ipynb) and [modulus_sweep.ipynb](../sobolev/grokking/modulus_sweep.ipynb), both of which use `axis_probe = True`. In axis-loss-off configurations like [grokking_baseline_with_decay.ipynb](../sobolev/grokking/grokking_baseline_with_decay.ipynb) the score reads on the order of 0.9 at $p = 17$.

The score is not a flawed probe; it is sensitive to the architecture's preconditioning, exactly as it should be. A future refinement (recommended in the takeaways of the experimental notebooks) is to **filter out axial top modes** ($k_1^* = 0$ or $k_2^* = 0$) before the score computation; that change restores comparability between axis-loss-on and axis-loss-off configurations.

## 8. Excluded loss as Fourier-mode projection

**Definition.** Let $S \subset \widehat{\Lambda}$ be a set of dominant Fourier modes (e.g., the top-2 axis-0 modes from the DFT). For trunk activations $H \in \mathbb{R}^{|\Lambda| \times D}$, the **Fourier-projection** orthogonal to $S$ is

$$\Pi_{S^c}(H) := H - \sum_{n \in S} \widehat{H}(n) \, \xi_n,$$

i.e., the trunk activations with the named modes set to zero in Fourier space. This is the orthogonal projection onto the complement of the modal subspace spanned by $\{\xi_n : n \in S\}$.

[mechinterp.py:excluded_loss_at_freqs](../../../sobolev_distill_character/mechinterp.py) implements this projection on the trunk activations, then re-evaluates the value MSE through the rest of the network. The reported quantity is

$$\Delta(t) := \mathrm{MSE}(\Pi_{S^c}(H_t)) - \mathrm{MSE}(H_t).$$

**Mechanistic content.** $\Delta(t)$ is the loss the model would incur if the named Fourier modes were unavailable. Under a Nanda-style trig-identity solution, all the predictive content lives in the named modes; ablating them sends the loss up dramatically. The temporal trajectory $\Delta(t)$ therefore reads off how much of the model's predictive content is *concentrated in the named Fourier modes* at training step $t$.

The empirical observation (in [dynamics_excluded_loss.ipynb](../sobolev/grokking/dynamics_excluded_loss.ipynb)) is that $\Delta$ rises monotonically from $\sim 0.014$ at the start of training to $\sim 1.006$ at the end -- the Fourier modes go from carrying a small fraction of the predictive content to carrying essentially all of it.

This is the $H^2$-distillation analogue of the Nanda excluded-loss diagnostic in the OpenAI / Welch-Labs grokking arc.

## 9. Code anchors

| structure | code |
|---|---|
| 1-D DFT along an axis | [mechinterp.py:dft_trunk_along_axis](../../../sobolev_distill_character/mechinterp.py) |
| 2-D FFT of neuron surfaces | [mechinterp.py:fft2_neuron_surface](../../../sobolev_distill_character/mechinterp.py) |
| Channel energies $E_{\mathrm{cc}}, E_{\mathrm{ss}}, E_{\mathrm{cs}}, E_{\mathrm{sc}}$ | `channel_energy` field of `Surface2DReport` |
| Sum-of-angles score | `sum_of_angles_score` field; computed in lines surrounding 314-326 of `mechinterp.py` |
| Excluded loss | [mechinterp.py:excluded_loss_at_freqs](../../../sobolev_distill_character/mechinterp.py) |
| Per-axis Fourier targets in the dataset | `axis_target_cos_x, axis_target_sin_x, ...` in [dataset.py](../../../sobolev_distill_character/dataset.py) |

## 10. Further reading

- **G. B. Folland**, *A Course in Abstract Harmonic Analysis*, 2nd ed., CRC Press, 2016. Chapter 4 is the canonical reference for Pontryagin duality on LCA groups.
- **D. Bump**, *Automorphic Forms and Representations*, Cambridge University Press, 1998. Chapter 3 covers characters of finite abelian groups; Section 3.7 covers the trig-identity decomposition in the form used here.
- **R. M. Gray**, *Toeplitz and Circulant Matrices: A Review*, Foundations and Trends in Communications and Information Theory, vol. 2, no. 3, 2006. Background on the DFT as the canonical isomorphism for circulant matrices, which is the natural finite-group counterpart of Pontryagin duality.
- **N. Nanda et al.**, *Progress measures for grokking via mechanistic interpretability*, ICLR 2023. The excluded-loss diagnostic was introduced here in the transformer context; this chapter adapts it to the Sobolev-distillation pipeline.

---

Cross-reference: chapter 02 establishes that the teacher lives in the band-limited subspace $\mathcal{T}_p \otimes \mathcal{T}_p$, which in dual language is supported on a finite subset of $\widehat{T^2} = \mathbb{Z}^2$. Chapter 03 explains why $H^2$ control is the right notion of Sobolev approximation; the trig-identity decomposition here gives the per-mode content of that approximation. Chapter 05 lifts the per-neuron picture to the full SVD / PCA of trunk activations.
