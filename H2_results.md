# H2 — Cheap Global-Statistics Addition Results (§2.3)

**Question:** Does injecting cheap global topological statistics (diameter, girth, nodes/edges ratio, number of components) alongside standard local encodings yield modest performance gains and improve sample efficiency without significant computational overhead?

**Definition:**
`Δbase(D) = Accuracy_D(fθ(Pbase ∪ PH2)) − Accuracy_D(fθ(Pbase))`
`ΔnoPE(D) = Accuracy_D(fθ(Pbase ∪ PH2)) − Accuracy_D(fθ(∅))`

* Positive Δbase means the addition of global statistics improved the model beyond the standard GraphGPS baseline.
* Positive ΔnoPE means the combined encodings outperformed the zero-embedding fallback.

---

## Results

| Dataset | Baseline (Pbase) | No-PE (P=∅) | H2 (Pbase ∪ PH2) | **Δbase** | **ΔnoPE** | Verdict |
|---------|------------------|-------------|------------------|-----------|-----------|---------|
| MUTAG   | 0.8579 ± 0.0411 | 0.8947 ± 0.0235 | 0.8790 ± 0.0579 | **+0.0211** | -0.0158 | ✓ Modest gain vs base |
| ENZYMES | 0.5533 ± 0.0420 | 0.5217 ± 0.0454 | 0.5200 ± 0.0785 | **-0.0333** | -0.0017 | ⚠ Noisy prior regression |
| NCI1    | 0.8071 ± 0.0143 | 0.7409 ± 0.0150 | 0.8073 ± 0.0148 | **+0.0002** | +0.0664 | = Neutral (Ties baseline) |

Summary Statistics:
* **Mean \|Δbase\|:** 0.0182
* **Mean \|ΔnoPE\|:** 0.0280

---

## Interpretation

**MUTAG demonstrates a modest gain:** On this small, structurally simple dataset, supplementing the baseline with global statistics provided a +2.1 percentage point improvement over the baseline (`Pbase`). This directly supports the H2 hypothesis. However, the `No-PE` ablation scored the highest overall (0.8947). Because MUTAG is extremely small (188 graphs), combining LapPE and the GraphStats parameter space likely introduces too much capacity, causing slight overfitting compared to the zero-embedding fallback.

**NCI1 acts as a "Do No Harm" control:** The H2 model (0.8073) effectively tied the Baseline (0.8071). While the global stats successfully rescued the model from the severe `No-PE` performance drop (+0.0664), they did not provide any *new* actionable structural information beyond what the standard Laplacian PE already mapped. The stats integrated safely, but the marginal gain was negligible.

**ENZYMES reveals a noisy prior effect:** For the complex 6-class ENZYMES problem, global scalars like diameter and girth actively hurt performance, dropping accuracy by -3.3 percentage points compared to the baseline. Forcing the network's readout layer to accommodate these macro-level stats acted as a distraction from the more critical local message passing, suggesting that raw topological scalars can be redundant or detrimental when robust localized structural encodings are already present.

**Overall conclusion:** Hypothesis 2 holds conditionally. Injecting cheap global statistics yields modest gains on specific structural tasks (MUTAG). However, on complex biochemical networks (NCI1, ENZYMES), these stats fail to significantly outperform standard Laplacian encodings.

---

## Experiment Details

| Setting | Value |
|---------|-------|
| Model | GPS (GINE + Transformer) |
| Positional encoding (Pbase) | LapPE (Standard configuration) |
| Structural additions (PH2) | Global stats (Diameter, Girth, Components, N/E Ratio) |
| Injection point | Readout layer (with MLP Projection and Normalization) |
| Splits | 10-fold stratified CV |
| Hardware | dGPU via `prime-run` / NVML |

### Architectural Stabilization Note
To ensure stable gradient flow, the raw topological scalars (which vary wildly in magnitude, such as infinite diameter for disconnected graphs) were routed through a 2-layer MLP projection before being concatenated with the standard graph embeddings. This projection maps the dense topological statistics into a compatible latent space, allowing the network to integrate the hints safely without gradient shock.

---

## Recommendations for Future Work

### 1. Evaluate the "Low-Data Regime"
Hypothesis 2 specifically wagers that these cheap statistics improve sample efficiency when training data is scarce. To fully prove this claim, the `Pbase ∪ PH2` configurations must be evaluated on restricted dataset splits (e.g., using `dataset.split [0.1,0.1,0.8]` for 10% data or `[0.25,0.1,0.65]` for 25% data) and their convergence speeds compared against the standard baseline.

### 2. Quantify Computational Costs
While performance varied by dataset, H2's secondary claim of computational efficiency remains untested in this summary. Future aggregations should extract the $T_{prep}$ (preprocessing time) and per-epoch runtime overhead (`epoch_overhead_s`) from the `stats.json` logs to empirically prove these statistics add negligible computational cost compared to heavy structural encodings (e.g., Ricci Curvature).

### 3. Scale to Larger Benchmarks
Testing H2 on larger, more diverse datasets like ZINC or CIFAR10 will help determine if the "noisy prior" effect seen in ENZYMES is a rule for complex graphs, or an exception.

---

*Run date: 2026-05-08 | Context: CS 768 H2 Evaluation | Results generated via `scripts/collect_h2_results.py`*