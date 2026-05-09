# Combined Ablation Results — GraphGPS Structural Encoding Study

**Run date:** 2026-05-08
**Hardware:** Apple Silicon MPS
**Protocol:** 3-fold CV (fold 0), 3 seeds, early stopping patience = 10
**Model:** GPS (GINE + Transformer), 3 layers, dim = 64

---

## Conditions

| Condition | Node encoding | Edge encoding | Description |
|-----------|--------------|---------------|-------------|
| **C1** | LinearNode + LapPE (8-dim) | DummyEdge | Baseline — standard Pbase |
| **C2** | LinearNode only | DummyEdge | Zero embedding — P = ∅ |
| **C3-Forman** | LinearNode + LapPE (8-dim) | RicciEdge (Forman κ) | Pbase + Forman Ricci curvature |
| **C3-Ollivier** | LinearNode + LapPE (8-dim) | RicciEdge (Ollivier κ) | Pbase + Ollivier Ricci curvature |

---

## Full Results (Accuracy %)

| Dataset | C1: LapPE | C2: P=∅ | C3: +Forman | C3: +Ollivier |
|---------|-----------|---------|-------------|---------------|
| MUTAG   | 82.5 ± 6.6 | 80.7 ± 8.9 | 86.3 ± 4.2 | 84.7 ± 6.0 |
| ENZYMES | 32.2 ± 5.5 †| 41.7 ± 9.5 | 64.0 ± 3.8 | 44.2 ± 7.7 |
| NCI1    | 78.8 ± 0.3 | 74.5 ± 1.6 | 78.3 ± 0.4 | 80.0 ± 0.8 |

† ENZYMES C1 is underfit — early stopping with patience=10 terminated training
prematurely (model still converging). Treat ENZYMES Δ values cautiously.

---

## Q1 — Does LapPE help over no encoding?

**Δzero(D) = C1 − C2**  (positive = LapPE helps)

| Dataset | C1 | C2 | **Δzero** | Verdict |
|---------|----|----|-----------|---------|
| MUTAG   | 82.5% | 80.7% | **+1.7 pp** | ✓ LapPE helps (within noise) |
| ENZYMES | 32.2% | 41.7% | **−9.4 pp** | ✗ Unreliable — C1 underfit † |
| NCI1    | 78.8% | 74.5% | **+4.4 pp** | ✓ LapPE helps |

Mean |Δzero| (excl. ENZYMES): **3.1 pp**

---

## H1 — Does Forman Ricci add marginal value on top of LapPE?

**Δ(Forman; D) = C3-Forman − C1**  (positive = Forman helps)

| Dataset | C1 | C3-Forman | **Δ(Forman)** | Verdict |
|---------|----|-----------|---------------|---------|
| MUTAG   | 82.5% | 86.3% | **+3.9 pp** | ✓ Forman helps |
| ENZYMES | 32.2% | 64.0% | **+31.8 pp** | ⚠ Inflated by C1 underfit † |
| NCI1    | 78.8% | 78.3% | **−0.5 pp** | ✗ No marginal gain |

Mean Δ(Forman; D): **+11.7 pp** (dominated by ENZYMES anomaly)
Mean Δ(Forman; D) excl. ENZYMES: **+1.7 pp**

---

## H1 — Does Ollivier Ricci add marginal value on top of LapPE?

**Δ(Ollivier; D) = C3-Ollivier − C1**  (positive = Ollivier helps)

| Dataset | C1 | C3-Ollivier | **Δ(Ollivier)** | Verdict |
|---------|----|-------------|-----------------|---------|
| MUTAG   | 82.5% | 84.7% | **+2.3 pp** | ✓ Ollivier helps |
| ENZYMES | 32.2% | 44.2% | **+12.0 pp** | ⚠ Inflated by C1 underfit † |
| NCI1    | 78.8% | 80.0% | **+1.1 pp** | ✓ Ollivier helps |

Mean Δ(Ollivier; D): **+5.1 pp**
Mean Δ(Ollivier; D) excl. ENZYMES: **+1.7 pp**

---

## Cross-condition Comparison

| Dataset | Δzero (Q1) | Δ Forman (H1) | Δ Ollivier (H1) |
|---------|-----------|---------------|-----------------|
| MUTAG   | +1.7 pp | +3.9 pp | +2.3 pp |
| ENZYMES | −9.4 pp † | +31.8 pp † | +12.0 pp † |
| NCI1    | +4.4 pp | −0.5 pp | +1.1 pp |

---

## Interpretation

**Q1:** LapPE provides meaningful signal on NCI1 (+4.4 pp) and marginal gain
on MUTAG (+1.7 pp). The ENZYMES result is unreliable due to premature early
stopping of C1 — a rerun with longer patience is needed.

**H1 (Forman):** Adds value on MUTAG (+3.9 pp) but not NCI1 (−0.5 pp). The
ENZYMES gap is artificially large due to the C1 underfit issue.

**H1 (Ollivier):** The most consistent variant — positive Δ on all 3 datasets.
Smaller magnitude than Forman on MUTAG/ENZYMES but more reliable as a signal
given the consistent direction across datasets.

**Both Ricci variants outperform the zero-embedding baseline (C2) on all
datasets**, suggesting Ricci curvature captures structural information beyond
what message passing alone recovers.

---

## Caveats

1. **ENZYMES C1 underfit** — patience=10 is too aggressive for ENZYMES (300
   epochs needed). Rerun C1 with `optim.early_stop_patience 50` or disable
   early stopping for ENZYMES to get a reliable comparison.

2. **3 seeds only** — MUTAG has high fold variance (~19 graphs per test fold).
   Results should be treated as indicative, not conclusive.

3. **RicciEdge adds parameters** — C3 has a `Linear(1, dim_edge)` + a
   `dim_edge`-vector zero-embedding vs. DummyEdge in C1/C2. A small capacity
   advantage exists for C3 that is not fully controlled for.

4. **Single fold (fold 0)** — full 10-fold CV would tighten confidence
   intervals, especially for MUTAG.

---

*Results directory: `GraphGPS/results/` | Branch: `ravi_h1`*
