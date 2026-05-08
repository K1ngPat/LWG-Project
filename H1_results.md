# H1 — Marginal Value of Ricci Curvature (§2.2 / §2.3)

**Question (Q2):** For each left-behind structural encoding `p`, what is the
marginal gain on top of the standard `Pbase` (LapPE)?

**Definition:**
```
Δ(p; D) = Perf_D(fθ(Pbase ∪ {p})) − Perf_D(fθ(Pbase))
```
Positive `Δ` means the extra encoding helped; negative means it hurt.

**Hypothesis H1:** Ricci curvature yields **positive** `Δ(p; D)` on datasets
with **pronounced bottlenecks / bridges** or where message propagation is
hampered.

This file is the H1 analog of `Q1_results.md`. Q1 contrasted `Pbase` against
the empty encoding (`P = ∅`); here we contrast `Pbase` against `Pbase ∪
{Ricci}`.

---

## Experimental setup

| Setting | Value |
|---------|-------|
| Datasets | MUTAG, ENZYMES, NCI1 (the three Q1 datasets that completed) |
| Backbone | GPS (GINE + Transformer), identical to Q1 C1 |
| C1 baseline | LapPE only (8-dim, DeepSet encoder, `max_freqs=8`) |
| C3 condition | LapPE **+** per-edge Ricci curvature via `RicciEdge` encoder |
| Ricci variants | Forman (combinatorial, fast); Ollivier (transport, slower) |
| Edge encoder swap | `DummyEdge` → `RicciEdge` (linear(1→`dim_edge`) + learnable `0_emb` for `NaN`) |
| Splits / seeds | 10-fold stratified CV (fold 0), 10 seeds — same as Q1 small datasets |
| Sign convention | Higher-is-better metric: `Δ = C3 − C1`. MAE: `Δ = C1 − C3` |

The **only** differences between C1 and C3 YAMLs are
`dataset.edge_encoder_name: RicciEdge` and a new `ricci:` block (variant /
alpha / normalize). Splits, seeds, model dims, optimizer, and epoch budgets
are byte-identical so `Δ(p; D)` is a clean marginal.

---

## Forman Ricci results

Run with:

```bash
cd LWG-Project/GraphGPS
bash scripts/run_h1_ablation.sh
python scripts/collect_h1_results.py --variant forman
```

> **Note on C1 baseline:** C1 results reuse the Q1 ablation run (same config,
> 10 seeds, fold 0, identical hyperparameters). The C1 result directories were
> not re-generated as part of the H1 run (`RUN_BASELINE=0`).

| Dataset | C1: LapPE only | C3: LapPE + Forman | **Δ(Forman; D)** | Verdict |
|---------|----------------|---------------------|------------------|---------|
| MUTAG   | 87.7 ± 2.5 %  | 86.3 ± 4.2 %        | **−1.4 pp**      | ✗ Ricci did not help (within noise) |
| ENZYMES | 68.9 ± 3.4 %  | 64.0 ± 3.8 %        | **−4.9 pp**      | ✗ Ricci hurt |
| NCI1    | 79.4 ± 1.1 %† | 78.3 ± 0.4 %        | **−1.1 pp**      | ✗ Ricci did not help |

† NCI1 C1 ran only 50 epochs (converging at epoch 36); treat as conservative lower bound.
NCI1 C3: 3 seeds, early stopping patience=10.

Mean Δ(Forman; D) across all three datasets: **−2.5 pp**

*Run date: 2026-05-08 | Seeds: 10 (MUTAG, ENZYMES), 3 (NCI1) | Fold: 0*

---

## Ollivier Ricci results (deferred)

Identical setup with `cfg.ricci.variant: ollivier`. Run after Forman with:

```bash
RUN_OLLIVIER=1 bash scripts/run_h1_ablation.sh
python scripts/collect_h1_results.py --variant ollivier
```

| Dataset | C1: LapPE only | C3: LapPE + Ollivier | **Δ(Ollivier; D)** | T_prep (s/graph) | Verdict |
|---------|----------------|-----------------------|--------------------|------------------|---------|
| MUTAG   | _pending_      | _pending_             | _pending_          | _pending_        | _pending_ |
| ENZYMES | _pending_      | _pending_             | _pending_          | _pending_        | _pending_ |
| NCI1    | _pending_      | _pending_             | _pending_          | _pending_        | _pending_ |

---

## Cost report (PDF §2.3 requirement)

For each `(D, p)`, record:

* `T_prep(p; D)`: total Ricci precompute time printed by
  `master_loader.py` (look for the line `Done! Ricci precompute took ...`
  in the run log).
* Per-epoch overhead vs C1 baseline (Ricci is precomputed once, so the
  per-epoch cost should match C1 within noise).
* Extra storage: `data.edge_curvature` of shape `[num_edges, 1]` per graph.

| Dataset | Variant  | T_prep (total) | Per-epoch (C3 / C1) |
|---------|----------|----------------|---------------------|
| MUTAG   | Forman   | _pending_      | _pending_           |
| MUTAG   | Ollivier | _pending_      | _pending_           |
| ENZYMES | Forman   | _pending_      | _pending_           |
| ENZYMES | Ollivier | _pending_      | _pending_           |
| NCI1    | Forman   | _pending_      | _pending_           |
| NCI1    | Ollivier | _pending_      | _pending_           |

---

## Interpretation guide

H1 is **supported** if `Δ(Ricci; D) > 0` is concentrated on datasets with
high bridge fraction / negative-curvature edge mass. To test it:

1. Run the Forman pass (above).
2. Run `python scripts/check_dataset_stats.py` to populate the bridge column.
3. Plot `Δ(p; D)` vs. bridge fraction (or mean Ollivier curvature when
   that pass completes). H1 predicts a positive correlation.
4. Repeat with Ollivier when compute permits — the proposal calls for both.

### Caveats to flag

* `DummyEdge → RicciEdge` increases edge-encoder parameter count from a
  single embedding to a Linear(1, `dim_edge`) plus a `dim_edge`-vector
  zero-embedding. The PDF says "same hyperparameters where possible";
  this is a documented capacity asymmetry and should be noted alongside
  any positive `Δ`.
* The Q1 fold-0 / 10-seed protocol for MUTAG (188 graphs, ~19 per fold)
  has high test-fold variance. Treat single-fold `Δ` cautiously and prefer
  the ENZYMES / NCI1 numbers as the primary H1 signal (same caveat that
  shaped the Q1 conclusions).
* `NaN` curvatures (mathematically undefined edges) are routed through
  `RicciEdgeEncoder.zero_emb`, matching the proposal's `0_emb` fallback
  in §2.1.

---

*Run date: TBD | Runner: `scripts/run_h1_ablation.sh` | Results in:
`GraphGPS/results/`*
