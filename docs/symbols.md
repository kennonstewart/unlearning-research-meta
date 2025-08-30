# Symbols and notation

This repository uses the following symbols consistently across code and docs:

- G: Upper bound on per-step gradient norm (Lipschitz constant of losses), i.e., ||g_t||_2 ≤ G.
- λ (lambda): Strong convexity parameter of the loss (when assumed).
- c, C: Uniform eigenvalue bounds on the inverse-curvature preconditioner B_t used by (online) L-BFGS, i.e., c I ≼ B_t ≼ C I.
- D: Diameter of the feasible domain W, i.e., sup_{u,v∈W} ||u − v||_2 ≤ D.
- S_T: AdaGrad statistic, cumulative squared gradients S_T = ∑_{t=1}^T ||g_t||_2^2.
- P_T: Path length of the comparator sequence, P_T = ∑_{t=2}^T ||w_t^* − w_{t-1}^*||_2.
- N*: Sample-complexity gate; the stream index after which predictions are enabled.
- σ_step: Gaussian noise scale applied per valid delete step.
- ρ: zCDP (zero-Concentrated DP) budget; ρ_tot is the total budget, ρ_s per delete.
- (ε, δ): Privacy/unlearning parameters used for reporting; internally we compose via ρ (zCDP) and convert for reporting.

Theory overlays referenced in plots:
- Strongly-convex schedule (η_t = 1/(λ t)):
  - Static regret bound: R_T ≤ (G^2/(λ c))(1 + ln T) [+ optional path term G P_T].
- AdaGrad schedule (η_t = D/√S_t):
  - Adaptive bound: R_T ≤ G D √(c C S_T).