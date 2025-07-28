AGENTS.md  – Meta-Repository Guide

for ChatGPT / Codex agents contributing code

⸻

1  Repository Overview

unlearning-research-meta/
├── code/                 # canonical, installable source
│   └── memory_pair/      # ← import here for all algorithms
├── data/                 
│   └── data_loader/      # unified, fail-safe dataset loaders
└── experiments/          # ⚑ each sub-folder is an independent study
    ├── sublinear_regret/
    ├── deletion_capacity/
    └── post_deletion_accuracy/

	•	Top-level AGENTS.md (this file) describes global conventions.
	•	Every directory inside experiments/ has its own AGENTS.md
that specifies the exact scaffold, CLI, and commit protocol
for that experiment.

⸻

2  General Rules for Agents
	1.	Stay in scope
When adding or editing code, touch only the paths explicitly
mentioned in the sub-module’s AGENTS.md.
Global files (code/, data/, root config) should be modified
only if the change is required by multiple experiments
and the top-level maintainer approves.
	2.	Import policy
Algorithms

from code.memory_pair.src.memory_pair import MemoryPair

Data

from data_loader import get_rotating_mnist_stream

Never duplicate Memory-Pair logic or re-download datasets in
experiment folders.

	3.	Editable installs
After changing code/ or data/, run

pip install -e code/memory_pair
pip install -e data/data_loader

to ensure namespace packages resolve.

	4.	Commit provenance
Each experiment script must write results into
experiments/<name>/results/ and auto-commit them with a message
pattern described in that sub-module’s AGENTS.md
(e.g. "EXP:sublinear_regret cifar-drift memorypair 9f2e6a").
	5.	Reproducibility
All public scripts must accept --seed and call the
data_loader.utils.set_global_seed helper.
Deterministic outputs are mandatory for CI checks.

⸻

2.1  RDP Accountant & Joint Optimization

The MemoryPair algorithm supports advanced privacy accounting via
Rényi Differential Privacy (RDP) with joint m-σ optimization and
adaptive recalibration.

	•	Privacy Accountant Types:
	  --accountant rdp    # Recommended for research
	  --accountant legacy # Backward compatibility

	•	Joint m-σ Optimization:
The RDPOdometer finds the largest deletion capacity m and
smallest noise scale σ satisfying both:
1. Privacy: m deletions with sensitivity bound ≤ RDP→(ε,δ) budget
2. Regret: R_total(m,σ)/T ≤ γ_priv

Binary search on m ∈ [1, m_max], then solve:
σ ≥ max_α sqrt(m·α·sens²/(2·εα_budget[α]))

	•	Per-Delete Sensitivity:
Each delete uses actual influence norm ||d|| rather than
global L/λ bound. Empirical 95th percentile becomes the
sensitivity bound for subsequent optimizations.

	•	Adaptive Recalibration:
When gradient EMA drifts >threshold from calibrated value:
1. Pause deletes, collect fresh statistics
2. Re-run joint optimization with remaining RDP budget  
3. Resume with updated m and σ

	•	CLI Usage:
python run.py --accountant rdp --alphas "2,4,8,16" \
  --ema-beta 0.9 --recal-window 5000 --recal-threshold 0.3 \
  --m-max 1000

	•	Expected Outputs:
Logs include: eps_converted, m_current, sigma_current,
sens_count, sens_q95, recalibrations_count

⸻

3  How to Contribute
	1.	Read the sub-module brief
Navigate to experiments/<your-feature>/AGENTS.md and follow the
step-by-step instructions.
If a directory is missing that file, open an issue before coding.
	2.	Run sanity checks
	•	python data/data_loader/sanity_check.py
	•	Unit tests inside code/memory_pair/tests (if any)
	•	Experiment-specific smoke test (usually python run.py --help)
	3.	Open a pull request
Target branch: main.
PR title format:

EXP:<submodule> – <short description>

Include CLI log snippets and pointers to the auto-committed results
commit hash.

⸻

4  Contact / Maintainers
	•	Primary maintainer: @kennonstewart
	•	Issues & discussions: GitHub Issues tab.
	•	For urgent build failures, tag maintainers in your PR.

⸻

TL;DR
Look inside each experiments/<name>/AGENTS.md for local
instructions; defer to this file for repo-wide conventions. Follow
the import paths, keep results under version control, and ensure every
script is reproducible with a single command.
