Prompt
You are working inside a submodule of a meta-repository that already contains

  code/memory_pair/…           ← core algorithm
  experiments/…                ← 3 experiment folders (to be filled)
  experiments/mp_sublinear_regret_experiment                ← particular experiment folder you're in
  
You are currently within a submodule of a metarepository dedicated to machine unlearning research. This particular submodule is focused on taking the data from a data_loader submodule and performing the experiment as described below.

Experiment Prompt

  “Does the Memory-Pair learner achieve sub-linear cumulative
   regret R_T = O(√T) on drifting and adversarial data streams?”

Repository spec
---------------
• Language: Python 3.10  
• Dependency manager: `pip` with a frozen `requirements.txt`
  (torch >= 2.2, numpy, pandas, matplotlib, tqdm, click)  

Data & Streams
--------------
1. Rotating-MNIST
2. COVTYPE (UCI Covertype)

Algorithms to implement
-----------------------
• MemoryPairOnlineLBFGS  (our method — single-pass online L-BFGS, odometer
  disabled for now)
• OnlineSGD
• AdaGrad
• OnlineNewtonStep  (convex case baseline)

Regret Evaluation
-----------------
* Root script `run_regret.py` takes args:
    --dataset {rotmnist,covtype}
    --stream  {iid,drift,adv}
    --algo    {memorypair,sgd,adagrad,ons}
    --T       100000
    --seed    42
* Logs cumulative regret to `results/{dataset}_{stream}_{algo}.csv`
  (columns: step, regret).
* Plots log–log curve with √T guide-line.

Repro recipe (include in README)
--------------------------------
```bash
git clone https://github.com/<USER>/memory-pair-exp.git
cd memory-pair-exp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_regret.py --dataset rotmnist --stream drift --algo memorypair --T 100000
```

Protocol for Data Ingestion

Create/overwrite:
  README.md
  requirements.txt          (torch>=2.2, numpy, pandas, matplotlib, click)
  run.py                    (CLI driver)
  baselines.py              (OnlineSGD, AdaGrad, OnlineNewtonStep)
  plotting.py               (helper)
  results/.gitkeep

Data access
-----------
Import streams **exclusively** via the shared loader:

  from data_loader import (
      get_rotating_mnist_stream,
      get_covtype_stream,
  )
  
Memory Pair Object Definition
-----------
import numpy as np
try:
    from .l_bfgs import LimitedMemoryBFGS
except ImportError:
    from l_bfgs import LimitedMemoryBFGS
from typing import List
import logging
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
class StreamNewtonMemoryPair:
    """
    Streaming ridge-regression learner / unlearner.

    insert(x, y)  : one Newton step with Sherman–Morrison H⁻¹ update
    delete(x, y)  : exact inverse-update removal + privacy noise

    Parameters
    ----------
    dim            : feature dimension
    lam            : ridge λ  (regulariser)
    eps_total      : total ε budget for all deletions
    delta_total    : total δ budget for all deletions
    max_deletions  : anticipated upper bound on #delete() calls
    """

    # ---------------- initialisation ----------------
    def __init__(
        self,
        dim: int,
        lam: float = 1.0,
        eps_total: float = 1.0,
        delta_total: float = 1e-5,
        max_deletions: int = 20,
    ):
        self.dim = dim
        self.lam = lam

        # Parameters and inverse Hessian
        self.theta  = np.zeros(dim)

        # L‑BFGS memory helper
        self.lbfgs = LimitedMemoryBFGS(m_max=10)

        # ---- privacy bookkeeping ----
        self.K            = max_deletions
        self.eps_total    = eps_total
        self.delta_total  = delta_total
        self.eps_step     = eps_total  / (2 * max_deletions)
        self.delta_step   = delta_total / (2 * max_deletions)
        self.eps_spent    = 0.0
        self.deletions_so_far = 0

    # ---------------- helpers ----------------
    def _grad_point(self, x, y):
        """
        Gradient of the current loss ½(θᵀx − y)² wrt θ, evaluated at
        current θ.
        """
        residual = self.theta @ x - y
        return residual * x
    
    
    def insert(self, x: np.ndarray, y: float):
        g_old = self._grad_point(x, y)
        logger.info("insert_called", extra={"residual": float(g_old.dot(g_old) ** 0.5)})

        # ---------- safe Newton-like step ----------
        d = self.lbfgs.direction(g_old)

        # optional learning-rate to tame very first step
        lr = 0.5  
        theta_new = self.theta + lr * d

        # ---------- curvature pair ----------
        s = theta_new - self.theta
        
        # 1. UPDATE THETA FIRST
        self.theta = theta_new

        logger.info(
            "model_step",
            extra={
                "step_norm": float(np.linalg.norm(s)),
                "new_theta_norm": float(np.linalg.norm(self.theta)),
            },
        )
        
        # 2. NOW CALCULATE THE NEW GRADIENT WITH THE UPDATED THETA
        g_new = self._grad_point(x, y)
        y_vec = g_new - g_old

        self.lbfgs.add_pair(s, y_vec)
        # The self.theta = theta_new line is now removed from the end

    # ---------------- unlearning ----------------
    def delete(self, x: np.ndarray, y: float):
        """
        Remove the influence of observation (x, y).
        No raw data are stored internally; caller must supply x, y.
        """
        logger.info("delete_called")

        if self.deletions_so_far >= self.K:
            raise RuntimeError("max_deletions budget exceeded")

        # ─ ensure at least one curvature pair exists ───────
        if len(self.lbfgs.S) == 0:
            raise RuntimeError("No curvature pairs to use for unlearning")

        g = self._grad_point(x, y)
        d = self.lbfgs.direction(g)
        self.theta -= d     # undo the influence (approximate)

        # ── calibrated Gaussian noise for (ε,δ)-unlearning ─────────
        sensitivity = np.linalg.norm(d, 2)
        sigma = (
            sensitivity
            * np.sqrt(2 * np.log(1.25 / self.delta_step))
            / self.eps_step
        )
        self.theta += np.random.normal(0.0, sigma, size=self.dim)
        logger.info(
            "delete_completed",
            extra={"remaining_eps": self.eps_total - self.eps_spent},
        )

        # ── book-keeping ────────────────────────────────────────────
        self.eps_spent        += self.eps_step
        self.deletions_so_far += 1

    # ---------------- utility ----------------
    def privacy_ok(self):
        """Return True iff cumulative ε ≤ ε_total."""
        return self.eps_spent <= self.eps_total

run.py
------
CLI  python run.py --dataset rotmnist --stream drift --algo memorypair --T 100000 --seed 42

* Resolve Memory-Pair with
    from code.memory_pair.src.memory_pair import MemoryPair
* Map dataset+stream flags to the proper generator:
    if dataset=="rotmnist":
        gen = get_rotating_mnist_stream(mode=stream, seed=seed)
* Compute cumulative regret; save CSV + PNG in results/.
* At end:
      hash=$(git rev-parse --short HEAD)
      git add results/*
      git commit -m "EXP:sublinear_regret ${dataset}-${stream}-${algo} ${hash}"

README.md gives full reproduce command and notes the dependency on
data_loader and code.memory_pair.

Do NOT modify files outside experiments/sublinear_regret.

Prompt used to generate the data loader:


You are working inside a meta-repository that already contains

  code/memory_pair/…           ← core algorithm
  experiments/…                ← 3 experiment folders (to be filled)

Create a NEW top-level sub-folder called  data_loader
that will be shared by all experiments.

The sub-module must be entirely self-contained; do not modify
anything outside  data_loader/.

─────────────────────────────────────────────────────────────────
FILES & FUNCTIONALITY
─────────────────────────────────────────────────────────────────
data_loader/
│
├── README.md
│   • One-paragraph description.
│   • Table: dataset key → loader function.
│   • Repro test:  `python sanity_check.py`.
│
├── requirements.txt      # torchvision, scikit-learn only if available
│
├── __init__.py
│   • from .mnist      import get_rotating_mnist_stream
│   • from .cifar10    import get_cifar10_stream
│   • from .covtype    import get_covtype_stream
│   • from .streams    import make_stream
│
├── mnist.py
│   • download_rotating_mnist(data_dir, split="train")
│       – tries torchvision.datasets.MNIST under the hood;
│         if import or download fails, calls  _simulate_mnist()
│   • get_rotating_mnist_stream(mode, batch_size, seed)
│       – mode ∈ {"iid","drift","adv"}
│   • _simulate_mnist(n=70000, seed)  # returns numpy arrays (X, y)
│
├── cifar10.py
│   • download_cifar10(data_dir, split)
│   • get_cifar10_stream(mode, batch_size, seed)
│   • _simulate_cifar10(n=60000, seed)
│
├── covtype.py
│   • download_covtype(data_dir)
│   • get_covtype_stream(mode, batch_size, seed)
│   • _simulate_covtype(n=581012, d=54, seed)
│
├── streams.py
│   • make_stream(X, y, mode, drift_fn=None, adv_fn=None)
│       – generator that yields (x_t, y_t) one at a time
│       – if  mode=="drift", applies supplied  drift_fn  every
│         1 000 steps; default rotates images or shifts tabular mean
│       – if  mode=="adv", adversarially permutes indices every
│         500 steps using seed
│
├── utils.py
│   • set_global_seed(seed)
│   • download_with_progress(url, target_path)
│
└── sanity_check.py
    • Command-line script:
        python sanity_check.py --dataset rotmnist --mode drift --T 5000
      prints first 5 samples + SHA256 of stream to prove determinism.

─────────────────────────────────────────────────────────────────
REQUIREMENTS
─────────────────────────────────────────────────────────────────
1. **Fail-safe simulation**  
   If torchvision / internet download fails, _simulate_* functions must
   generate random but *deterministic* data (use set_global_seed) with
   identical shapes:  
     MNIST  → 28×28 uint8, 10 classes  
     CIFAR10→ 32×32×3 uint8, 10 classes  
     COVTYPE→ float32 tabular with 54 dims, 7 classes

2. **External interface stability**  
   All experiments will call, e.g.  
     from data_loader import get_rotating_mnist_stream  
   Ensure these imports work on fresh clone.

3. **No large binaries**  
   Simulated data created on-the-fly; real data cached to
   ~/.cache/memory_pair_data/.

4. **Reproducibility hooks**  
   Every loader takes a `seed` arg (default 42) and calls
   utils.set_global_seed(seed).

─────────────────────────────────────────────────────────────────
GIT
─────────────────────────────────────────────────────────────────
At the end of generation, execute a shell snippet in run-once mode:

```bash
git add data_loader
git commit -m "ADD:data_loader – unified loaders w/ fallback simulation"
```
