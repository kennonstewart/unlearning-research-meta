# Sub‑linear Regret Experiment

This experiment empirically validates the **Memory Pair** online learner’s theoretical guarantee that its cumulative regret grows *sub‑linearly*—specifically \( R_T = \tilde O(\sqrt{T}) \)—with the number \(T\) of processed examples, even when the data stream is drifting or adversarial.

## Why sub‑linear regret matters

For an online learner, **cumulative regret**
R_T \;=\;\sum_{t=1}^{T} \bigl(\ell_t(\theta_t) - \ell_t(\theta^\star)\bigr)
\]
measures how much worse we perform than the best fixed model chosen in hindsight.  
The Memory Pair algorithm combines an *insert* rule (online L‑BFGS step) with a symmetric *delete* rule (inverse L‑BFGS + calibrated Gaussian noise).  
Under Assumptions 5.9 and Theorem 5.8 of the accompanying paper, the learner satisfies

\[
R_T \;=\; \mathcal O\!\bigl(\sqrt{T}\bigr)\quad\Longrightarrow\quad
\frac{R_T}{T}\xrightarrow[T\to\infty]{}0,
\]

so its average excess loss vanishes.  
This script tests whether the empirical slope of \(\log R_T\) versus \(\log T\) is below 1, confirming sub‑linearity.

## Streams and algorithms

| Flag | Meaning | Function |
|------|---------|----------|
| `--dataset rotmnist` | Rotating‑MNIST stream | `get_rotating_mnist_stream` |
| `--dataset covtype`  | Covertype forest‑cover stream | `get_covtype_stream` |
| `--stream iid`   | i.i.d. shuffle of the data |   |
| `--stream drift` | Slowly drifting label/feature distribution |   |
| `--stream adv`   | Fully adversarial permutation |   |
| `--algo memorypair` | Memory Pair learner (default) |   |
| `--algo sgd`, `adagrad`, `ons` | Baseline online learners |   |

The experiment logs cumulative regret every 100 steps, writes the values to
`results/<dataset>_<stream>_<algo>.csv`, and produces a matching `.png` log‑log plot.

## Quick start

```bash
# Install dependencies once
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run 1e5 steps of Memory Pair on rotating‑MNIST with drift
PYTHONPATH=.:data python run.py \
    --dataset rotmnist \
    --stream drift \
    --algo memorypair \
    --T 100000 \
    --seed 42
```

After completion the script auto‑commits the CSV and PNG with a message like:

```
EXP:sublinear_regret rotmnist-drift-memorypair <git‑hash>
```

## Interpreting the plot

A slope **< 1** on the log–log plot of \(R_T\) versus \(T\) confirms sub‑linear growth.  
For the Memory Pair, theory predicts an asymptotic slope of **1/2**; baselines such as vanilla SGD may match this on i.i.d. data but typically degrade under drift or adversarial scheduling.

---

For full theoretical details see Section 5 of *“Memory Pairs: Stream‑Native Learning and Unlearning with Sub‑linear Regret”* (PDF in the project root).
