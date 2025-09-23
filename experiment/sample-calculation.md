# 📐 Worked Example of the Memory Pair Regret Bounds

Suppose we are running the **Memory Pair algorithm** in a simple stream setting. We want to see how the regret bound behaves under fixed constants and how it decomposes.

---

## Step 1. Fixing the Constants

Let’s pick values that make the math clean but realistic:

* Gradient bound: $G = 2.0$
  (our loss gradients are at most length 2)
* Strong convexity constant: $\lambda = 0.5$
  (our loss is moderately curved, ensuring stability)
* Horizon: $T = 1000$
  (we will process 1,000 events)
* Path length of comparator: $P_T = 25$
  (the “best possible model” drifts a little bit across the stream — 25 units of total movement)

---

## Step 2. The Regret Bound Formula

From the paper:

$$
R_T^{dyn} \;\leq\; \frac{G^2}{\lambda} \,\big(1 + \ln T\big) \;+\; G P_T.
$$

This has **two terms**:

1. **Static regret term**: due to the learner’s own imperfections.
   $\frac{G^2}{\lambda}(1 + \ln T)$

2. **Pathwise regret term**: due to drift in the comparator sequence.
   $G P_T$

---

## Step 3. Compute Each Term

1. **Static term**

$$
\frac{G^2}{\lambda}(1 + \ln T) = \frac{(2.0)^2}{0.5} \times (1 + \ln 1000).
$$

* $(2.0)^2 = 4$.
* $4 / 0.5 = 8$.
* $1 + \ln 1000 \approx 1 + 6.91 = 7.91$.

So the static term ≈ $8 \times 7.91 = 63.3$.

---

2. **Path term**

$$
G P_T = 2.0 \times 25 = 50.
$$

---

## Step 4. Combine

$$
R_T^{dyn} \;\leq\; 63.3 + 50 \;=\; 113.3.
$$

So, across 1,000 events, the **total dynamic regret** is bounded above by \~113.

---

## Step 5. Interpret the Decomposition

* **Static part (≈ 63)**:
  This is the “cost of being an online learner.” Even if the environment never drifted ($P_T = 0$), the learner would still accumulate regret that grows only **logarithmically** with time (about $\ln T$).
  In practice: the algorithm is nearly as good as the best fixed model chosen in hindsight.

* **Pathwise part (≈ 50)**:
  This is the “cost of drift.” Because the best model itself changes slightly over time, we must pay extra regret proportional to how far that comparator wanders.
  In practice: smooth changes in the data distribution translate into a linear adjustment term.

* **Together**: The algorithm adapts to streaming data with drift, while ensuring that the static term remains logarithmic (a very favorable guarantee compared to the classical $O(\sqrt{T})$ bounds).

---

## Step 6. Average Regret

For intuition, divide by $T = 1000$:

$$
\frac{R_T^{dyn}}{T} \;\leq\; \frac{113.3}{1000} \approx 0.113.
$$

So on average, each prediction is at most **0.113 worse than the drifting comparator**.

---

## Step 7. Relating to Deletions

If the run included **m deletions**, we’d add the deletion penalty term:

$$
\Delta_m = m \, G \, \sigma_{step} \sqrt{2 \ln(1/\delta_B)},
$$

where $\sigma_{step}$ is the noise scale used for certified unlearning.
For example, if $m=10$, $G=2$, $\sigma_{step}=0.05$, and $\delta_B=0.1$:

$$
\Delta_m \approx 10 \times 2 \times 0.05 \times \sqrt{2 \ln 10} \approx 3.4.
$$

This would bump the total regret bound slightly, from 113.3 → 116.7.

---

# 📊 Statistician’s Takeaway

* The **static term** is like the “bias floor” of the algorithm: it reflects the inherent difficulty of being online rather than batch-trained.
* The **path term** is like a “penalty for drift”: if the true best model shifts, we can’t track it perfectly.
* Both terms are **additive**, making the decomposition interpretable: one part belongs to the learner, the other to the environment.
* Deletions add a further **noise-induced term** proportional to the number of deletes.

This gives a clean way to quantify how regret arises from different sources.