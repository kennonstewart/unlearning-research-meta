# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Grid Runner Analysis
# Structured notebook paired with analysis.ipynb.
# - Clean diffs via Jupytext, formatting via nbQA Black/Ruff
# - Sections mirror the analysis questions (variance, knees, privacy–utility, deletions vs regret, etc.)

# %%
import numpy as np
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("talk")
sns.set_style("whitegrid")

# %% [markdown]
# ## 0) Config & Inputs

# %%
# Path to event-level parquet files
# - If using grid_runner with event outputs: .../results_parquet/events/<grid_id>/*.parquet
EVENTS_PATH = pathlib.Path("results_parquet/events/")
assert EVENTS_PATH.exists(), f"Path not found: {EVENTS_PATH}"


# %% [markdown]
# ## 1) Load a Sample of observations


# %%
# list all parquet files in path
print(f"Looking for parquet files in {EVENTS_PATH}...")
files = list(EVENTS_PATH.glob("./**/*.parquet"))
print(f"Found {len(files)} files.")

# sample 25 files
files_sample = np.random.choice(files, min(30, len(files)), replace=False)
print(f"Using {len(files_sample)} files for analysis.")


# %%
def load_events(files: list) -> pd.DataFrame:
    data = {}
    for file in files:
        print(f"Loading {file}...")
        df = pd.read_parquet(file)
        # get grid id from grandparent folder name
        grid_id = file.parent.parent.name
        df["grid_id"] = grid_id
        df["seed"] = file.parent.name.split("seed=")[-1]
        data[file] = df
    return pd.concat(data, ignore_index=True)


df = load_events(files_sample).copy()
print(df.shape)
# print sorted list of columns
print(sorted(df.columns.tolist()))
# run_id combines grid and seed for grouping
if "grid_id" in df.columns and "seed" in df.columns:
    df["run_id"] = df["grid_id"].astype(str) + ":" + df["seed"].astype(str)
else:
    df["run_id"] = "unknown"

# basic flags
df["is_delete"] = (df.get("event_type", "") == "delete") | (
    df.get("op", "") == "delete"
)
df["is_insert"] = (df.get("event_type", "") == "insert") | (
    df.get("op", "") == "insert"
)

# helpful cumulative counts
df["deletes_cum"] = df.groupby("run_id")["is_delete"].cumsum()
df["inserts_cum"] = df.groupby("run_id")["is_insert"].cumsum()
df["del_ratio"] = df["deletes_cum"] / df["inserts_cum"].clip(lower=1)

print(df[["grid_id", "seed"]].drop_duplicates().shape[0], "runs loaded")

# %% [markdown]
# ## 2) Helper utilities


# %%
def per_run_summary(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.sort_values("event").groupby(["grid_id", "seed", "run_id"], dropna=False)
    out = g.agg(
        final_avg_regret=("avg_regret", "last"),
        final_acc=("acc", "last"),
        total_deletes=("is_delete", "sum"),
        total_inserts=("is_insert", "sum"),
        final_rho_spent=("rho_spent", "last"),
        final_rho_remaining=("rho_remaining", "last"),
        final_sigma_step=("sigma_step", "last"),
        final_m_used=("m_used", "last"),
        final_m_capacity=("m_capacity", "last"),
        N_gamma=("N_gamma", "last"),
        final_S_scalar=("S_scalar", "last"),
        final_P_T=("P_T", "last"),
        final_P_T_est=("P_T_est", "last"),
    ).reset_index()
    out["del_ratio"] = out["total_deletes"] / out["total_inserts"].clip(lower=1)
    out["rho_frac"] = out["final_rho_spent"] / (
        (
            out["final_rho_spent"].fillna(0) + out["final_rho_remaining"].fillna(0)
        ).replace(0, np.nan)
    )
    return out


runs = per_run_summary(df)
runs.head()

# %% [markdown]
# ## 3) Within-grid between-seed variance (sanity)

# %%
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(
    data=runs.dropna(subset=["final_avg_regret"]),
    x="grid_id",
    y="final_avg_regret",
    ax=ax,
)
ax.set_title("Final avg_regret by grid_id (seed variance)")
ax.set_xlabel("grid_id")
ax.set_ylabel("final_avg_regret")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4) Is avg_regret monotone? Knees vs N_gamma and delete bursts


# %%
def detect_bursts(
    frame: pd.DataFrame, window: int = 200, threshold: int = 10
) -> pd.Series:
    r = frame["is_delete"].rolling(window, min_periods=1).sum() >= threshold
    starts = r & (~r.shift(1, fill_value=False))
    return starts


def plot_regret_with_annotations(run: pd.DataFrame, title: str = ""):
    run = run.sort_values("event").copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(run["event"], run["avg_regret"], alpha=0.8, label="avg_regret")
    # N_gamma line (if set)
    if pd.notna(run["N_gamma"].iloc[-1]):
        ax.axvline(run["N_gamma"].iloc[-1], color="red", ls="--", label="N_gamma")
    # bursts
    starts = detect_bursts(run)
    for e in run.loc[starts, "event"]:
        ax.axvline(e, color="k", alpha=0.15)
    ax.set_title(title or run["run_id"].iloc[0])
    ax.set_xlabel("event")
    ax.set_ylabel("avg_regret")
    ax.legend()
    plt.tight_layout()
    plt.show()


# pick the largest spread grid
spread = runs.groupby("grid_id")["final_avg_regret"].std().sort_values(ascending=False)
if not spread.empty:
    gid = spread.index[0]
    sample_run = df[df["grid_id"] == gid].groupby("run_id").head(50000)  # cap
    for rid, sub in sample_run.groupby("run_id"):
        plot_regret_with_annotations(sub, f"{gid} / {rid}")

# %% [markdown]
# ## 5) Privacy–utility frontier (rho_frac vs final_acc / final_avg_regret)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
sns.scatterplot(data=runs, x="rho_frac", y="final_acc", ax=axes[0])
axes[0].set_title("final_acc vs rho_frac")
sns.scatterplot(data=runs, x="rho_frac", y="final_avg_regret", ax=axes[1])
axes[1].set_title("final_avg_regret vs rho_frac")
for ax in axes:
    ax.set_xlabel("rho_frac = rho_spent / (rho_spent + rho_remaining)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6) Regret scaling with deletions and delete ratio

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
sns.regplot(
    data=runs,
    x="total_deletes",
    y="final_avg_regret",
    scatter_kws=dict(alpha=0.6),
    ax=axes[0],
)
axes[0].set_title("final_avg_regret vs total_deletes")
sns.regplot(
    data=runs,
    x="del_ratio",
    y="final_avg_regret",
    scatter_kws=dict(alpha=0.6),
    ax=axes[1],
)
axes[1].set_title("final_avg_regret vs del_ratio")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7) Attribution: noise vs learning dynamics
# Approx: final_avg_regret ≈ a * sqrt(S_T/N) + b * (m_used * sigma_step)/N

# %%
tmp = runs.copy()
tmp["N"] = df.groupby("run_id")["event"].max().reset_index(drop=True)
tmp["X1"] = np.sqrt(tmp["final_S_scalar"].fillna(0)) / tmp["N"].replace(0, np.nan)
tmp["X2"] = (tmp["final_m_used"].fillna(0) * tmp["final_sigma_step"].fillna(0)) / tmp[
    "N"
].replace(0, np.nan)
tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["final_avg_regret", "X1", "X2"], how="any"
)
if not tmp.empty:
    import statsmodels.api as sm

    X = sm.add_constant(tmp[["X1", "X2"]])
    y = tmp["final_avg_regret"]
    model = sm.OLS(y, X).fit()
    print(model.summary())
else:
    print("Insufficient data for attribution model.")

# %% [markdown]
# ## 8) Stepsize policy prevalence vs regret

# %%
stepshare = (
    df.groupby("run_id")
    .agg(sc_frac=("sc_active", "mean"), final_avg_regret=("avg_regret", "last"))
    .reset_index()
)
sns.scatterplot(data=stepshare, x="sc_frac", y="final_avg_regret")
plt.title("final_avg_regret vs share of strongly-convex steps")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10) Drift vs deletes attribution over windows


# %%
def window_attribution(frame: pd.DataFrame, window: int = 500) -> pd.DataFrame:
    r = frame.sort_values("event").copy()
    r["block"] = (r["event"] // window).astype(int)
    agg = r.groupby("block").agg(
        avg_regret_end=("avg_regret", "last"),
        deletes=("is_delete", "sum"),
        dP=(
            "P_T",
            lambda s: (s.dropna().iloc[-1] - s.dropna().iloc[0])
            if s.dropna().size > 0
            else 0,
        ),
    )
    agg["d_regret"] = agg["avg_regret_end"].diff().fillna(0)
    return agg.reset_index()


# pick one run to illustrate
one = df[df["run_id"] == df["run_id"].iloc[0]]
w = window_attribution(one, window=500)
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(w["block"], w["d_regret"])
axes[0].set_ylabel("Δ avg_regret")
axes[1].bar(w["block"], w["deletes"])
axes[1].set_ylabel("deletes")
axes[2].plot(w["block"], w["dP"])
axes[2].set_ylabel("Δ P_T")
axes[2].set_xlabel("block")
plt.tight_layout()
plt.show()

# %%
# Assumes df (event-level) and runs (per-run summaries) are already built

def last_nonnull(s):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan

# Per-run extras from event stream
per_run_extras = (
    df.sort_values("event")
      .groupby("run_id")
      .agg(
          N=("event", "max"),
          sc_frac=("sc_active", "mean"),
          sigma_last=("sigma_step", last_nonnull),
          rho_spent_last=("rho_spent", last_nonnull),
          rho_remaining_last=("rho_remaining", last_nonnull),
          stepsize_sc_share=("stepsize_policy", lambda s: np.mean(s == "strongly-convex")),
          P_T_last=("P_T", last_nonnull),
          P_T_est_last=("P_T_est", last_nonnull),
          sens_delete_q95=("sens_delete", lambda s: np.nan if s.isna().all() else np.nanquantile(s.dropna(), 0.95)),
          deletes_total=("event_type", lambda s: np.sum(s == "delete")),
          inserts_total=("event_type", lambda s: np.sum(s == "insert")),
      )
      .reset_index()
)

feat = runs.merge(per_run_extras, on="run_id", how="left")

# Ratios/normalizations
feat["del_ratio"] = feat["deletes_total"] / feat["inserts_total"].clip(lower=1)
feat["rho_frac"] = feat["rho_spent_last"] / (feat["rho_spent_last"].fillna(0) + feat["rho_remaining_last"].fillna(0)).replace(0, np.nan)
feat["sigma_eff"] = feat["sigma_last"]
feat["m_per_N"] = feat["final_m_used"] / feat["N"].replace(0, np.nan)
feat["Ssqrt_per_N"] = np.sqrt(feat["final_S_scalar"].fillna(0)) / feat["N"].replace(0, np.nan)
feat["P_T_final"] = feat["P_T_last"].fillna(feat["P_T_est_last"])
# Optional: join sweep/manifest.json for grid parameters if not in metrics
# manifest = json.load(open("results/.../sweep/manifest.json"))
# manifest_df = pd.DataFrame([{"grid_id": gid, **params} for gid, params in manifest.items()])
# feat = feat.merge(manifest_df, on="grid_id", how="left")

target = feat["final_avg_regret"] 
# %%
# Choose numeric candidate features (drop ID columns and the target)
cand = [
    "del_ratio", "deletes_total", "m_per_N", "rho_frac", "sigma_eff",
    "Ssqrt_per_N", "final_S_scalar", "sc_frac", "stepsize_sc_share",
    "P_T_final", "N_gamma", "final_m_capacity", "final_m_used",
    "G_hat", "D_hat", "c_hat", "C_hat"
]
cand = [c for c in cand if c in feat.columns]

X = feat[cand].copy()
y = target.copy()

pearson = X.apply(lambda col: col.corr(y), axis=0)
spearman = X.apply(lambda col: col.corr(y, method="spearman"), axis=0)

print("Top positive (Pearson):")
print(pearson.sort_values(ascending=False).head(10))
print("\nTop positive (Spearman):")
print(spearman.sort_values(ascending=False).head(10))

# Heatmap of correlations among features and with target
import seaborn as sns, matplotlib.pyplot as plt
corr_mat = feat[cand + ["final_avg_regret"]].corr(method="spearman")
plt.figure(figsize=(10, 8))
sns.heatmap(corr_mat, annot=False, cmap="vlag", center=0)
plt.title("Spearman correlations (features + final_avg_regret)")
plt.show()

# Within-grid (demean per grid_id) to isolate within-cell signal
def demean_by_group(s, g):
    return s - s.groupby(g).transform("mean")

g = feat["grid_id"]
X_w = X.apply(lambda col: demean_by_group(col, g))
y_w = demean_by_group(y, g)

pearson_w = X_w.apply(lambda col: col.corr(y_w))
spearman_w = X_w.apply(lambda col: col.corr(y_w, method="spearman"))

print("\nWithin-grid Spearman:")
print(spearman_w.sort_values(ascending=False))
# %%
controls = ["del_ratio", "Ssqrt_per_N"]
controls = [c for c in controls if c in X.columns]

def partial_corr(x, y, C):
    import statsmodels.api as sm
    # Residualize x and y on controls C, then correlate residuals
    Xc = sm.add_constant(C)
    rx = x - sm.OLS(x, Xc, missing="drop").fit().fittedvalues
    ry = y - sm.OLS(y, Xc, missing="drop").fit().fittedvalues
    return rx.corr(ry)

pcorrs = {}
for col in cand:
    if col in controls: 
        continue
    xcol = X[col]
    C = X[controls]
    pcorrs[col] = partial_corr(xcol, y, C)

print("\nPartial correlations (control: del_ratio, Ssqrt_per_N):")
print(pd.Series(pcorrs).sort_values(ascending=False))
# %%
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
import numpy as np

# Features: numeric + (optional) categorical grid_id as fixed effects proxy
num_cols = cand  # numeric features prepared above
cat_cols = []    # optionally include 'grid_id' as one-hot; with 25 samples this may be too many dummies

# Drop rows with missing y or all-NaN in X
data = feat.dropna(subset=["final_avg_regret"])[num_cols + ["final_avg_regret", "grid_id"]].copy()
X_num = data[num_cols]
y = data["final_avg_regret"].values
groups = data["grid_id"].values

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        # ("cat", OneHotEncoder(drop="first"), cat_cols),  # enable if you want grid fixed effects
    ],
    remainder="drop",
)

enet = ElasticNet(max_iter=5000)

param_grid = {
    "model__alpha": np.logspace(-3, 1, 20),
    "model__l1_ratio": [0.2, 0.5, 0.8, 1.0],  # 1.0 = LASSO
}

pipe = Pipeline(steps=[("prep", pre), ("model", enet)])
cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))

grid = GridSearchCV(
    pipe, param_grid, scoring="r2", cv=cv, n_jobs=-1, refit=True
)
grid.fit(X_num, y, groups=groups)
print("Best params:", grid.best_params_, "CV R2:", grid.best_score_)

# Inspect coefficients
best = grid.best_estimator_
coef = best.named_steps["model"].coef_
coef_series = pd.Series(coef, index=num_cols)  # (if no categorical terms)
print("\nElastic Net coefficients (standardized scale):")
print(coef_series.sort_values(ascending=False))
# %%
