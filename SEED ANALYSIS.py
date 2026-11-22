"""
Repeated-seed analysis for the CSI-based tabular Q-learning estimator.

Reproduces the released pipeline (Context_Aware_RL_Crop_Recommendation_OptionA_REVISED_v3.ipynb)
and reruns it with several seeds, so the manuscript can report mean ± SD instead of a single run.

  Mode A (training seeds): the split, CSI cut-points, state bins, the 20,000-sample subset and the
          nearest-neighbour targets are fixed (seed 42); only the Q-learning seed changes
          (epsilon-greedy draws and episode order).
  Mode B (split seeds):    the whole pipeline is rerun with a different seed for the split,
          the 20,000-sample subset and Q-learning.

Seed 42 in either mode must reproduce the paper's primary run (test MAE 0.165142, persistence
0.162626). Check this first: if it does not match, the data file or library versions differ.

It also writes Supplementary Table S1 (all CSV variables with type and observed range).

Usage:
    python repeated_seed_analysis.py --csv "agriculture_dataset(4).csv"
    python repeated_seed_analysis.py --csv data.csv --train-seeds 0 1 2 3 4 5 6 7 8 9 --split-seeds 1 2 3 4 5

Outputs (in ./repeated_seed_outputs/):
    runs.csv              one row per run
    summary.txt           mean ± SD per mode, ready to paste into the manuscript
    table_S1_variables.csv
    versions.txt          Python and library versions for the Implementation details paragraph
"""
import argparse
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---- configuration copied from the released notebook ----
CSI_WEIGHTS = {"M": 0.50, "C": 0.30, "P": 0.20}
PH_OPTIMUM_RANGES = {"Wheat": (6.0, 7.5), "Maize": (6.0, 7.5), "Rice": (4.0, 6.0)}
REQUIRED = ["Soil_Moisture", "Chlorophyll_Content", "Soil_pH", "NDVI",
            "Temperature", "Humidity", "Rainfall", "Crop_Health_Label"]
STATE_FEATURES = ["NDVI", "Soil_Moisture", "Soil_pH", "Temperature", "Humidity", "Rainfall"]
ACTIONS = np.array([0, 1, 2], dtype=int)
ALPHA, GAMMA, EPSILON, EPISODES = 0.10, 0.90, 0.20, 100
K_NEIGHBOURS, MAX_TRANSITIONS = 20, 20000
PRIMARY_SEED = 42


def load(csv):
    df = pd.read_csv(csv)
    work = df.copy()
    for col in REQUIRED:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["Crop_Type"] = work["Crop_Type"].astype(str)
    work = work.dropna(subset=REQUIRED + ["Crop_Type"]).reset_index(drop=True)
    return df, work


def ph_deviation(frame):
    bounds = frame["Crop_Type"].map(PH_OPTIMUM_RANGES)
    lo = bounds.map(lambda b: b[0]).astype(float).to_numpy()
    hi = bounds.map(lambda b: b[1]).astype(float).to_numpy()
    ph = frame["Soil_pH"].to_numpy(float)
    return np.maximum(lo - ph, 0.0) + np.maximum(ph - hi, 0.0)


def prepare(work, split_seed):
    """Split, CSI classes, state encoding, nearest-neighbour targets. Everything fitted on train only."""
    train_df, temp_df = train_test_split(work, test_size=0.30, random_state=split_seed,
                                         stratify=work["Crop_Health_Label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=split_seed,
                                       stratify=temp_df["Crop_Health_Label"])
    train_df, val_df, test_df = (f.copy().reset_index(drop=True) for f in (train_df, val_df, test_df))

    m_min, m_max = train_df["Soil_Moisture"].min(), train_df["Soil_Moisture"].max()
    c_min, c_max = train_df["Chlorophyll_Content"].min(), train_df["Chlorophyll_Content"].max()
    p_scale = float(np.max(ph_deviation(train_df)))

    def csi(frame):
        m = 1 - ((frame["Soil_Moisture"] - m_min) / (m_max - m_min)).clip(0, 1)
        c = 1 - ((frame["Chlorophyll_Content"] - c_min) / (c_max - c_min)).clip(0, 1)
        p = np.clip(ph_deviation(frame) / p_scale, 0, 1)
        return (CSI_WEIGHTS["M"] * m + CSI_WEIGHTS["C"] * c + CSI_WEIGHTS["P"] * p).to_numpy()

    for f in (train_df, val_df, test_df):
        f["CSI"] = csi(f)
    cuts = np.quantile(train_df["CSI"], [0.20, 0.40, 0.60, 0.80]).astype(float)
    for f in (train_df, val_df, test_df):
        f["Stress_Level"] = np.searchsorted(cuts, f["CSI"].to_numpy(), side="right").astype(int)

    scaler = StandardScaler().fit(train_df[STATE_FEATURES])
    xtr = scaler.transform(train_df[STATE_FEATURES])
    bins = [np.unique(np.quantile(xtr[:, j], [1 / 3, 2 / 3])) for j in range(len(STATE_FEATURES))]

    def encode(frame, scaled):
        lv = frame["Stress_Level"].to_numpy(int)
        cols = [np.searchsorted(bins[j], scaled[:, j], side="right") for j in range(len(STATE_FEATURES))]
        return [(int(lv[i]),) + tuple(int(c[i]) for c in cols) for i in range(len(frame))]

    xva, xte = scaler.transform(val_df[STATE_FEATURES]), scaler.transform(test_df[STATE_FEATURES])
    s_tr, s_va, s_te = encode(train_df, xtr), encode(val_df, xva), encode(test_df, xte)
    tr_level = train_df["Stress_Level"].to_numpy(int)

    nn = NearestNeighbors(n_neighbors=min(K_NEIGHBOURS + 1, len(train_df)), metric="euclidean", n_jobs=-1).fit(xtr)

    def transitions(levels, states, scaled, exclude=None):
        _, nbrs = nn.kneighbors(scaled, return_distance=True)
        out, fallback = [], 0
        for i, row in enumerate(nbrs):
            s = int(levels[i])
            chosen = None
            for j in row:
                if exclude is not None and int(j) == int(exclude[i]):
                    continue
                if s <= tr_level[j] <= min(4, s + 2):
                    chosen = int(j)
                    break
            if chosen is None:
                fallback += 1
                out.append((states[i], s, s, states[i]))
            else:
                out.append((states[i], s, int(tr_level[chosen]), s_tr[chosen]))
        return out, fallback

    rng = np.random.default_rng(split_seed)
    idx = np.sort(rng.choice(len(train_df), MAX_TRANSITIONS, replace=False)) if len(train_df) > MAX_TRANSITIONS \
        else np.arange(len(train_df))
    train_tr, fb = transitions(tr_level[idx], [s_tr[i] for i in idx], xtr[idx], exclude=idx)
    val_tr, _ = transitions(val_df["Stress_Level"].to_numpy(int), s_va, xva)
    test_tr, _ = transitions(test_df["Stress_Level"].to_numpy(int), s_te, xte)
    return dict(train=train_tr, val=val_tr, test=test_tr, fallback=fb, cuts=cuts,
                sizes=(len(train_df), len(val_df), len(test_df)))


def train_q(transitions, seed):
    rng = np.random.default_rng(seed)
    Q = defaultdict(lambda: np.zeros(len(ACTIONS)))
    returns = []
    for _ in range(EPISODES):
        total = 0.0
        for k in rng.permutation(len(transitions)):
            state, cur, target, nxt = transitions[int(k)]
            a = int(rng.integers(len(ACTIONS))) if rng.random() < EPSILON else int(np.argmax(Q[state]))
            r = -float(abs(min(4, cur + int(ACTIONS[a])) - target))
            Q[state][a] += ALPHA * (r + GAMMA * float(np.max(Q[nxt])) - Q[state][a])
            total += r
        returns.append(total)
    return Q, np.asarray(returns)


def evaluate(transitions, Q):
    y = np.array([t[2] for t in transitions])
    cur = np.array([t[1] for t in transitions])
    pred = np.array([min(4, t[1] + int(ACTIONS[int(np.argmax(Q[t[0]]))])) for t in transitions])
    return dict(q_mae=mean_absolute_error(y, pred), persist_mae=mean_absolute_error(y, cur),
                q_exact=float(np.mean(pred == y)), q_within1=float(np.mean(np.abs(pred - y) <= 1)),
                persist_exact=float(np.mean(cur == y)))


def run(work, split_seed, q_seed, mode):
    prep = prepare(work, split_seed)
    Q, returns = train_q(prep["train"], q_seed)
    row = dict(mode=mode, split_seed=split_seed, q_seed=q_seed, n_states=len(Q),
               final10_return=float(np.mean(returns[-10:])), fallback=prep["fallback"])
    for part in ("val", "test"):
        for k, v in evaluate(prep[part], Q).items():
            row[f"{part}_{k}"] = v
    print(f"[{mode}] split={split_seed} q={q_seed}  test MAE Q={row['test_q_mae']:.6f}  "
          f"persistence={row['test_persist_mae']:.6f}  states={len(Q)}", flush=True)
    return row, prep


def variable_table(df):
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s)
        rows.append(dict(variable=c, dtype=str(s.dtype), n_unique=int(s.nunique()),
                         min=s.min() if num else "", max=s.max() if num else "",
                         mean=round(float(s.mean()), 4) if num else "",
                         example=s.iloc[0]))
    return pd.DataFrame(rows)


def msd(x):
    return f"{np.mean(x):.6f} ± {np.std(x, ddof=1):.6f}" if len(x) > 1 else f"{x[0]:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--train-seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--split-seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="repeated_seed_outputs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    df, work = load(a.csv)
    variable_table(df).to_csv(out / "table_S1_variables.csv", index=False)
    (out / "versions.txt").write_text(
        f"Python {platform.python_version()}\nNumPy {np.__version__}\npandas {pd.__version__}\n"
        f"scikit-learn {sklearn.__version__}\n")

    rows = []
    # Primary run (reproducibility check) and mode A share the fixed split; prepare it once.
    prep = prepare(work, PRIMARY_SEED)
    for q_seed in [PRIMARY_SEED] + [s for s in a.train_seeds if s != PRIMARY_SEED]:
        Q, returns = train_q(prep["train"], q_seed)
        row = dict(mode="primary" if q_seed == PRIMARY_SEED else "train_seed", split_seed=PRIMARY_SEED,
                   q_seed=q_seed, n_states=len(Q), final10_return=float(np.mean(returns[-10:])),
                   fallback=prep["fallback"])
        for part in ("val", "test"):
            for k, v in evaluate(prep[part], Q).items():
                row[f"{part}_{k}"] = v
        rows.append(row)
        print(f"[{row['mode']}] q_seed={q_seed}  test MAE Q={row['test_q_mae']:.6f}  "
              f"persistence={row['test_persist_mae']:.6f}", flush=True)
    for s in a.split_seeds:
        rows.append(run(work, s, s, "split_seed")[0])

    res = pd.DataFrame(rows)
    res.to_csv(out / "runs.csv", index=False)

    p = res[res["mode"] == "primary"].iloc[0]
    lines = [f"Primary run (seed {PRIMARY_SEED}): test MAE Q-learning {p.test_q_mae:.6f}, persistence "
             f"{p.test_persist_mae:.6f}  [paper: 0.165142 and 0.162626]"]
    for mode, label in (("train_seed", "training seeds (fixed split)"), ("split_seed", "split seeds (full pipeline)")):
        sub = res[res["mode"].isin([mode, "primary"] if mode == "train_seed" else [mode])]
        if sub.empty:
            continue
        worse = int((sub.test_q_mae > sub.test_persist_mae).sum())
        lines.append(f"{label}: n = {len(sub)} runs; test MAE Q-learning {msd(sub.test_q_mae.values)}; "
                     f"persistence {msd(sub.test_persist_mae.values)}; exact agreement Q-learning "
                     f"{msd(100 * sub.test_q_exact.values)}%; Q-learning MAE > persistence in {worse}/{len(sub)} runs")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    sys.exit(main())
