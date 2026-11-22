#!/usr/bin/env python3
"""
Learned vs rule-based irrigation management in a process-based crop simulator
(AquaCrop-OSPy). Companion code for "Path B" of the revised manuscript.

What this script does
---------------------
1. Wraps AquaCrop-OSPy as a weekly-decision environment (one episode = one season).
2. Trains a tabular Q-learning management policy (actions: 0, 15 or 30 mm irrigation)
   whose reward is the economic return of the season: crop revenue minus water and
   application costs.
3. Evaluates the learned policy on held-out years against:
      - Rainfed (no irrigation)
      - Rule policy: irrigate 30 mm when root-zone depletion exceeds the FAO-56
        management-allowed depletion p (the simulator analogue of Table 4)
      - Fixed schedule: 15 mm every week
4. Writes tables (CSV + Markdown), statistical tests and figures.

Scope (state this in the paper)
-------------------------------
AquaCrop-OSPy simulates water-limited growth. It has no nitrogen or potassium
module, so only the irrigation decision (A0 vs A1) can be learned here. N and K
decisions need a nutrient-aware simulator such as DSSAT (gym-DSSAT). All outcomes
are simulated, not field-measured.

Usage
-----
  pip install aquacrop pandas numpy scipy matplotlib
  # Demo with weather bundled in AquaCrop-OSPy (Brussels, 1976-2005):
  python crop_rl_policy.py --out results_demo
  # With Dutch KNMI daily data (e.g. etmgeg_260.txt for De Bilt):
  python crop_rl_policy.py --knmi etmgeg_260.txt --train-years 1991-2014 \
         --test-years 2015-2024 --out results_knmi
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass, asdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aquacrop import (AquaCropModel, Crop, InitialWaterContent,  # noqa: E402
                      IrrigationManagement, Soil)
from aquacrop.utils import get_filepath, prepare_weather  # noqa: E402


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class Config:
    crop: str = "PotatoGDD"
    planting: str = "04/20"            # MM/DD
    season_end: str = "10/31"          # MM/DD, simulation stops at harvest anyway
    soils: tuple = ("Sand", "LoamySand", "SandyLoam")
    init_wc_pct_train: tuple = (100, 75, 50)   # % of TAW at planting (training variety)
    init_wc_pct_test: tuple = (75,)
    decision_interval: int = 7         # days between decisions
    irr_depths: tuple = (0.0, 15.0, 30.0)  # mm, action set
    # Economics -- ASSUMPTIONS: replace with sourced Dutch values in the paper
    price_eur_per_t_fresh: float = 150.0   # ware potato price, EUR / t fresh
    water_cost_eur_per_mm: float = 1.5     # EUR / mm / ha (pumping + water)
    event_cost_eur: float = 20.0           # EUR / ha per irrigation event (labour, energy)
    reward_scale: float = 1000.0           # rewards in kEUR/ha for learning stability
    # Variance reduction: terminal reward = revenue relative to the SAME season grown
    # rainfed. The baseline does not depend on the agent's actions, so the optimal
    # policy is unchanged, but the year-to-year yield noise is removed from the signal.
    rainfed_baseline: bool = True
    # Rule baseline: FAO-56 depletion fraction p for potato (Allen et al. 1998, Table 22)
    rule_p: float = 0.35
    rule_depth: float = 30.0
    fixed_depth: float = 15.0
    # Q-learning
    episodes: int = 3000
    alpha: float = 0.1
    gamma: float = 1.0                 # episodic, finite horizon (~14 decisions)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.7
    seeds: tuple = (0, 1, 2, 3, 4)
    # State discretisation
    dep_bins: tuple = (0.10, 0.25, 0.35, 0.50)   # root-zone depletion fraction Dr/TAW
    rain_bins: tuple = (5.0, 20.0)               # mm in previous 7 days
    cc_bins: tuple = (0.3, 0.7)                  # canopy cover fraction


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------
def load_knmi(path: str) -> pd.DataFrame:
    """Read a KNMI daily station file (etmgeg_XXX.txt) into AquaCrop format.

    Uses TN, TX (0.1 degC), RH (0.1 mm; -1 means < 0.05 mm) and EV24 (Makkink
    reference evapotranspiration, 0.1 mm). Note: AquaCrop expects FAO-56
    Penman-Monteith ET0; Makkink is a close substitute in the Netherlands but this
    should be stated in the paper.
    """
    with open(path, encoding="latin-1") as f:
        lines = f.readlines()
    hdr_idx = next(i for i, l in enumerate(lines) if "YYYYMMDD" in l)
    cols = [c.strip().lstrip("#").strip() for c in lines[hdr_idx].split(",")]
    rows = [l.split(",") for l in lines[hdr_idx + 1:] if l.strip() and not l.startswith("#")]
    df = pd.DataFrame([[v.strip() for v in r] for r in rows], columns=cols[:len(rows[0])])
    out = pd.DataFrame({
        "MinTemp": pd.to_numeric(df["TN"], errors="coerce") / 10.0,
        "MaxTemp": pd.to_numeric(df["TX"], errors="coerce") / 10.0,
        "Precipitation": pd.to_numeric(df["RH"], errors="coerce").clip(lower=0) / 10.0,
        "ReferenceET": pd.to_numeric(df["EV24"], errors="coerce") / 10.0,
        "Date": pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d"),
    })
    out = out.dropna(subset=["Date"]).reset_index(drop=True)
    out[["MinTemp", "MaxTemp", "Precipitation", "ReferenceET"]] = (
        out[["MinTemp", "MaxTemp", "Precipitation", "ReferenceET"]].interpolate(limit_direction="both"))
    out["ReferenceET"] = out["ReferenceET"].clip(lower=0.1)
    return out


def load_weather(knmi_path: str | None) -> pd.DataFrame:
    if knmi_path:
        return load_knmi(knmi_path)
    return prepare_weather(get_filepath("brussels_climate.txt"))


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class AquaCropIrrigationEnv:
    """One episode = one growing season; one step = `decision_interval` days."""

    def __init__(self, weather: pd.DataFrame, cfg: Config):
        self.weather = weather
        self.cfg = cfg
        self.n_actions = len(cfg.irr_depths)
        self.n_states = 5 * (len(cfg.dep_bins) + 1) * (len(cfg.rain_bins) + 1) * (len(cfg.cc_bins) + 1)

    # --- state -----------------------------------------------------------
    def _raw_state(self):
        ic = self.model._init_cond
        dep_frac = ic.depletion / ic.taw if ic.taw > 0 else 0.0
        t = self.model._clock_struct.time_step_counter
        rain7 = float(np.sum(self.model._weather[max(0, t - 7):t, 2])) if t > 0 else 0.0
        return dict(stage=int(min(max(ic.growth_stage, 0), 4)), dep=float(dep_frac),
                    rain7=rain7, cc=float(ic.canopy_cover))

    def encode(self, s: dict) -> int:
        c = self.cfg
        d = int(np.digitize(s["dep"], c.dep_bins))
        r = int(np.digitize(s["rain7"], c.rain_bins))
        k = int(np.digitize(s["cc"], c.cc_bins))
        nd, nr, nk = len(c.dep_bins) + 1, len(c.rain_bins) + 1, len(c.cc_bins) + 1
        return ((s["stage"] * nd + d) * nr + r) * nk + k

    # --- episode ---------------------------------------------------------
    def reset(self, year: int, soil: str, init_pct: float):
        c = self.cfg
        self.model = AquaCropModel(
            sim_start_time=f"{year}/{c.planting}", sim_end_time=f"{year}/{c.season_end}",
            weather_df=self.weather, soil=Soil(soil),
            crop=Crop(c.crop, planting_date=c.planting),
            initial_water_content=InitialWaterContent(wc_type="Pct", value=[init_pct]),
            irrigation_management=IrrigationManagement(irrigation_method=5, MaxIrr=max(c.irr_depths) + 1),
        )
        self.baseline_yield = self._rainfed_yield(year, soil, init_pct) if c.rainfed_baseline else 0.0
        self.model._initialize()
        self.irr_total, self.n_events, self.done = 0.0, 0, False
        return self._raw_state()

    def _rainfed_yield(self, year, soil, init_pct):
        key = (year, soil, init_pct)
        if not hasattr(self, "_cache"):
            self._cache = {}
        if key not in self._cache:
            c = self.cfg
            m = AquaCropModel(
                sim_start_time=f"{year}/{c.planting}", sim_end_time=f"{year}/{c.season_end}",
                weather_df=self.weather, soil=Soil(soil), crop=Crop(c.crop, planting_date=c.planting),
                initial_water_content=InitialWaterContent(wc_type="Pct", value=[init_pct]),
                irrigation_management=IrrigationManagement(irrigation_method=0))
            m.run_model(till_termination=True)
            self._cache[key] = float(m._outputs.final_stats.iloc[0]["Fresh yield (tonne/ha)"])
        return self._cache[key]

    def step(self, action: int):
        c = self.cfg
        depth = c.irr_depths[action]
        cost = 0.0
        if depth > 0:
            cost = depth * c.water_cost_eur_per_mm + c.event_cost_eur
            self.irr_total += depth
            self.n_events += 1
        for day in range(c.decision_interval):
            # irrigation is applied on the first day of the decision interval only
            self.model._param_struct.IrrMngt.depth = depth if day == 0 else 0.0
            self.model.run_model(num_steps=1, initialize_model=False)
            if self.model._clock_struct.model_is_finished:
                self.done = True
                break
        reward = -cost
        info = {}
        if self.done:
            fs = self.model._outputs.final_stats.iloc[0]
            fresh = float(fs["Fresh yield (tonne/ha)"])
            reward += (fresh - self.baseline_yield) * c.price_eur_per_t_fresh
            info = dict(fresh_yield=fresh, dry_yield=float(fs["Dry yield (tonne/ha)"]),
                        irrigation_mm=self.irr_total, events=self.n_events)
        return self._raw_state(), reward / c.reward_scale, self.done, info


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------
def rule_policy(cfg):
    """Simulator analogue of Table 4: irrigate when depletion exceeds FAO-56 p."""
    a_rule = cfg.irr_depths.index(cfg.rule_depth)
    return lambda s, env: a_rule if (s["dep"] > cfg.rule_p and s["stage"] in (1, 2, 3)) else 0


def rainfed_policy(cfg):
    return lambda s, env: 0


def fixed_policy(cfg):
    a = cfg.irr_depths.index(cfg.fixed_depth)
    return lambda s, env: a if s["stage"] in (1, 2, 3) else 0


def q_policy(Q):
    return lambda s, env: int(np.argmax(Q[env.encode(s)]))


def run_episode(env, policy, year, soil, init_pct):
    s = env.reset(year, soil, init_pct)
    ret, done, info = 0.0, False, {}
    while not done:
        a = policy(s, env)
        s, r, done, info = env.step(a)
        ret += r
    # absolute season profit (add back the rainfed baseline revenue if it was subtracted)
    info["profit_eur"] = ret * env.cfg.reward_scale + env.baseline_yield * env.cfg.price_eur_per_t_fresh
    return info


# --------------------------------------------------------------------------
# Q-learning
# --------------------------------------------------------------------------
def train_q(args):
    seed, weather, cfg, scenarios = args
    rng = np.random.default_rng(seed)
    env = AquaCropIrrigationEnv(weather, cfg)
    Q = np.zeros((env.n_states, env.n_actions))
    visits = np.zeros_like(Q)
    curve = []
    n_decay = max(1, int(cfg.episodes * cfg.eps_decay_frac))
    for ep in range(cfg.episodes):
        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * ep / n_decay)
        year, soil, init_pct = scenarios[rng.integers(len(scenarios))]
        s = env.reset(year, soil, init_pct)
        x = env.encode(s)
        ret, done = 0.0, False
        while not done:
            a = int(rng.integers(env.n_actions)) if rng.random() < eps else int(np.argmax(Q[x]))
            s2, r, done, info = env.step(a)
            x2 = env.encode(s2)
            target = r if done else r + cfg.gamma * np.max(Q[x2])
            Q[x, a] += cfg.alpha * (target - Q[x, a])
            visits[x, a] += 1
            ret += r
            x = x2
        curve.append(dict(seed=seed, episode=ep, epsilon=eps, return_keur=ret,
                          irrigation_mm=info["irrigation_mm"], fresh_yield=info["fresh_yield"]))
    return seed, Q, visits, curve


# --------------------------------------------------------------------------
# Evaluation and statistics
# --------------------------------------------------------------------------
def evaluate(weather, cfg, policies: dict, scenarios):
    env = AquaCropIrrigationEnv(weather, cfg)
    rows = []
    for name, pol in policies.items():
        for (year, soil, init_pct) in scenarios:
            info = run_episode(env, pol, year, soil, init_pct)
            rows.append(dict(policy=name, year=year, soil=soil, init_pct=init_pct, **info))
    return pd.DataFrame(rows)


def add_water_productivity(df):
    base = df[df.policy == "Rainfed"].set_index(["year", "soil", "init_pct"])["fresh_yield"]
    key = list(zip(df.year, df.soil, df.init_pct))
    df["yield_gain_t"] = df["fresh_yield"].values - base.loc[key].values
    # Irrigation water productivity: extra fresh yield (kg) per m3 irrigation water
    # 1 mm over 1 ha = 10 m3 ; 1 t = 1000 kg
    df["IWP_kg_per_m3"] = np.where(df.irrigation_mm > 0,
                                   df.yield_gain_t * 1000.0 / (df.irrigation_mm * 10.0), np.nan)
    return df


def summarise(df):
    def ci95(x):
        x = np.asarray(x, float)
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return np.nan
        from scipy import stats
        return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    metrics = ["fresh_yield", "irrigation_mm", "events", "IWP_kg_per_m3", "profit_eur"]
    out = []
    for p, g in df.groupby("policy", sort=False):
        row = {"policy": p, "n": len(g)}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_sd"] = g[m].std(ddof=1)
            row[f"{m}_ci95"] = ci95(g[m])
        out.append(row)
    return pd.DataFrame(out)


def paired_tests(df, ref="Rule (FAO-56 p)"):
    from scipy import stats
    res = []
    learned = sorted(p for p in df.policy.unique() if p.startswith("Q-learning"))
    key = ["year", "soil", "init_pct"]
    r = df[df.policy == ref].set_index(key)
    for p in learned:
        q = df[df.policy == p].set_index(key).loc[r.index]
        for m in ["profit_eur", "irrigation_mm", "fresh_yield"]:
            d = q[m].values - r[m].values
            try:
                w = stats.wilcoxon(q[m].values, r[m].values, zero_method="zsplit")
                pval = w.pvalue
            except ValueError:
                pval = np.nan
            res.append(dict(comparison=f"{p} vs {ref}", metric=m, mean_diff=d.mean(),
                            sd_diff=d.std(ddof=1), n_pairs=len(d), wilcoxon_p=pval))
    out = pd.DataFrame(res)
    if len(out):  # Holm correction within each metric
        out["p_holm"] = np.nan
        for m, g in out.groupby("metric"):
            order = g["wilcoxon_p"].sort_values().index
            k = len(order)
            running = 0.0
            for i, idx in enumerate(order):
                running = max(running, min(1.0, (k - i) * out.loc[idx, "wilcoxon_p"]))
                out.loc[idx, "p_holm"] = running
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_figures(curves, test_df, Q_list, env, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Learning curve (training return, mean +/- SD over seeds, smoothed)
    c = pd.DataFrame(curves)
    c["smooth"] = c.groupby("seed")["return_keur"].transform(lambda x: x.rolling(100, min_periods=1).mean())
    g = c.groupby("episode")["smooth"]
    m, s = g.mean(), g.std().fillna(0)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(m.index, m.values, color="#1f5f8b", lw=1.5)
    ax.fill_between(m.index, m - s, m + s, color="#1f5f8b", alpha=0.2, lw=0)
    ax.set_xlabel("Training episode (season)")
    ax.set_ylabel(("Return vs rainfed season (kEUR/ha)" if env.cfg.rainfed_baseline else "Season return (kEUR/ha)") + "\n100-episode moving mean")
    ax.set_title("Q-learning training curve (mean ± SD over seeds)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_learning_curve.png"), dpi=300)
    plt.close(fig)

    # 2. Profit and irrigation per policy on test scenarios
    order = list(test_df.policy.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, lab in [(axes[0], "profit_eur", "Season profit (EUR/ha)"),
                         (axes[1], "irrigation_mm", "Seasonal irrigation (mm)")]:
        data = [test_df[test_df.policy == p][col].values for p in order]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([p.replace(" (", "\n(") for p in order], fontsize=7, rotation=0)
        ax.set_ylabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Held-out test seasons", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_policy_comparison.png"), dpi=300)
    plt.close(fig)

    # 3. Learned policy map: greedy action by depletion bin x growth stage
    #    (rain7 = lowest bin, canopy cover = highest bin), majority over seeds
    c_ = env.cfg
    nd, nr, nk = len(c_.dep_bins) + 1, len(c_.rain_bins) + 1, len(c_.cc_bins) + 1
    grid = np.zeros((5, nd))
    for st in range(5):
        for d in range(nd):
            x = ((st * nd + d) * nr + 0) * nk + (nk - 1)
            acts = [c_.irr_depths[int(np.argmax(Q[x]))] if Q[x].any() else np.nan for Q in Q_list]
            grid[st, d] = np.nanmedian(acts) if not all(np.isnan(acts)) else np.nan
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(c_.irr_depths), aspect="auto")
    edges = (0,) + tuple(c_.dep_bins) + (1,)
    ax.set_xticks(range(nd))
    ax.set_xticklabels([f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(nd)], fontsize=7)
    ax.set_yticks(range(5))
    ax.set_yticklabels(["0 pre-emerg.", "1 emergence", "2 canopy dev.", "3 yield form.", "4 late"], fontsize=7)
    ax.set_xlabel("Root-zone depletion Dr/TAW")
    ax.set_title("Learned irrigation (mm), dry week, full canopy", fontsize=10)
    for i in range(5):
        for j in range(nd):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if grid[i, j] > 15 else "black")
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="grey")
    fig.colorbar(im, ax=ax, label="mm")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_learned_policy_map.png"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_years(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def fmt(m, sd, d=1):
    return f"{m:.{d}f} ± {sd:.{d}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knmi", help="KNMI daily station file (etmgeg_XXX.txt). Default: bundled Brussels weather")
    ap.add_argument("--train-years", default="1977-1997")
    ap.add_argument("--test-years", default="1998-2005")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (0..n-1)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    cfg = Config()
    if a.episodes:
        cfg.episodes = a.episodes
    if a.seeds:
        cfg.seeds = tuple(range(a.seeds))
    os.makedirs(a.out, exist_ok=True)
    weather = load_weather(a.knmi)

    train_sc = [(y, s, w) for y in parse_years(a.train_years) for s in cfg.soils for w in cfg.init_wc_pct_train]
    test_sc = [(y, s, w) for y in parse_years(a.test_years) for s in cfg.soils for w in cfg.init_wc_pct_test]
    print(f"Training scenarios: {len(train_sc)} | test scenarios: {len(test_sc)} | "
          f"episodes: {cfg.episodes} | seeds: {list(cfg.seeds)}")

    t0 = time.time()
    with Pool(min(a.workers, len(cfg.seeds))) as pool:
        results = pool.map(train_q, [(s, weather, cfg, train_sc) for s in cfg.seeds])
    print(f"Training finished in {time.time() - t0:.0f} s")

    curves, Q_list = [], []
    policies = {"Rainfed": rainfed_policy(cfg), "Fixed 15 mm/week": fixed_policy(cfg),
                "Rule (FAO-56 p)": rule_policy(cfg)}
    for seed, Q, visits, curve in results:
        np.save(os.path.join(a.out, f"qtable_seed{seed}.npy"), Q)
        curves += curve
        Q_list.append(Q)
        policies[f"Q-learning (seed {seed})"] = q_policy(Q)
    pd.DataFrame(curves).to_csv(os.path.join(a.out, "training_curve.csv"), index=False)

    test_df = add_water_productivity(evaluate(weather, cfg, policies, test_sc))
    test_df.to_csv(os.path.join(a.out, "test_results_per_scenario.csv"), index=False)

    # Pool learned seeds into one "Q-learning" row for the summary (seed variability kept in CSV)
    pooled = test_df.copy()
    pooled.loc[pooled.policy.str.startswith("Q-learning"), "policy"] = "Q-learning (all seeds)"
    summ = summarise(pooled)
    summ.to_csv(os.path.join(a.out, "summary_table.csv"), index=False)
    seed_summ = summarise(test_df[test_df.policy.str.startswith("Q-learning")])
    seed_summ.to_csv(os.path.join(a.out, "summary_per_seed.csv"), index=False)
    tests = paired_tests(test_df)
    tests.to_csv(os.path.join(a.out, "paired_tests.csv"), index=False)

    env = AquaCropIrrigationEnv(weather, cfg)
    make_figures(curves, pooled, Q_list, env, a.out)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({**asdict(cfg), "train_years": a.train_years, "test_years": a.test_years,
                   "weather": a.knmi or "AquaCrop-OSPy brussels_climate.txt"}, f, indent=2)

    # Markdown table for the manuscript
    lines = ["| Policy | Fresh yield (t/ha) | Irrigation (mm) | Events | IWP (kg/m³) | Profit (EUR/ha) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for _, r in summ.iterrows():
        iwp = "–" if np.isnan(r.IWP_kg_per_m3_mean) else fmt(r.IWP_kg_per_m3_mean, r.IWP_kg_per_m3_sd)
        lines.append(f"| {r.policy} | {fmt(r.fresh_yield_mean, r.fresh_yield_sd)} | "
                     f"{fmt(r.irrigation_mm_mean, r.irrigation_mm_sd, 0)} | {fmt(r.events_mean, r.events_sd)} | "
                     f"{iwp} | {fmt(r.profit_eur_mean, r.profit_eur_sd, 0)} |")
    md = "\n".join(lines)
    md += "\n\nValues: mean ± SD over test scenarios (years × soils"
    md += f"; Q-learning pooled over {len(cfg.seeds)} seeds).\n\nPaired Wilcoxon tests vs rule policy (Holm-adjusted):\n\n"
    if len(tests):
        md += "| Comparison | Metric | Mean diff | SD diff | n | Wilcoxon p | Holm p |\n| --- | --- | --- | --- | --- | --- | --- |\n"
        md += "\n".join(f"| {t.comparison} | {t.metric} | {t.mean_diff:.2f} | {t.sd_diff:.2f} | {t.n_pairs} | "
                        f"{t.wilcoxon_p:.3g} | {t.p_holm:.3g} |" for t in tests.itertuples())
    else:
        md += "(none)"
    with open(os.path.join(a.out, "summary_table.md"), "w") as f:
        f.write(md + "\n")
    print("\n" + md)
    print(f"\nAll outputs written to: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
