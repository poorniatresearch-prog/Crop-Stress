# Learned vs rule-based irrigation policy in AquaCrop-OSPy ("Path B")

This code goes with the revised manuscript. It trains a **tabular Q-learning
management policy** in the process-based crop simulator AquaCrop-OSPy. It then
compares that policy with the rule layer (Table 4 analogue), rainfed management
and a fixed schedule on **held-out seasons**. It answers Editor point 1,
R1.4, R3.9, R3.16 and Reviewer 4 ("a decision policy that is actually learned").

## 1. Install (Python 3.10 or newer)

```bash
pip install aquacrop pandas numpy scipy matplotlib
```

## 2. Run

```bash
# (a) Demo with the Brussels weather bundled in AquaCrop-OSPy (1976-2005)
python crop_rl_policy.py --out results_demo

# (b) Paper run with Dutch weather from KNMI
#     Download the daily file for a station, e.g. De Bilt (260):
#     https://www.knmi.nl/nederland-nu/klimatologie/daggegevens  -> etmgeg_260.txt
python crop_rl_policy.py --knmi etmgeg_260.txt \
       --train-years 1991-2014 --test-years 2015-2024 --out results_knmi
```

Options: `--episodes 3000` (training seasons per seed), `--seeds 5`,
`--workers N` (parallel seeds). The default run takes about 25 minutes on 2 CPU cores.

## 3. What the environment is

| Element | Definition |
| --- | --- |
| Episode | One potato season (`PotatoGDD`, planted 20 April) on one soil (Sand, LoamySand, SandyLoam) |
| Decision step | Every 7 days |
| State | Growth stage (0-4) x root-zone depletion Dr/TAW (5 bins) x rain in the previous 7 days (3 bins) x canopy cover (3 bins) = 225 states |
| Actions | Irrigate 0, 15 or 30 mm (= A0 / A1 at two depths) |
| Reward | -(1.5 EUR/mm x depth + 20 EUR per event) at each step; at harvest + 150 EUR/t x (fresh yield - fresh yield of the same season grown rainfed). The rainfed term does not depend on the agent's actions, so the optimal policy is unchanged; it removes year-to-year yield noise from the learning signal |
| Algorithm | Tabular Q-learning, alpha = 0.1, gamma = 1 (episodic), epsilon 1.0 -> 0.05 over 70% of episodes |
| Train / test | Different years (no year appears in both) |
| Baselines | Rainfed; Rule: irrigate 30 mm when Dr/TAW > p = 0.35 (FAO-56, potato); Fixed 15 mm per week |

The prices and costs in `Config` are **assumptions**. Replace them with sourced
Dutch values (e.g., Wageningen Economic Research / KWIN-AGV) and cite them.

## 4. Outputs (in `--out`)

| File | Use in the paper |
| --- | --- |
| `summary_table.md` / `summary_table.csv` | New results table: yield, irrigation, events, irrigation water productivity, profit (mean +/- SD, 95% CI) |
| `paired_tests.csv` | Wilcoxon signed-rank tests, learned vs rule, Holm-adjusted |
| `summary_per_seed.csv` | Seed-to-seed variability (repeated runs) |
| `test_results_per_scenario.csv` | Raw per-season results (Supplementary) |
| `fig_learning_curve.png` | Training convergence (mean +/- SD over seeds) |
| `fig_policy_comparison.png` | Profit and irrigation per policy on test seasons |
| `fig_learned_policy_map.png` | What the learned policy does, by growth stage and depletion |
| `qtable_seed*.npy`, `config.json`, `training_curve.csv` | Reproducibility |

## 5. Expected output (demo run, Brussels weather; train 1977-1997, test 1998-2005)

5 seeds x 3,000 episodes, about 26 min on 2 cores. Test = 8 years x 3 soils = 24 seasons.

| Policy | Fresh yield (t/ha) | Irrigation (mm) | Events | IWP (kg/m3) | Profit (EUR/ha) |
| --- | --- | --- | --- | --- | --- |
| Rainfed | 37.4 +/- 5.3 | 0 +/- 0 | 0.0 +/- 0.0 | - | 5615 +/- 802 |
| Fixed 15 mm/week | 40.3 +/- 2.5 | 156 +/- 7 | 10.4 +/- 0.5 | 1.9 +/- 2.2 | 5600 +/- 365 |
| Rule (FAO-56 p) | 40.3 +/- 2.4 | 82 +/- 39 | 2.8 +/- 1.3 | 2.8 +/- 3.1 | 5862 +/- 393 |
| Q-learning (all seeds) | 39.4 +/- 3.2 | 51 +/- 24 | 2.4 +/- 1.1 | 3.4 +/- 4.0 | 5782 +/- 500 |

Reading: the learned policy uses 38% less water than the FAO-56 rule. It loses
0.9 t/ha of yield and EUR 80/ha (1.4%) of profit, which is not significant after
Holm correction for 4 of the 5 seeds. It beats rainfed and the fixed schedule.
The rule is a strong baseline. Your KNMI numbers will differ. Report them as they
come out. If the training curve is still rising, try `--episodes 6000`.

## 6. Limits to state in the manuscript

* AquaCrop-OSPy has **no nitrogen or potassium module**. Only the irrigation
  decision is learned here. N and K decisions need DSSAT (gym-DSSAT).
  Crop switching is a between-season decision and is not simulated.
* Outcomes are **simulated**, not field-measured. Default crop and soil
  parameters are not calibrated for Dutch fields.
* KNMI `EV24` is Makkink reference ET, not FAO-56 Penman-Monteith ET0.
* The Brussels demo is a check that the code runs. Report the KNMI run in the paper.
