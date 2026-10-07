# Thesis forecast results

All figures are PDFs styled per `CLAUDE.md` / `thesis_style.py` (no titles,
serif, muted palette, frameless legend). All numbers in **NOK/MWh** for spot
and **NOK/MW/h** for FCR (sim raw scaled ×10 and ×6.5 respectively).

## How to reproduce

```sh
# 1. Historical baseline (NO2, ≈10 s)
python compute_historical.py

# 2. Per-(path, year) aggregates from the 226 GB of raw sim files (≈16 min wall)
python compute_simulation.py spot &
python compute_simulation.py fcr &
wait

# 3. Plots and tables
python make_calibration_table.py
python plot_yearly_mean.py
python plot_joint.py
python plot_spreads.py
python plot_backtest.py
```

## Suggested thesis sections and which outputs go where

### 1. Model & calibration
- `data/calibration_table.tex` — booktabs LaTeX table of MRS parameters in
  thesis units. Both Spot and FCR side by side.

### 2. Historical baseline
- `data/historical_metrics.csv` — yearly NO2 spot (2000–2025) and FCR
  (2015–2025): mean, daily/weekly spread, intra-year amplitude, hourly vol.

### 3. Long-run forecast — yearly means
- `figures/fan_yearly_mean_spot.pdf`
- `figures/fan_yearly_mean_fcr.pdf`
- `data/decadal_summary.csv`

  Note: the simulated band is genuinely narrow because the *mean of 8760
  hourly draws per year* is tightly determined by the law of large numbers.
  The wider distribution lives in hourly prices, captured by the metrics
  below.

### 4. BESS-relevant volatility (intra-day, intra-week)
- `figures/fan_intra_day_spread_spot.pdf` — BESS daily arbitrage proxy.
- `figures/fan_intra_week_spread_spot.pdf` — BESS intra-week arbitrage proxy.
- `figures/fan_hourly_vol_spot.pdf` — hourly log-return volatility.

### 5. PSH-relevant volatility (intra-year)
- `figures/fan_intra_year_amp_spot.pdf` — max minus min monthly mean
  within each year. The natural "seasonal arbitrage" envelope.

### 6. FCR market dynamics
- `figures/fan_intra_day_spread_fcr.pdf`
- `figures/fan_intra_week_spread_fcr.pdf`
- `figures/fan_intra_year_amp_fcr.pdf`
- `figures/fan_hourly_vol_fcr.pdf`
- `data/spread_decadal_summary.csv` — decadal P10/P50/P90 of every spread
  metric for both markets.

### 7. Joint Spot–FCR
- `figures/joint_yearly_mean.pdf` — decade-coloured scatter of
  decade-mean spot vs decade-mean FCR per path.
- `data/correlation_summary.txt` — sim vs historical Pearson ρ.

### 8. Model validation / backtest
- `figures/backtest_returns_spot.pdf`,
  `figures/backtest_returns_fcr.pdf` — historical vs simulated hourly
  log-return density.
- `figures/backtest_qq_spot.pdf`,
  `figures/backtest_qq_fcr.pdf` — QQ plots.
- `data/backtest_moments.csv` — mean, std, skew, kurtosis, AR(1)
  for historical and a sampled simulation set.

  Headline finding: simulated hourly returns are wider but less peaked
  than historical (lower kurtosis). The MRS captures regime shifts and
  base volatility but under-represents the tail-mass + near-zero peakedness
  of empirical hourly behaviour.

## Honest limitations to mention in the thesis

1. **Long-run mean is a calibration artefact, not a forecast** — the
   simulated baseline (~350 NOK/MWh spot, ~132 NOK/MW/h FCR) reflects the
   training window, which includes the 2021–2024 crisis years. Real long-run
   prices will depend on policy, hydrogen/EV demand, transmission build-out,
   etc. — none of which the model knows about.
2. **No structural uncertainty** — the forecast bands reflect path-level
   stochastic variation only, not uncertainty in the parameters themselves.
3. **Hourly tail mass under-modelled** — see backtest QQ.
4. **Joint Spot–FCR correlation is ~0.5 (decade-mean) in simulation, ~0.7
   in the 11-year historical sample** — the small historical N makes the
   gap inconclusive but worth flagging.
5. **No leap-year handling**: simulation assumes 8760 h/year; real
   calendars have 8784 h in leap years. Negligible for thesis-level use.

## Files

```
data/
    historical_metrics.csv       hist NO2 yearly metrics, all markets
    historical_returns_*.npy     subsampled hourly log-returns for backtest
    sim_metrics_<market>.npz     10 000 × 70 arrays of every metric
    decadal_summary.csv          yearly-mean decadal stats
    spread_decadal_summary.csv   spread metric decadal stats
    calibration_table.csv/.tex   MRS parameter table in thesis units
    backtest_moments.csv         dist moments hist vs sim
    correlation_summary.txt      Spot–FCR ρ
figures/
    fan_yearly_mean_<market>.pdf
    fan_intra_day_spread_<market>.pdf
    fan_intra_week_spread_<market>.pdf
    fan_intra_year_amp_<market>.pdf
    fan_hourly_vol_<market>.pdf
    joint_yearly_mean.pdf
    backtest_returns_<market>.pdf
    backtest_qq_<market>.pdf
```
