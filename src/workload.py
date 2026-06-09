from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HORIZON_HOURS = 168  # one week


# ─────────────────────────────────────────────────────────────────────────────
# Class-specific 24-hour IT load profile generators
# Each function returns a raw (24,) array whose values are then normalised to
# mean = 1.0 by _week_profile_for_task before scaling to MW.
#
# Dispatch key: task["flexibility_hours"]
#   0  → _interactive_24h   Class 1 – Interactive / latency-critical
#   2  → _dag_pipeline_24h  Class 2 – DAG-constrained pipeline
#   12 → _batch_ml_24h      Class 3 – Delay-tolerant batch / ML training
#   24 → _best_effort_24h   Class 4 – Best-effort / background
#
# Any flexibility_hours value not in the dispatch table falls back to
# _interactive_24h so that existing YAML files with non-standard values
# continue to run without error.
# ─────────────────────────────────────────────────────────────────────────────

def _interactive_24h(seed: int) -> np.ndarray:
    """Class 1 – Interactive / latency-critical  (W = 0 h).

    Strong sinusoidal diurnal profile peaking at noon (12:00) and
    troughing at midnight (00:00).

    Equation:
        l_t = clip(0.5 + 0.5·sin(2π(t−6)/24) + ε_t,  0.30, 1.0)
        ε_t ~ N(0, 0.01²)  [σ kept small so empirical peak stays at 12:00]

    Peak derivation: sin = +1 when 2π(t−6)/24 = π/2  →  t = 6+6 = 12  ✓
    Trough:          sin = −1 when 2π(t−6)/24 = −π/2 →  t = 6−6 = 0   ✓

    Source: Radovanovic et al. (2023) Fig. 2 – inflexible load peaks at noon;
            Yang et al. SoCC 2022 – LC services diurnal pattern.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(24, dtype=float)
    shape = 0.5 + 0.5 * np.sin(2.0 * np.pi * (t - 6.0) / 24.0)
    noise = rng.normal(0.0, 0.01, size=24)  # small σ keeps empirical peak at 12:00
    return np.clip(shape + noise, 0.30, 1.0)


def _dag_pipeline_24h(seed: int) -> np.ndarray:
    """Class 2 – DAG-constrained pipeline  (W = 4 h).

    Step-function profile with three discrete execution windows driven by
    business-cycle data availability:

        06–09 h  Morning ETL:  overnight data ready at business open
        12–15 h  Midday:       mid-morning data accumulation refresh
        19–21 h  Evening:      end-of-business consolidation sweep
        else     Base ≈ 0.35   warm-standby idle between DAG stages

    Source: Lechowicz et al. PCAPS (2025) – DAG stage completions produce
            discrete power pulses; Alibaba cluster-trace-v2018 – batch DAG
            jobs exhibit discrete execution phases.
    """
    rng = np.random.default_rng(seed)
    shape = np.full(24, 0.35)
    shape[6:10]  = 1.55   # morning ETL burst
    shape[12:16] = 1.45   # midday ingestion burst
    shape[19:22] = 1.15   # evening sweep burst
    noise = rng.normal(0.0, 0.05, size=24)
    return np.clip(shape + noise, 0.10, None)


def _batch_ml_24h(seed: int) -> np.ndarray:
    """Class 3 – Delay-tolerant batch / ML training  (W = 12–24 h).

    Mild sinusoidal profile peaking at ~11:00 (morning job submission surge)
    and troughing at ~23:00.  Low amplitude (0.22) reflects queue-smoothing:
    arrival spikes are absorbed by the scheduler before reaching execution.

    Equation:
        P(t) = 0.75 + 0.22·sin(2π(t−5)/24)
        Peak: t = 5+6 = 11  ✓   Trough: t = 5−6+24 = 23  ✓

    Source: Radovanovic et al. (2023) – W = 24 h Google fleet policy;
            Hu et al. SC'21 – Helios trace submission peak before noon;
            Majumder et al. (2026) – batch activity declines from midday.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(24, dtype=float)
    shape = 0.75 + 0.22 * np.sin(2.0 * np.pi * (t - 5.0) / 24.0)
    noise = rng.normal(0.0, 0.01, size=24)  # small σ keeps empirical peak at 11:00
    return np.clip(shape + noise, 0.30, None)


def _best_effort_24h(seed: int) -> np.ndarray:
    """Class 4 – Best-effort / background  (W = 24 h – ∞).

    Exact inverse of Class 1: peaks at midnight (00:00), troughs at noon.
    Reflects the VCC mechanism of Radovanovic et al. (2023) Fig. 2: the VCC
    is maximally permissive at midnight (interactive trough, low carbon) and
    minimally permissive at noon (interactive peak, high carbon).

    Equation:
        P(t) = 0.85 − 0.30·sin(2π(t−6)/24)
        Midnight peak: sin = −1 at t = 0  →  P = 1.15  ✓
        Noon trough:   sin = +1 at t = 12 →  P = 0.55  ✓

    Source: Radovanovic et al. (2023) Fig. 2; Wiesner et al. (2021) –
            nightly backups moved to overnight low-carbon window.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(24, dtype=float)
    shape = 0.85 - 0.30 * np.sin(2.0 * np.pi * (t - 6.0) / 24.0)
    noise = rng.normal(0.0, 0.025, size=24)
    return np.clip(shape + noise, 0.20, None)


# ─── Dispatch tables keyed by flexibility_hours ──────────────────────────────
_PROFILE_FN: dict[int, object] = {
    0:  _interactive_24h,   # Class 1 – W=0 h  (Sukprasert et al. EuroSys 2024)
    2:  _dag_pipeline_24h,  # backward compat only
    4:  _dag_pipeline_24h,  # Class 2 – W=4 h  (Acun et al. Carbon Explorer 2023)
    6:  _batch_ml_24h,      # Class 3 – W=6 h  (Radovanovic et al. IEEE TPS 2023)
    12: _batch_ml_24h,      # Class 3 – W=12 h (Radovanovic et al. IEEE TPS 2023)
    24: _best_effort_24h,   # Class 4 – W=24 h (Wiesner et al. ACM Middleware 2021)
}
_DOW_VECTORS: dict[int, np.ndarray] = {
    0:  np.array([1.04, 1.05, 1.05, 1.04, 1.02, 0.88, 0.85]), # strong weekday/weekend
    2:  np.array([1.03, 1.04, 1.04, 1.03, 1.02, 0.92, 0.87]), # backward compat
    4:  np.array([1.03, 1.04, 1.04, 1.03, 1.02, 0.92, 0.87]), # business-driven (W=4)
    6:  np.array([1.02, 1.03, 1.03, 1.02, 1.01, 0.96, 0.94]), # mild differential (W=6)
    12: np.array([1.02, 1.03, 1.03, 1.02, 1.01, 0.96, 0.94]), # mild differential (W=12)
    24: np.array([1.01, 1.01, 1.01, 1.01, 1.01, 0.99, 0.97]), # near-flat (W=24)
}


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helpers kept for backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

def _diurnal_24h(seed: int) -> np.ndarray:
    """Legacy single-day diurnal shape (mean=1.0).  Used by _week_profile."""
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    shape = 1.0 + 0.30 * np.sin(2 * np.pi * (hours - 9) / 24)
    noise = rng.normal(0.0, 0.03, size=24)
    return np.clip(shape + noise, 0.1, None)


def _week_profile(seed: int) -> np.ndarray:
    """Legacy 168h profile used as fallback.  Mean = 1.0."""
    rng = np.random.default_rng(seed)
    base = _diurnal_24h(seed)
    dow = np.array([1.02, 1.03, 1.03, 1.02, 1.00, 0.94, 0.93])
    dow_noise = rng.normal(0.0, 0.02, size=7)
    factors = dow + dow_noise
    week = np.concatenate([base * factors[d] for d in range(7)])
    return week / week.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: class-aware 168h profile
# ─────────────────────────────────────────────────────────────────────────────

def _week_profile_for_task(task: dict, seed: int) -> np.ndarray:
    """Build a 168h normalised IT load shape for one task.

    Dispatches to the class-appropriate 24h generator based on
    flexibility_hours, applies class-specific day-of-week scaling, then
    normalises so that the weekly mean equals exactly 1.0.

    Falls back to _interactive_24h for any flexibility_hours value not in
    the dispatch table, preserving backward compatibility.
    """
    rng = np.random.default_rng(seed)
    w = int(task["flexibility_hours"])

    profile_fn = _PROFILE_FN.get(w, _interactive_24h)
    dow_base   = _DOW_VECTORS.get(w, _DOW_VECTORS[0])

    base_24 = profile_fn(seed)                              # (24,)
    dow_noise   = rng.normal(0.0, 0.015, size=7)
    dow_factors = np.clip(dow_base + dow_noise, 0.50, 1.50)

    week = np.concatenate([base_24 * dow_factors[d] for d in range(7)])  # (168,)
    m = week.mean()
    return week / m if m > 1e-9 else week


# ─────────────────────────────────────────────────────────────────────────────
# Public API  ←  signatures are IDENTICAL to the original workload.py
# ─────────────────────────────────────────────────────────────────────────────

def generate_demand_profile(config: dict) -> pd.DataFrame:
    """Return a DataFrame indexed by hour (0..167) with one column per task name.

    Each column's mean over the week equals share_of_demand * total_capacity_mw.
    Seasonal variation is NOT applied here — demand is season-independent IT load.
    Seasonal differences (LMP, MOER) are handled entirely by the optimizer.

    Profile shapes are class-specific, dispatched by flexibility_hours:

    ┌─────────────────────┬──────┬─────────────────────────────────────────┐
    │ Task (default YAML) │  W   │ IT load profile                         │
    ├─────────────────────┼──────┼─────────────────────────────────────────┤
    │ interactive_serving │  0 h │ Sinusoidal, peak NOON (Radovanovic 2023)│
    │ etl_pipeline        │  2 h │ Step-function, 3 discrete bursts/day    │
    │ batch_ml_training   │ 12 h │ Mild sinusoid, morning submission peak  │
    │ cold_backup         │ 24 h │ Inverse sinusoid, peak midnight         │
    └─────────────────────┴──────┴─────────────────────────────────────────┘
    """
    seed = int(config.get("seed", 42))
    total_capacity = float(config["total_capacity_mw"])

    data: dict[str, np.ndarray] = {}
    for task in config["tasks"]:
        share = float(task["share_of_demand"])
        per_task_mean = share * total_capacity
        task_seed = seed + hash(task["name"]) % 1000
        normalised_shape = _week_profile_for_task(task, task_seed)
        data[task["name"]] = normalised_shape * per_task_mean

    df = pd.DataFrame(data)
    df.index.name = "hour"
    return df


def write_demand_csv(demand_df: pd.DataFrame, out_path: str | Path) -> None:
    demand_df.to_csv(out_path)


def baseline_fractions(flexibility_hours: int) -> np.ndarray:
    """Flat schedule: spread demand evenly over [0, W] shift offsets."""
    w = int(flexibility_hours)
    return np.full(w + 1, 1.0 / (w + 1))


def baseline_power_matrix(
    demand_df: pd.DataFrame,
    config: dict,
) -> dict[str, np.ndarray]:
    """Baseline = tasks run at their release hour (no spreading).

    This is the realistic "no optimization" counterfactual: each task's
    power at hour t is simply its released demand at hour t.
    """
    return {
        task["name"]: demand_df[task["name"]].to_numpy(dtype=float)
        for task in config["tasks"]
    }
