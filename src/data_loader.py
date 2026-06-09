from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yaml


CAISO_NODE = "TH_NP15_GEN-APND"

# 30-day averaging windows per season (used for LMP fetch).
SEASON_RANGES = {
    "summer": (date(2024, 7, 1), date(2024, 7, 31)),
    "winter": (date(2024, 1, 1), date(2024, 1, 31)),
    "shoulder": (date(2024, 4, 1), date(2024, 4, 30)),
}

# Cached WattTime MOER diurnal profiles (lb CO2 / MWh), hour 1..24.
# TODO: replace with a 30-day seasonal average once WattTime credentials are
# available. Structure mirrors the LMP 30-day-average schema.
# Summer: deep midday solar trough, steep evening ramp.
# Winter: flatter with gas-heavy evening peak.
# Shoulder: moderate midday dip.
CACHED_MOER = {
    "summer": [
        900.5, 895.2, 890.1, 885.3, 882.0, 879.8, 882.5, 890.3,
        910.2, 950.4, 1010.5, 1054.72, 1040.3, 1020.1, 998.5, 975.3,
        955.2, 940.1, 925.4, 910.8, 870.16, 878.3, 890.2, 898.5,
    ],
    "winter": [
        945.2, 942.1, 938.5, 935.2, 920.47, 922.3, 928.4, 935.6,
        942.3, 948.7, 955.2, 960.4, 965.1, 987.54, 982.3, 975.4,
        968.2, 960.1, 955.3, 950.4, 945.2, 940.1, 935.8, 930.5,
    ],
    "shoulder": [
        980.2, 960.4, 940.1, 920.5, 900.3, 880.1, 860.4, 620.3,
        200.5, 0.0, 10.2, 50.4, 80.1, 60.3, 40.5, 120.8,
        450.3, 780.5, 950.2, 990.4, 1010.5, 1030.2, 1050.8, 1069.24,
    ],
}

# Typical 24h CAISO LMP diurnal fallback profile ($/MWh), used if gridstatus
# is unavailable at runtime. Shapes approximate observed 2024 NP15 days.
FALLBACK_LMP = {
    "summer": [
        42, 38, 35, 33, 33, 36, 44, 55,
        48, 35, 25, 20, 18, 17, 22, 35,
        58, 95, 120, 110, 88, 70, 58, 48,
    ],
    "winter": [
        55, 52, 50, 48, 50, 58, 75, 90,
        82, 70, 62, 58, 55, 57, 62, 78,
        100, 110, 102, 92, 82, 72, 65, 60,
    ],
    "shoulder": [
        38, 35, 32, 30, 30, 33, 42, 52,
        42, 30, 22, 18, 16, 15, 18, 28,
        48, 72, 85, 80, 68, 55, 48, 42,
    ],
}


def load_task_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _fetch_caiso_lmp_day(date: str) -> list[float] | None:
    """Try to fetch a 24h CAISO day-ahead LMP series for the given date.

    Returns None if gridstatus is unavailable or the API call fails.
    """
    try:
        import gridstatus
    except ImportError:
        return None
    try:
        iso = gridstatus.CAISO()
        df = iso.get_lmp(
            date=date,
            market="DAY_AHEAD_HOURLY",
            locations=[CAISO_NODE],
        )
        df = df.sort_values("Time").reset_index(drop=True)
        lmp_col = "LMP" if "LMP" in df.columns else df.columns[-1]
        values = df[lmp_col].to_numpy(dtype=float)
        if len(values) < 24:
            return None
        return values[:24].tolist()
    except Exception:
        return None


def _daterange(start: date, end: date) -> list[str]:
    days = (end - start).days + 1
    return [(start + timedelta(days=i)).isoformat() for i in range(days)]


def _fetch_30day_avg(season: str) -> tuple[list[float], str]:
    """Fetch each day in the season window, average hourly values by hour-of-day.

    Returns (24h mean series, source tag). Falls back to FALLBACK_LMP if fewer
    than 7 days could be retrieved.
    """
    start, end = SEASON_RANGES[season]
    dates = _daterange(start, end)
    collected: list[list[float]] = []
    for d in dates:
        s = _fetch_caiso_lmp_day(d)
        if s is not None and len(s) >= 24:
            collected.append(s[:24])
    if len(collected) < 7:
        return FALLBACK_LMP[season], f"fallback_cached_only_{len(collected)}_days"
    avg = np.mean(np.array(collected, dtype=float), axis=0)
    print(f"  [{season}] averaged {len(collected)}/{len(dates)} days of CAISO LMP")
    return avg.tolist(), "caiso_api_30day_avg"


def build_lmp_csv(out_path: str | Path) -> pd.DataFrame:
    rows = []
    for season in SEASON_RANGES:
        values, source = _fetch_30day_avg(season)
        for hour, v in enumerate(values, start=1):
            rows.append({"season": season, "hour": hour, "value": float(v), "source": source})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def build_moer_csv(out_path: str | Path) -> pd.DataFrame:
    # Source tag notes that a 30-day WattTime average should replace this
    # hardcoded profile once API credentials are available.
    rows = []
    for season, values in CACHED_MOER.items():
        for hour, v in enumerate(values, start=1):
            rows.append({
                "season": season, "hour": hour, "value": float(v),
                "source": "watttime_cached_TODO_30day_avg",
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def ensure_grid_data(inputs_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs_dir = Path(inputs_dir)
    lmp_path = inputs_dir / "energy_cost.csv"
    moer_path = inputs_dir / "grid_intensity.csv"
    if not lmp_path.exists():
        build_lmp_csv(lmp_path)
    if not moer_path.exists():
        build_moer_csv(moer_path)
    return pd.read_csv(lmp_path), pd.read_csv(moer_path)


def season_24h(df: pd.DataFrame, season: str) -> np.ndarray:
    sub = df[df["season"] == season].sort_values("hour")
    return sub["value"].to_numpy(dtype=float)
