from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEASON_ORDER = ["summer", "winter", "shoulder"]

# ---------------------------------------------------------------------
# Scenario order / labels for report plots
# ---------------------------------------------------------------------
SCENARIO_ORDER_ALL = [
    "battery_off",
    "bess_50mw_100mwh",
    "bess_100mw_200mwh",
    "bess_200mw_500mwh",
    "bess_200mw_800mwh",
]

SCENARIO_ORDER_BESS = [
    "bess_50mw_100mwh",
    "bess_100mw_200mwh",
    "bess_200mw_500mwh",
    "bess_200mw_800mwh",
]

SCENARIO_LABELS = {
    "battery_off": "Battery OFF",
    "bess_50mw_100mwh": "BESS 50MW / 100MWh",
    "bess_100mw_200mwh": "BESS 100MW / 200MWh",
    "bess_200mw_500mwh": "BESS 200MW / 500MWh",
    "bess_200mw_800mwh": "BESS 200MW / 800MWh",
}

SEASON_COLORS = {
    "summer": "#e74c3c",
    "winter": "#2980b9",
    "shoulder": "#27ae60",
}


def _apply_48h_xaxis(ax, start: int) -> None:
    ax.set_xlim(start, start + 47)
    tick_positions = [start + k for k in (0, 6, 12, 18, 24, 30, 36, 42)]
    tick_labels = [f"{((start + k) % 24):02d}" for k in (0, 6, 12, 18, 24, 30, 36, 42)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.axvline(start + 24, color="black", linestyle="--", linewidth=1.0, alpha=0.5)


def _task_colors(config: dict) -> dict[str, str]:
    return {t["name"]: t["color"] for t in config["tasks"]}


# ---------------------------------------------------------------------
# 1) Multi-scenario stacked workload plot
# Rows:
#   Baseline
#   Battery OFF
#   BESS low
#   BESS mid
#   BESS high
# Columns:
#   Summer / Winter / Shoulder
# ---------------------------------------------------------------------
def plot_workload_shift_multi(
    schedule_df: pd.DataFrame,
    config: dict,
    out_path: str | Path,
    alpha: float = 0.5,
    window: tuple[int, int] = (72, 120),
) -> None:
    colors = _task_colors(config)

    # Put interactive at the bottom so it looks visually fixed.
    task_order = [
        "interactive_serving",
        "etl_pipeline",
        "batch_ml_training",
        "cold_backup",
    ]

    scenario_rows = [
        ("baseline", "Baseline\n(No optimization)"),
        ("battery_off", "Battery OFF\n(Workload shifting only)"),
        ("bess_50mw_100mwh", "Battery ON\n50 MW / 100 MWh"),
        ("bess_100mw_200mwh", "Battery ON\n100 MW / 200 MWh"),
        ("bess_200mw_500mwh", "Battery ON\n200 MW / 500 MWh"),
        ("bess_200mw_800mwh", "Battery ON\n200 MW / 800 MWh"),
    ]

    start, end = window
    hours = np.arange(start, end)

    df = schedule_df[schedule_df["alpha"] == alpha].copy()

    fig, axes = plt.subplots(
        nrows=len(scenario_rows),
        ncols=len(SEASON_ORDER),
        figsize=(18, 15),
        sharex=True,
        sharey=True,
    )

    # Get a common ymax across all panels
    ymax = 0.0
    for season in SEASON_ORDER:
        sub = df[(df["season"] == season) & (df["hour"].between(start, end - 1))]
        if sub.empty:
            continue

        # baseline
        base_total = (
            sub.groupby(["hour", "task_name"])["power_mw_baseline"]
            .first()
            .unstack(fill_value=0.0)
            .reindex(columns=task_order, fill_value=0.0)
            .sum(axis=1)
        )
        ymax = max(ymax, float(base_total.max()))

        # optimized scenarios
        for scenario_name, _ in scenario_rows[1:]:
            ssub = sub[sub["scenario"] == scenario_name]
            if ssub.empty:
                continue
            opt_total = (
                ssub.groupby(["hour", "task_name"])["power_mw_optimized"]
                .first()
                .unstack(fill_value=0.0)
                .reindex(columns=task_order, fill_value=0.0)
                .sum(axis=1)
            )
            ymax = max(ymax, float(opt_total.max()))

    ymax *= 1.05

    for col, season in enumerate(SEASON_ORDER):
        season_df = df[
            (df["season"] == season)
            & (df["hour"].between(start, end - 1))
        ].copy()

        for row, (scenario_name, row_label) in enumerate(scenario_rows):
            ax = axes[row, col]

            if scenario_name == "baseline":
                temp = (
                    season_df.groupby(["hour", "task_name"])["power_mw_baseline"]
                    .first()
                    .unstack(fill_value=0.0)
                    .reindex(columns=task_order, fill_value=0.0)
                )
            else:
                temp = (
                    season_df[season_df["scenario"] == scenario_name]
                    .groupby(["hour", "task_name"])["power_mw_optimized"]
                    .first()
                    .unstack(fill_value=0.0)
                    .reindex(columns=task_order, fill_value=0.0)
                )

            temp = temp.reindex(index=hours, fill_value=0.0)

            stacks = np.vstack([temp[t].to_numpy() for t in task_order])

            ax.stackplot(
                hours,
                stacks,
                colors=[colors[t] for t in task_order],
                labels=task_order,
                alpha=0.9,
            )

            if row == 0:
                ax.set_title(season.capitalize(), fontsize=12)

            if col == 0:
                ax.set_ylabel(f"{row_label}\nPower (MW)")

            _apply_48h_xaxis(ax, start)
            ax.set_ylim(0, ymax)
            ax.grid(True, alpha=0.25)

            if row == len(scenario_rows) - 1:
                ax.set_xlabel("Hour of day")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        f"48-hour stacked workload comparison across battery scenarios (α={alpha})",
        fontsize=14,
        y=0.985,
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),   # put legend just below title
        ncol=4,
        framealpha=0.95,
        fontsize=9,
    )

    fig.text(
        0.5,
        0.01,
        "Rows show optimization scenario; columns show season.",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])

    fig.text(
        0.5,
        0.01,
        "Rows show optimization scenario; columns show season.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# 2) Multi-scenario total load overlay
# ---------------------------------------------------------------------
def plot_total_load_multi(
    schedule_df: pd.DataFrame,
    out_path: str | Path,
    alpha: float = 0.5,
    window: tuple[int, int] = (72, 120),
) -> None:
    start, end = window
    hours = np.arange(start, end)

    df = schedule_df[
        (schedule_df["alpha"] == alpha)
        & (schedule_df["hour"].between(start, end - 1))
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for col, season in enumerate(SEASON_ORDER):
        ax = axes[col]
        season_df = df[df["season"] == season].copy()

        # baseline
        baseline_series = (
            season_df.groupby(["hour", "task_name"])["power_mw_baseline"]
            .first()
            .groupby("hour")
            .sum()
            .reindex(hours, fill_value=0.0)
        )
        ax.plot(
            hours,
            baseline_series.to_numpy(),
            label="Baseline",
            linewidth=2.5,
            linestyle="--",
        )

        # optimized scenarios
        for scenario_name in SCENARIO_ORDER_ALL:
            ssub = season_df[season_df["scenario"] == scenario_name]
            if ssub.empty:
                continue

            total_series = (
                ssub.groupby("hour")["total_power_mw_optimized"]
                .first()
                .reindex(hours, fill_value=0.0)
            )

            ax.plot(
                hours,
                total_series.to_numpy(),
                label=SCENARIO_LABELS[scenario_name],
                linewidth=2.0,
            )

        ax.set_title(season.capitalize())
        _apply_48h_xaxis(ax, start)
        ax.set_xlabel("Hour of day")
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Total load (MW)")

    axes[-1].legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle(f"48-hour total load comparison (α={alpha})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Shared grouped-bar helper
# ---------------------------------------------------------------------
def _grouped_bar_by_season(
    kpi_df: pd.DataFrame,
    out_path: str | Path,
    value_col: str,
    title: str,
    ylabel: str,
    alpha: float = 0.5,
    scenario_order: list[str] | None = None,
) -> None:
    if scenario_order is None:
        scenario_order = SCENARIO_ORDER_ALL

    df = kpi_df[kpi_df["alpha"] == alpha].copy()

    x = np.arange(len(scenario_order))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, season in enumerate(SEASON_ORDER):
        vals = []
        for scenario_name in scenario_order:
            row = df[(df["season"] == season) & (df["scenario"] == scenario_name)]
            if row.empty:
                vals.append(0.0)
            else:
                vals.append(float(row.iloc[0][value_col]))

        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=season.capitalize(),
            color=SEASON_COLORS[season],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenario_order], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# 3) Cost savings
# ---------------------------------------------------------------------
def plot_cost_savings_by_scenario(
    kpi_df: pd.DataFrame,
    out_path: str | Path,
    alpha: float = 0.5,
) -> None:
    _grouped_bar_by_season(
        kpi_df=kpi_df,
        out_path=out_path,
        value_col="cost_savings_usd",
        title=f"Weekly cost savings by scenario (α={alpha})",
        ylabel="Cost savings (USD/week)",
        alpha=alpha,
        scenario_order=SCENARIO_ORDER_ALL,
    )


# ---------------------------------------------------------------------
# 4) Carbon savings
# ---------------------------------------------------------------------
def plot_carbon_savings_by_scenario(
    kpi_df: pd.DataFrame,
    out_path: str | Path,
    alpha: float = 0.5,
) -> None:
    _grouped_bar_by_season(
        kpi_df=kpi_df,
        out_path=out_path,
        value_col="carbon_savings_tons",
        title=f"Weekly carbon savings by scenario (α={alpha})",
        ylabel="Carbon savings (tons/week)",
        alpha=alpha,
        scenario_order=SCENARIO_ORDER_ALL,
    )


# ---------------------------------------------------------------------
# 5) Incremental BESS savings
# ---------------------------------------------------------------------
def plot_incremental_bess_savings(
    kpi_df: pd.DataFrame,
    out_path: str | Path,
    alpha: float = 0.5,
) -> None:
    _grouped_bar_by_season(
        kpi_df=kpi_df,
        out_path=out_path,
        value_col="incremental_bess_savings_usd",
        title=f"Incremental BESS savings vs Battery OFF (α={alpha})",
        ylabel="Incremental savings (USD/week)",
        alpha=alpha,
        scenario_order=SCENARIO_ORDER_BESS,
    )


# ---------------------------------------------------------------------
# 6) BESS economic breakdown (summer only)
# ---------------------------------------------------------------------
def plot_bess_economic_breakdown(
    kpi_df: pd.DataFrame,
    out_path: str | Path,
    alpha: float = 0.5,
    season: str = "summer",
) -> None:
    df = kpi_df[
        (kpi_df["alpha"] == alpha)
        & (kpi_df["season"] == season)
        & (kpi_df["scenario"].isin(SCENARIO_ORDER_BESS))
    ].copy()

    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER_BESS, ordered=True)
    df = df.sort_values("scenario")

    x = np.arange(len(df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - 1.5 * width, df["incremental_bess_savings_usd"], width=width, label="Incremental savings")
    ax.bar(x - 0.5 * width, df["battery_weekly_ownership_cost_usd"], width=width, label="Ownership cost")
    ax.bar(x + 0.5 * width, df["battery_degradation_cost_usd"], width=width, label="Degradation cost")
    ax.bar(x + 1.5 * width, df["net_incremental_bess_value_usd"], width=width, label="Net BESS value")

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in df["scenario"]], rotation=20, ha="right")
    ax.set_ylabel("USD/week")
    ax.set_title(f"BESS economic breakdown ({season.capitalize()}, α={alpha})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# 7) Battery throughput
# ---------------------------------------------------------------------
def plot_battery_throughput(
    kpi_df: pd.DataFrame,
    out_path: str | Path,
    alpha: float = 0.5,
) -> None:
    _grouped_bar_by_season(
        kpi_df=kpi_df,
        out_path=out_path,
        value_col="battery_throughput_mwh",
        title=f"Battery throughput by scenario (α={alpha})",
        ylabel="Battery throughput (MWh/week)",
        alpha=alpha,
        scenario_order=SCENARIO_ORDER_BESS,
    )


# ---------------------------------------------------------------------
# 8) Net incremental BESS value
# ---------------------------------------------------------------------
def plot_net_incremental_bess_value(
    kpi_df: pd.DataFrame,
    out_path: str | Path,
    alpha: float = 0.5,
) -> None:
    _grouped_bar_by_season(
        kpi_df=kpi_df,
        out_path=out_path,
        value_col="net_incremental_bess_value_usd",
        title=f"Net incremental BESS value (α={alpha})",
        ylabel="Net value (USD/week)",
        alpha=alpha,
        scenario_order=SCENARIO_ORDER_BESS,
    )
    
def plot_signals(
    lmp_df: pd.DataFrame,
    moer_df: pd.DataFrame,
    out_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    season_colors = {"summer": "#e74c3c", "winter": "#2980b9", "shoulder": "#27ae60"}

    for season in SEASON_ORDER:
        sub = lmp_df[lmp_df["season"] == season].sort_values("hour")
        axes[0].plot(sub["hour"], sub["value"], label=season, color=season_colors[season], linewidth=2)

        sub = moer_df[moer_df["season"] == season].sort_values("hour")
        axes[1].plot(sub["hour"], sub["value"], label=season, color=season_colors[season], linewidth=2)

    axes[0].set_title("CAISO NP15 Day-Ahead LMP")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("LMP ($/MWh)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("CAISO Grid Carbon Intensity (MOER)")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("MOER (lb CO₂/MWh)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_pareto(
    pareto_df: pd.DataFrame,
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    season_colors = {"summer": "#e74c3c", "winter": "#2980b9", "shoulder": "#27ae60"}

    for season in SEASON_ORDER:
        sub = pareto_df[pareto_df["season"] == season].sort_values("alpha")
        if sub.empty:
            continue

        ax.plot(
            sub["total_cost_norm_optimized"],
            sub["total_carbon_norm_optimized"],
            marker="o",
            label=f"{season} optimized",
            color=season_colors[season],
            linewidth=2,
        )

        base_row = sub.iloc[0]
        ax.scatter(
            base_row["total_cost_norm_baseline"],
            base_row["total_carbon_norm_baseline"],
            marker="*",
            s=220,
            edgecolor="black",
            linewidth=1.0,
            color=season_colors[season],
            label=f"{season} baseline",
            zorder=5,
        )

    ax.set_xlabel("Normalized cost")
    ax.set_ylabel("Normalized carbon")
    ax.set_title("Cost-carbon Pareto frontier by season")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)