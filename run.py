from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    ensure_grid_data,
    load_task_config,
    season_24h,
)
from src.workload import (
    baseline_power_matrix,
    generate_demand_profile,
    write_demand_csv,
)
from src.optimizer import solve, tile_signal

from src.visualize import (
    plot_signals,
    plot_pareto,
    plot_workload_shift_multi,
    plot_total_load_multi,
    plot_cost_savings_by_scenario,
    plot_carbon_savings_by_scenario,
    plot_incremental_bess_savings,
    plot_bess_economic_breakdown,
    plot_battery_throughput,
    plot_net_incremental_bess_value,
    SEASON_ORDER,
)


# Sweep all report alpha scenarios.
ALPHAS = [round(x, 1) for x in np.arange(0.0, 1.0001, 0.1)]

# Optional delay penalty.
GAMMA = 0.1
CARBON_CONSTRAINT_ENABLED = True

# 0.0 means optimized carbon must be <= baseline carbon.
# Try 1.0 or 3.0 later if feasible.
CARBON_SAVINGS_TARGET_PCT = 0.0

# Battery size scenarios for comparison.
BATTERY_SIZE_SCENARIOS = [
    {"name": "battery_off", "enabled": False, "capacity_mwh": 0, "power_mw": 0},

    {"name": "bess_50mw_100mwh", "enabled": True, "capacity_mwh": 100, "power_mw": 50},
    {"name": "bess_100mw_200mwh", "enabled": True, "capacity_mwh": 200, "power_mw": 100},
    {"name": "bess_200mw_500mwh", "enabled": True, "capacity_mwh": 500, "power_mw": 200},
    {"name": "bess_200mw_800mwh", "enabled": True, "capacity_mwh": 800, "power_mw": 200},
]

# ─────────────────────────────────────────────────────────────────────────────
# Battery economics assumptions
# CAPEX uses NREL duration-based structure:
# CAPEX = energy_capacity_kWh × $/kWh + power_capacity_kW × $/kW
# Fixed O&M is assumed as 2.5% of CAPEX per year.
# ─────────────────────────────────────────────────────────────────────────────

BATTERY_ENERGY_COST_PER_KWH = 241.0   # $/kWh
BATTERY_POWER_COST_PER_KW = 372.0     # $/kW
BATTERY_FIXED_OM_FRAC = 0.025         # 2.5% of CAPEX per year
BATTERY_DISCOUNT_RATE = 0.07
BATTERY_LIFETIME_YEARS = 15


def capital_recovery_factor(discount_rate: float, lifetime_years: int) -> float:
    r = discount_rate
    n = lifetime_years
    return r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)


def battery_annualized_costs(
    capacity_mwh: float,
    power_mw: float,
) -> dict:
    """Calculate battery CAPEX, annualized CAPEX, fixed O&M, and weekly ownership cost."""
    if capacity_mwh <= 0 or power_mw <= 0:
        return {
            "battery_capex_usd": 0.0,
            "battery_annualized_capex_usd": 0.0,
            "battery_fixed_om_usd_per_year": 0.0,
            "battery_annual_ownership_cost_usd": 0.0,
            "battery_weekly_ownership_cost_usd": 0.0,
        }

    capacity_kwh = capacity_mwh * 1000.0
    power_kw = power_mw * 1000.0

    capex = (
        capacity_kwh * BATTERY_ENERGY_COST_PER_KWH
        + power_kw * BATTERY_POWER_COST_PER_KW
    )

    crf = capital_recovery_factor(BATTERY_DISCOUNT_RATE, BATTERY_LIFETIME_YEARS)
    annualized_capex = capex * crf
    fixed_om = BATTERY_FIXED_OM_FRAC * capex

    annual_ownership_cost = annualized_capex + fixed_om
    weekly_ownership_cost = annual_ownership_cost / 52.0

    return {
        "battery_capex_usd": capex,
        "battery_annualized_capex_usd": annualized_capex,
        "battery_fixed_om_usd_per_year": fixed_om,
        "battery_annual_ownership_cost_usd": annual_ownership_cost,
        "battery_weekly_ownership_cost_usd": weekly_ownership_cost,
    }

def normalized_totals(
    power_inner: np.ndarray,
    lmp_norm_inner: np.ndarray,
    moer_norm_inner: np.ndarray,
) -> tuple[float, float]:
    cost_norm = float(np.sum(lmp_norm_inner * power_inner))
    carbon_norm = float(np.sum(moer_norm_inner * power_inner))
    return cost_norm, carbon_norm


def compute_kpis(
    scenario: str,
    season: str,
    alpha: float,
    baseline_total: np.ndarray,
    optimized_total: np.ndarray,
    lmp_inner: np.ndarray,
    moer_inner: np.ndarray,
    res,
    config: dict,
) -> dict:
    cost_base_usd = float(np.sum(baseline_total * lmp_inner))
    cost_opt_usd = float(np.sum(optimized_total * lmp_inner))
    carbon_base_lbs = float(np.sum(baseline_total * moer_inner))
    carbon_opt_lbs = float(np.sum(optimized_total * moer_inner))

    cost_sav_usd = cost_base_usd - cost_opt_usd
    carbon_sav_lbs = carbon_base_lbs - carbon_opt_lbs

    if res.battery_charge is not None and res.battery_discharge is not None:
        battery_throughput_mwh = float(np.sum(res.battery_charge + res.battery_discharge))
    else:
        battery_throughput_mwh = 0.0

    kappa = float(config.get("battery", {}).get("degradation_cost", 0.0))
    battery_degradation_cost_usd = kappa * battery_throughput_mwh
    battery_enabled = bool(config.get("battery", {}).get("enabled", False))

    battery_capacity_mwh = (
        float(config.get("battery", {}).get("capacity_mwh", 0.0))
        if battery_enabled
        else 0.0
    )

    battery_power_mw = (
        float(config.get("battery", {}).get("max_discharge_mw", 0.0))
        if battery_enabled
        else 0.0
    )

    battery_costs = battery_annualized_costs(
        capacity_mwh=battery_capacity_mwh,
        power_mw=battery_power_mw,
    )
    
    return {
        "scenario": scenario,
        "season": season,
        "alpha": alpha,

        "cost_baseline_usd": cost_base_usd,
        "cost_optimized_usd": cost_opt_usd,
        "cost_savings_usd": cost_sav_usd,
        "cost_savings_pct": 100.0 * cost_sav_usd / cost_base_usd if cost_base_usd else 0.0,
        "cost_savings_annualized_usd": cost_sav_usd * 52,

        "carbon_baseline_lbs": carbon_base_lbs,
        "carbon_optimized_lbs": carbon_opt_lbs,
        "carbon_savings_lbs": carbon_sav_lbs,
        "carbon_savings_pct": 100.0 * carbon_sav_lbs / carbon_base_lbs if carbon_base_lbs else 0.0,
        "carbon_savings_tons": carbon_sav_lbs / 2000.0,
        "carbon_savings_annualized_tons": (carbon_sav_lbs / 2000.0) * 52,

        # Battery size and economics
        "battery_enabled": battery_enabled,
        "battery_capacity_mwh": battery_capacity_mwh,
        "battery_power_mw": battery_power_mw,
        "battery_duration_hours": battery_capacity_mwh / battery_power_mw if battery_power_mw > 0 else 0.0,

        "battery_capex_usd": battery_costs["battery_capex_usd"],
        "battery_annualized_capex_usd": battery_costs["battery_annualized_capex_usd"],
        "battery_fixed_om_usd_per_year": battery_costs["battery_fixed_om_usd_per_year"],
        "battery_annual_ownership_cost_usd": battery_costs["battery_annual_ownership_cost_usd"],
        "battery_weekly_ownership_cost_usd": battery_costs["battery_weekly_ownership_cost_usd"],

        # Battery operation
        "battery_throughput_mwh": battery_throughput_mwh,
        "battery_degradation_cost_usd": battery_degradation_cost_usd,    }


def main() -> None:
    inputs_dir = ROOT / "inputs"
    outputs_dir = ROOT / "outputs"
    plots_dir = outputs_dir / "plots"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_task_config(inputs_dir / "task_config.yaml")
    lmp_df, moer_df = ensure_grid_data(inputs_dir)

    # Demand is the same across battery_on and battery_off scenarios.
    demand_df = generate_demand_profile(base_config)
    write_demand_csv(demand_df, inputs_dir / "demand_profile.csv")

    baseline_by_task = baseline_power_matrix(demand_df, base_config)
    baseline_total = np.sum(list(baseline_by_task.values()), axis=0)

    w_max = max(int(t["flexibility_hours"]) for t in base_config["tasks"])
    h_pad = 168 + 2 * w_max

    all_pareto_rows = []
    all_kpi_rows = []
    all_schedule_rows = []
    
    for bess in BATTERY_SIZE_SCENARIOS:
        scenario_name = bess["name"]

        config = copy.deepcopy(base_config)
        config.setdefault("battery", {})
        config["battery"]["enabled"] = bess["enabled"]

        if bess["enabled"]:
            config["battery"]["capacity_mwh"] = bess["capacity_mwh"]
            config["battery"]["max_charge_mw"] = bess["power_mw"]
            config["battery"]["max_discharge_mw"] = bess["power_mw"]

        print("\n" + "=" * 80)
        print(f"RUNNING SCENARIO: {scenario_name}")
        print("=" * 80)

        for season in SEASON_ORDER:
            lmp_24 = season_24h(lmp_df, season)
            moer_24 = season_24h(moer_df, season)

            lmp_pad = tile_signal(lmp_24, h_pad, start_offset=-w_max)
            moer_pad = tile_signal(moer_24, h_pad, start_offset=-w_max)

            lmp_inner = lmp_pad[w_max:w_max + 168]
            moer_inner = moer_pad[w_max:w_max + 168]

            lmp_norm_inner = lmp_inner / max(lmp_24.max(), 1e-9)
            moer_norm_inner = moer_inner / max(moer_24.max(), 1e-9)

            cost_norm_base, carbon_norm_base = normalized_totals(
                baseline_total,
                lmp_norm_inner,
                moer_norm_inner,
            )
            carbon_base_lbs = float(np.sum(baseline_total * moer_inner))

            carbon_cap_lbs = None
            if CARBON_CONSTRAINT_ENABLED:
                carbon_cap_lbs = carbon_base_lbs * (1.0 - CARBON_SAVINGS_TARGET_PCT / 100.0)

            print(
                f"\n[{scenario_name} | {season}] "
                f"baseline cost_norm={cost_norm_base:.2f} "
                f"carbon_norm={carbon_norm_base:.2f}"
            )

            for alpha in ALPHAS:
                res = solve(
                    demand_df=demand_df,
                    config=config,
                    lmp_24=lmp_24,
                    moer_24=moer_24,
                    alpha=alpha,
                    season=season,
                    gamma=GAMMA,
                    carbon_cap_lbs=carbon_cap_lbs,
                )
                if alpha == 0.5 and season == "summer":
                    diff = np.max(
                        np.abs(
                            res.power_by_task["interactive_serving"]
                            - baseline_by_task["interactive_serving"]
                        )
                    )
                    print(
                        f"[DEBUG] {scenario_name} {season} interactive_serving max difference = {diff:.6f} MW"
                    )

                cost_norm_opt, carbon_norm_opt = normalized_totals(
                    res.total_power,
                    lmp_norm_inner,
                    moer_norm_inner,
                )

                objective_base = alpha * cost_norm_base + (1.0 - alpha) * carbon_norm_base
                objective_opt = alpha * cost_norm_opt + (1.0 - alpha) * carbon_norm_opt

                all_pareto_rows.append({
                    "scenario": scenario_name,
                    "season": season,
                    "alpha": alpha,
                    "total_cost_norm_baseline": cost_norm_base,
                    "total_cost_norm_optimized": cost_norm_opt,
                    "total_carbon_norm_baseline": carbon_norm_base,
                    "total_carbon_norm_optimized": carbon_norm_opt,
                    "objective_norm_baseline": objective_base,
                    "objective_norm_optimized": objective_opt,
                    "objective_norm_savings": objective_base - objective_opt,
                    "solver_status": res.status,
                })

                kpi = compute_kpis(
                    scenario=scenario_name,
                    season=season,
                    alpha=alpha,
                    baseline_total=baseline_total,
                    optimized_total=res.total_power,
                    lmp_inner=lmp_inner,
                    moer_inner=moer_inner,
                    res=res,
                    config=config,
                )
                all_kpi_rows.append(kpi)

                print(
                    f"  α={alpha:.1f} status={res.status:10s} "
                    f"cost_saved={kpi['cost_savings_pct']:7.2f}% "
                    f"carbon_saved={kpi['carbon_savings_pct']:7.2f}% "
                    f"battery_throughput={kpi['battery_throughput_mwh']:8.1f} MWh"
                )

                # Save hourly schedule for every alpha and scenario.
                for t in range(168):
                    for task in config["tasks"]:
                        name = task["name"]
                        p_base = baseline_by_task[name][t]
                        p_opt = res.power_by_task[name][t]

                        all_schedule_rows.append({
                            "scenario": scenario_name,
                            "season": season,
                            "alpha": alpha,
                            "hour": t,
                            "task_name": name,
                            "power_mw_baseline": p_base,
                            "power_mw_optimized": p_opt,
                            "total_power_mw_optimized": res.total_power[t],
                            "battery_charge_mw": res.battery_charge[t] if res.battery_charge is not None else 0.0,
                            "battery_discharge_mw": res.battery_discharge[t] if res.battery_discharge is not None else 0.0,
                            "battery_soc_mwh": res.battery_soc[t] if res.battery_soc is not None else 0.0,
                            "lmp": lmp_inner[t],
                            "moer": moer_inner[t],
                            "cost_baseline": p_base * lmp_inner[t],
                            "cost_optimized": p_opt * lmp_inner[t],
                            "carbon_baseline": p_base * moer_inner[t],
                            "carbon_optimized": p_opt * moer_inner[t],
                        })

    pareto_df = pd.DataFrame(all_pareto_rows)
    kpi_df = pd.DataFrame(all_kpi_rows)
    schedule_df = pd.DataFrame(all_schedule_rows)

    # Build incremental BESS economic columns relative to the battery_off case.
    if "incremental_bess_savings_usd" not in kpi_df.columns:
        battery_off_ref = (
            kpi_df[kpi_df["scenario"] == "battery_off"][
                ["season", "alpha", "cost_savings_usd", "carbon_savings_tons"]
            ]
            .rename(
                columns={
                    "cost_savings_usd": "cost_savings_usd_battery_off",
                    "carbon_savings_tons": "carbon_savings_tons_battery_off",
                }
            )
        )

        kpi_df = kpi_df.merge(
            battery_off_ref,
            on=["season", "alpha"],
            how="left",
        )

        kpi_df["incremental_bess_savings_usd"] = (
            kpi_df["cost_savings_usd"] - kpi_df["cost_savings_usd_battery_off"]
        )

        kpi_df["incremental_bess_carbon_tons"] = (
            kpi_df["carbon_savings_tons"] - kpi_df["carbon_savings_tons_battery_off"]
        )

    kpi_df["net_incremental_bess_value_usd"] = (
        kpi_df["incremental_bess_savings_usd"]
        - kpi_df["battery_weekly_ownership_cost_usd"]
        - kpi_df["battery_degradation_cost_usd"]
    )

    kpi_df["net_incremental_bess_value_annual_usd"] = (
        kpi_df["net_incremental_bess_value_usd"] * 52.0
    )
    
    kpi_df["bess_should_be_included"] = np.where(
        (kpi_df["scenario"] != "battery_off")
        & (kpi_df["net_incremental_bess_value_usd"] > 0),
        "YES",
        np.where(kpi_df["scenario"] == "battery_off", "BASELINE", "NO"),
    )
    
    # Debug check: interactive_serving should not move because flexibility_hours = 0
    debug_df = schedule_df[
        (schedule_df["alpha"] == 0.5)
        & (schedule_df["season"] == "summer")
        & (schedule_df["task_name"] == "interactive_serving")
    ].copy()

    debug_df["interactive_diff"] = (
        debug_df["power_mw_optimized"] - debug_df["power_mw_baseline"]
    ).abs()

    interactive_debug = debug_df.groupby("scenario")["interactive_diff"].max()

    print("\n" + "=" * 80)
    print("INTERACTIVE SERVING SHIFT CHECK")
    print("=" * 80)
    print(interactive_debug.to_string())
    print("If all values are 0.000000 or extremely close to 0, interactive_serving is NOT shifting.")
    print("=" * 80 + "\n")
    interactive_debug_path = outputs_dir / "interactive_serving_shift_check.csv"
    interactive_debug.reset_index().to_csv(interactive_debug_path, index=False)
    print(f"Wrote: {interactive_debug_path}")
    
    pareto_path = outputs_dir / "pareto_frontier_battery_comparison.csv"
    kpi_path = outputs_dir / "kpi_summary_battery_comparison.csv"
    schedule_path = outputs_dir / "schedule_battery_comparison.csv"
    pareto_df.to_csv(pareto_path, index=False)
    kpi_df.to_csv(kpi_path, index=False)
    schedule_df.to_csv(schedule_path, index=False)

    # ---------------------------------------------------------------------
    # REPORT PLOTS
    # ---------------------------------------------------------------------
    plot_workload_shift_multi(
        schedule_df=schedule_df,
        config=base_config,
        out_path=plots_dir / "workload_shift_multi_48h.png",
        alpha=0.5,
        window=(72, 120),
    )

    plot_total_load_multi(
        schedule_df=schedule_df,
        out_path=plots_dir / "total_load_multi_48h.png",
        alpha=0.5,
        window=(72, 120),
    )

    plot_cost_savings_by_scenario(
        kpi_df=kpi_df,
        out_path=plots_dir / "cost_savings_by_scenario_alpha_05.png",
        alpha=0.5,
    )

    plot_carbon_savings_by_scenario(
        kpi_df=kpi_df,
        out_path=plots_dir / "carbon_savings_by_scenario_alpha_05.png",
        alpha=0.5,
    )

    plot_incremental_bess_savings(
        kpi_df=kpi_df,
        out_path=plots_dir / "incremental_bess_savings_alpha_05.png",
        alpha=0.5,
    )

    plot_bess_economic_breakdown(
        kpi_df=kpi_df,
        out_path=plots_dir / "bess_economic_breakdown_summer_alpha_05.png",
        alpha=0.5,
        season="summer",
    )

    plot_battery_throughput(
        kpi_df=kpi_df,
        out_path=plots_dir / "battery_throughput_alpha_05.png",
        alpha=0.5,
    )

    plot_net_incremental_bess_value(
        kpi_df=kpi_df,
        out_path=plots_dir / "net_incremental_bess_value_alpha_05.png",
        alpha=0.5,
    )

    print(f"Wrote: {plots_dir / 'workload_shift_multi_48h.png'}")
    print(f"Wrote: {plots_dir / 'total_load_multi_48h.png'}")
    print(f"Wrote: {plots_dir / 'cost_savings_by_scenario_alpha_05.png'}")
    print(f"Wrote: {plots_dir / 'carbon_savings_by_scenario_alpha_05.png'}")
    print(f"Wrote: {plots_dir / 'incremental_bess_savings_alpha_05.png'}")
    print(f"Wrote: {plots_dir / 'bess_economic_breakdown_summer_alpha_05.png'}")
    print(f"Wrote: {plots_dir / 'battery_throughput_alpha_05.png'}")
    print(f"Wrote: {plots_dir / 'net_incremental_bess_value_alpha_05.png'}")

        
    print("\n" + "=" * 80)
    print("SUMMARY: α SCENARIOS WITH BATTERY ON/OFF")
    print("=" * 80)

    summary_cols = [
    "scenario",
    "season",
    "alpha",
    "battery_power_mw",
    "battery_capacity_mwh",
    "battery_duration_hours",
    "cost_savings_pct",
    "cost_savings_usd",
    "carbon_savings_pct",
    "carbon_savings_tons",
    "battery_throughput_mwh",
    "battery_degradation_cost_usd",
    "battery_capex_usd",
    "battery_weekly_ownership_cost_usd",
    "incremental_bess_savings_usd",
    "net_incremental_bess_value_usd",
    "bess_should_be_included",
    ]


    print(kpi_df[summary_cols].to_string(index=False))

    # Helpful report tables: alpha = 0, 0.5, 1 only.
    report_alpha_df = kpi_df[kpi_df["alpha"].isin([0.0, 0.5, 1.0])].copy()
    report_alpha_path = outputs_dir / "report_alpha_summary_battery_comparison.csv"
    report_alpha_df.to_csv(report_alpha_path, index=False)
    
    bess_decision_df = kpi_df[
        (kpi_df["alpha"] == 0.5)
        & (kpi_df["scenario"] != "battery_off")
    ].copy()

    bess_decision_cols = [
        "scenario",
        "season",
        "battery_power_mw",
        "battery_capacity_mwh",
        "battery_duration_hours",
        "cost_savings_usd",
        "cost_savings_usd_battery_off",
        "incremental_bess_savings_usd",
        "battery_weekly_ownership_cost_usd",
        "battery_degradation_cost_usd",
        "net_incremental_bess_value_usd",
        "net_incremental_bess_value_annual_usd",
        "carbon_savings_tons",
        "incremental_bess_carbon_tons",
        "bess_should_be_included",
    ]

    bess_decision_path = outputs_dir / "bess_economic_decision_alpha_05.csv"
    bess_decision_df[bess_decision_cols].to_csv(bess_decision_path, index=False)

    print("\n" + "=" * 80)
    print("REPORT ALPHA SUMMARY: α = 0.0, 0.5, 1.0")
    print("=" * 80)
    print(report_alpha_df[summary_cols].to_string(index=False))

    # Pretty KPI summaries for α = 0.5, matching the old report format.
    print("\n" + "=" * 80)
    print("KPI SUMMARY FORMAT FOR REPORT: α = 0.5")
    print("=" * 80)

    alpha_05_df = kpi_df[kpi_df["alpha"] == 0.5].copy()

    for bess in BATTERY_SIZE_SCENARIOS:
        scenario_name = bess["name"]
        print(f"\n=== KPI Summary ({scenario_name}, α=0.5, one week) ===")

        sub = alpha_05_df[alpha_05_df["scenario"] == scenario_name]

        for season in SEASON_ORDER:
            row = sub[sub["season"] == season].iloc[0]

            print(
                f"{season.capitalize():9s} "
                f"cost saved {row['cost_savings_pct']:5.2f}%  "
                f"(${row['cost_savings_usd']:,.0f}/week, "
                f"${row['cost_savings_annualized_usd']:,.0f}/yr)  |  "
                f"carbon avoided {row['carbon_savings_pct']:5.2f}%  "
                f"({row['carbon_savings_tons']:,.1f} tons/week, "
                f"{row['carbon_savings_annualized_tons']:,.1f} tons/yr)"
            )

        print(f"\n=== Battery Economics ({scenario_name}, α=0.5, one week) ===")

        for season in SEASON_ORDER:
            row = sub[sub["season"] == season].iloc[0]

            if scenario_name == "battery_off":
                print(
                    f"{season.capitalize():9s} "
                    f"throughput 0.0 MWh/week  |  "
                    f"incremental BESS savings $0/week  |  "
                    f"net BESS value $0/week  |  "
                    f"decision BASELINE"
                )
            else:
                print(
                    f"{season.capitalize():9s} "
                    f"throughput {row['battery_throughput_mwh']:,.1f} MWh/week  |  "
                    f"degradation ${row['battery_degradation_cost_usd']:,.0f}/week  |  "
                    f"ownership ${row['battery_weekly_ownership_cost_usd']:,.0f}/week  |  "
                    f"incremental savings ${row['incremental_bess_savings_usd']:,.0f}/week  |  "
                    f"net value ${row['net_incremental_bess_value_usd']:,.0f}/week  |  "
                    f"decision {row['bess_should_be_included']}"
                )
            
    # Existing signal and pareto plot.
    plot_signals(lmp_df, moer_df, plots_dir / "signals.png")

    # Existing plot_pareto does not distinguish battery_on/off, so save
    # one separate pareto plot for each scenario.
    for bess in BATTERY_SIZE_SCENARIOS:
        scenario_name = bess["name"]
        sub = pareto_df[pareto_df["scenario"] == scenario_name]
        plot_pareto(sub, plots_dir / f"pareto_frontier_{scenario_name}.png")
    
    print(f"\nWrote: {pareto_path}")
    print(f"Wrote: {kpi_path}")
    print(f"Wrote: {schedule_path}")
    print(f"Wrote: {report_alpha_path}")
    print(f"Wrote: {bess_decision_path}")
    print(f"Wrote plots to: {plots_dir}")


if __name__ == "__main__":
    main()
    