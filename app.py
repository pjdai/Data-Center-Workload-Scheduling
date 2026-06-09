from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.data_loader import ensure_grid_data, season_24h
from src.workload import generate_demand_profile, baseline_power_matrix
from src.optimizer import solve


ROOT = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(ROOT, "inputs")

TOTAL_CAPACITY_MW = 500.0
HORIZON_HOURS = 168
SEASONS = ["summer", "winter", "shoulder"]

DEFAULT_TASKS = [
    {"name": "interactive_serving", "flexibility_hours": 0,  "share_of_demand": 30.0, "max_power_multiplier": 2.0, "color": "#e74c3c"},
    {"name": "etl_pipeline",        "flexibility_hours": 2,  "share_of_demand": 20.0, "max_power_multiplier": 2.0, "color": "#f39c12"},
    {"name": "batch_ml_training",   "flexibility_hours": 12, "share_of_demand": 35.0, "max_power_multiplier": 2.0, "color": "#3498db"},
    {"name": "cold_backup",         "flexibility_hours": 24, "share_of_demand": 15.0, "max_power_multiplier": 2.0, "color": "#2ecc71"},
]


# ---------------- data + solve helpers ----------------


@st.cache_data(show_spinner=False)
def load_grid():
    lmp_df, moer_df = ensure_grid_data(INPUTS_DIR)
    return lmp_df, moer_df


def tasks_with_ids(tasks):
    out = []
    for t in tasks:
        t2 = dict(t)
        if "_tid" not in t2:
            t2["_tid"] = uuid.uuid4().hex[:8]
        out.append(t2)
    return out


def build_config(tasks_pct, peak_mult):
    return {
        "total_capacity_mw": TOTAL_CAPACITY_MW,
        "peak_multiplier": float(peak_mult),
        "seed": 42,
        "tasks": [
            {
                "name": t["name"],
                "flexibility_hours": int(t["flexibility_hours"]),
                "share_of_demand": float(t["share_of_demand"]) / 100.0,
                "max_power_multiplier": float(t["max_power_multiplier"]),
                "color": t.get("color", "#888888"),
            }
            for t in tasks_pct
        ],
    }


def validate(tasks):
    errors, warnings = [], []
    names = [t["name"].strip() for t in tasks]
    if any(n == "" for n in names):
        errors.append("Task names cannot be empty.")
    if len(names) != len(set(names)):
        errors.append("Task names must be unique.")
    total_share = sum(float(t["share_of_demand"]) for t in tasks)
    if total_share <= 0:
        errors.append("Task shares must sum to a positive value.")
    if tasks and all(int(t["flexibility_hours"]) == 0 for t in tasks):
        warnings.append("All tasks have W=0 (fully inflexible) — optimized schedule equals baseline.")
    return errors, warnings


def normalize_shares(tasks):
    total = sum(float(t["share_of_demand"]) for t in tasks)
    if total <= 0:
        return tasks, None
    original = [round(float(t["share_of_demand"]), 1) for t in tasks]
    if abs(total - 100.0) < 0.05:
        return tasks, None
    scaled = []
    for t in tasks:
        t2 = dict(t)
        t2["share_of_demand"] = float(t["share_of_demand"]) * 100.0 / total
        scaled.append(t2)
    adjusted = [round(float(t["share_of_demand"]), 1) for t in scaled]
    return scaled, (original, adjusted)


def compute_kpis(baseline_total, opt_total, lmp_inner, moer_inner):
    cost_base = float(np.sum(baseline_total * lmp_inner))
    cost_opt = float(np.sum(opt_total * lmp_inner))
    carbon_base_lbs = float(np.sum(baseline_total * moer_inner))
    carbon_opt_lbs = float(np.sum(opt_total * moer_inner))
    base_mean = float(np.mean(baseline_total)) if baseline_total.size else 0.0
    opt_mean = float(np.mean(opt_total)) if opt_total.size else 0.0
    return {
        "cost_base_usd": cost_base,
        "cost_opt_usd": cost_opt,
        "cost_sav_usd": cost_base - cost_opt,
        "cost_sav_pct": 100.0 * (cost_base - cost_opt) / cost_base if cost_base else 0.0,
        "carbon_base_lbs": carbon_base_lbs,
        "carbon_opt_lbs": carbon_opt_lbs,
        "carbon_sav_lbs": carbon_base_lbs - carbon_opt_lbs,
        "carbon_sav_pct": 100.0 * (carbon_base_lbs - carbon_opt_lbs) / carbon_base_lbs if carbon_base_lbs else 0.0,
        "carbon_sav_tons": (carbon_base_lbs - carbon_opt_lbs) / 2000.0,
        "cv_base": (float(np.std(baseline_total)) / base_mean) if base_mean else 0.0,
        "cv_opt": (float(np.std(opt_total)) / opt_mean) if opt_mean else 0.0,
    }


def run_all_seasons(config, alpha):
    lmp_df, moer_df = load_grid()
    demand_df = generate_demand_profile(config)
    baseline_by_task = baseline_power_matrix(demand_df, config)
    baseline_total = np.sum(list(baseline_by_task.values()), axis=0)

    results = {}
    for season in SEASONS:
        lmp_24 = season_24h(lmp_df, season)
        moer_24 = season_24h(moer_df, season)
        res = solve(demand_df, config, lmp_24, moer_24, alpha, season)
        kpis = compute_kpis(baseline_total, res.total_power, res.lmp_inner, res.moer_inner)
        results[season] = {
            "status": res.status,
            "power_by_task": res.power_by_task,
            "total_power": res.total_power,
            "baseline_by_task": baseline_by_task,
            "baseline_total": baseline_total,
            "lmp_24": lmp_24,
            "moer_24": moer_24,
            "lmp_inner": res.lmp_inner,
            "moer_inner": res.moer_inner,
            "kpis": kpis,
        }
    return results


# ---------------- plotting ----------------


def plot_grid_signals(season_data, season):
    lmp_24 = season_data["lmp_24"]
    moer_24 = season_data["moer_24"]
    hours = np.arange(24)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, signal, label, unit, fmt in [
        (axes[0], lmp_24, "LMP", "$/MWh", "${:.0f}"),
        (axes[1], moer_24, "MOER", "lb CO$_2$/MWh", "{:.0f}"),
    ]:
        mean = float(np.mean(signal))
        ax.plot(hours, signal, color="black", lw=1.6)
        ax.axhline(mean, ls="--", color="gray", lw=1.0)
        ax.fill_between(hours, signal, mean, where=signal >= mean, color="red", alpha=0.30, interpolate=True)
        ax.fill_between(hours, signal, mean, where=signal <  mean, color="blue", alpha=0.30, interpolate=True)
        peak_h = int(np.argmax(signal))
        min_h  = int(np.argmin(signal))
        ax.annotate(f"peak  h{peak_h:02d}\n{fmt.format(signal[peak_h])}",
                    xy=(peak_h, signal[peak_h]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
        ax.annotate(f"min  h{min_h:02d}\n{fmt.format(signal[min_h])}",
                    xy=(min_h, signal[min_h]), xytext=(5, -24),
                    textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="blue", lw=0.8))
        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(unit)
        ax.set_title(f"{label} – {season.capitalize()}")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _stacked_pair(ax_top, ax_bot, data, config_tasks, title):
    task_names = [t["name"] for t in config_tasks]
    colors = [t["color"] for t in config_tasks]
    hrs_slice = slice(72, 120)
    x = np.arange(48)

    base_stack = [data["baseline_by_task"][n][hrs_slice] for n in task_names]
    opt_stack  = [data["power_by_task"][n][hrs_slice]    for n in task_names]

    ax_top.stackplot(x, *base_stack, colors=colors, labels=task_names, alpha=0.95)
    ax_bot.stackplot(x, *opt_stack,  colors=colors, labels=task_names, alpha=0.95)

    ymax = max(np.sum(base_stack, axis=0).max(), np.sum(opt_stack, axis=0).max()) * 1.05
    for a, label in [(ax_top, "Baseline"), (ax_bot, "Optimized")]:
        a.set_ylim(0, ymax)
        a.axvline(24, ls="--", color="gray", lw=0.8)
        a.set_xticks([0, 6, 12, 18, 24, 30, 36, 42])
        a.set_xticklabels(["00", "06", "12", "18", "00", "06", "12", "18"])
        a.set_ylabel(f"{label}\n(MW)", fontsize=9)
        a.grid(alpha=0.2)
    ax_top.set_title(title, fontsize=10)
    ax_bot.set_xlabel("Hour of day")


def plot_demand_48h_main(results, config_tasks, season):
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
    _stacked_pair(axes[0], axes[1], results[season], config_tasks,
                  title=f"{season.capitalize()} — 48h stacked demand (hours 72–119)")
    axes[0].legend(loc="upper right", fontsize=7, ncol=min(4, len(config_tasks)))
    fig.tight_layout()
    return fig


def plot_demand_48h_compare(results, config_tasks, season):
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 2.4), sharex=True)
    _stacked_pair(axes[0], axes[1], results[season], config_tasks,
                  title=f"{season.capitalize()}")
    for a in axes:
        a.tick_params(labelsize=7)
        a.set_ylabel(a.get_ylabel(), fontsize=7)
    fig.tight_layout()
    return fig


def plot_load_variability(season_data, season):
    base = season_data["baseline_total"]
    opt  = season_data["total_power"]
    hours = np.arange(HORIZON_HOURS)
    mean_opt = float(np.mean(opt))
    mean_base = float(np.mean(base))
    cv_opt = season_data["kpis"]["cv_opt"]
    cv_base = season_data["kpis"]["cv_base"]

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(hours, base, color="gray", lw=1.3, label=f"Baseline (CV = {cv_base:.3f})")
    ax.plot(hours, opt,  color="black", lw=1.6, label=f"Optimized (CV = {cv_opt:.3f})")
    ax.axhline(mean_opt, ls="--", color="black", lw=0.9)
    ax.fill_between(hours, opt, mean_opt, where=opt >= mean_opt, color="red",  alpha=0.25, interpolate=True)
    ax.fill_between(hours, opt, mean_opt, where=opt <  mean_opt, color="blue", alpha=0.25, interpolate=True)

    for d in range(1, 7):
        ax.axvline(d * 24, ls=":", color="lightgray", lw=0.7)

    ax.text(0.99, 0.96,
            f"CV optimized = {cv_opt:.3f}\nCV baseline  = {cv_base:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
    ax.set_xlim(0, HORIZON_HOURS - 1)
    ax.set_xticks(np.arange(0, 168 + 1, 24))
    ax.set_xlabel("Hour of week")
    ax.set_ylabel("Total demand (MW)")
    ax.set_title(f"Load variability — optimized vs. baseline ({season.capitalize()})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------- session state ----------------


def init_state():
    if "tasks" not in st.session_state:
        st.session_state.tasks = tasks_with_ids([dict(t) for t in DEFAULT_TASKS])
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
    if "season" not in st.session_state:
        st.session_state.season = "summer"
    if "alpha" not in st.session_state:
        st.session_state.alpha = 0.5
    if "peak_mult" not in st.session_state:
        st.session_state.peak_mult = 1.30
    if "results" not in st.session_state:
        st.session_state.results = None
    if "normalize_info" not in st.session_state:
        st.session_state.normalize_info = None
    if "first_load_done" not in st.session_state:
        st.session_state.first_load_done = False
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "solved_tasks" not in st.session_state:
        st.session_state.solved_tasks = None


# ---------------- sidebar ----------------


def render_sidebar():
    with st.sidebar:
        st.title("⚡ Scheduler")

        st.session_state.edit_mode = st.toggle("✏️ Edit mode", value=st.session_state.edit_mode)
        edit = st.session_state.edit_mode

        st.subheader("Grid")
        if edit:
            season = st.selectbox("Season", SEASONS,
                                  index=SEASONS.index(st.session_state.season),
                                  format_func=str.capitalize)
            alpha = st.slider("α (cost ↔ carbon)", 0.0, 1.0, float(st.session_state.alpha), 0.05,
                              help="α=1 → minimize cost. α=0 → minimize carbon.")
            st.session_state.season = season
            st.session_state.alpha = alpha
        else:
            st.markdown(f"**Season:** {st.session_state.season.capitalize()}")
            st.markdown(f"**α (cost ↔ carbon):** `{st.session_state.alpha:.2f}`")

        st.subheader("Data center")
        if edit:
            peak_mult = st.slider("Peak multiplier", 1.0, 2.0, float(st.session_state.peak_mult), 0.05)
            st.session_state.peak_mult = peak_mult
        else:
            st.markdown(f"**Peak multiplier:** `{st.session_state.peak_mult:.2f}×`")

        st.subheader("Task classes")

        to_remove_idx = None
        for i, task in enumerate(st.session_state.tasks):
            tid = task["_tid"]
            label = f"{task['name']}  ·  W={task['flexibility_hours']}h  ·  {task['share_of_demand']:.1f}%"
            with st.expander(label, expanded=edit):
                if edit:
                    new_name = st.text_input("Name", value=task["name"], key=f"name_{tid}")
                    new_w = st.number_input("Flexibility window W (hours)",
                                             min_value=0, max_value=72,
                                             value=int(task["flexibility_hours"]),
                                             step=1, key=f"w_{tid}")
                    new_share = st.number_input("Share of demand (%)",
                                                 min_value=0.0, max_value=100.0,
                                                 value=float(task["share_of_demand"]),
                                                 step=1.0, key=f"share_{tid}")
                    new_mult = st.number_input("Max power multiplier",
                                                min_value=1.0, max_value=5.0,
                                                value=float(task["max_power_multiplier"]),
                                                step=0.1, key=f"mult_{tid}")
                    new_color = st.color_picker("Color", value=task["color"], key=f"color_{tid}")

                    st.session_state.tasks[i]["name"] = new_name
                    st.session_state.tasks[i]["flexibility_hours"] = int(new_w)
                    st.session_state.tasks[i]["share_of_demand"] = float(new_share)
                    st.session_state.tasks[i]["max_power_multiplier"] = float(new_mult)
                    st.session_state.tasks[i]["color"] = new_color

                    disable_rm = len(st.session_state.tasks) <= 1
                    if st.button("✕ Remove this task", key=f"rm_{tid}", disabled=disable_rm):
                        to_remove_idx = i
                else:
                    swatch = (
                        f"<span style='display:inline-block;width:12px;height:12px;"
                        f"background:{task['color']};border:1px solid #999;"
                        f"vertical-align:middle;margin-right:6px;'></span>"
                    )
                    st.markdown(f"{swatch}`{task['color']}`", unsafe_allow_html=True)
                    st.markdown(f"**Name:** {task['name']}")
                    st.markdown(f"**Flexibility W:** {task['flexibility_hours']} h")
                    st.markdown(f"**Share of demand:** {float(task['share_of_demand']):.1f}%")
                    st.markdown(f"**Max power multiplier:** {float(task['max_power_multiplier']):.1f}×")

        if to_remove_idx is not None:
            st.session_state.tasks.pop(to_remove_idx)
            st.rerun()

        if edit:
            if st.button("➕ Add task", use_container_width=True):
                st.session_state.tasks.append({
                    "_tid": uuid.uuid4().hex[:8],
                    "name": f"new_task_{len(st.session_state.tasks)+1}",
                    "flexibility_hours": 6,
                    "share_of_demand": 10.0,
                    "max_power_multiplier": 2.0,
                    "color": "#888888",
                })
                st.rerun()

        st.markdown("---")
        run_clicked = st.button("▶ Run", type="primary", use_container_width=True)
        return run_clicked


# ---------------- main ----------------


def render_kpi_strip(kpis, season):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Cost saved", f"{kpis['cost_sav_pct']:.1f}%")
    col2.metric("$ / week", f"${kpis['cost_sav_usd']:,.0f}")
    col3.metric("Carbon avoided", f"{kpis['carbon_sav_pct']:.1f}%")
    col4.metric("Tons / week", f"{kpis['carbon_sav_tons']:,.1f}")
    col5.metric("Load CV",
                f"{kpis['cv_opt']:.3f}×",
                delta=f"baseline: {kpis['cv_base']:.3f}×",
                delta_color="off")
    st.caption(f"Season shown: **{season.capitalize()}**")


def render_main():
    results = st.session_state.results
    season  = st.session_state.season

    st.title("Data Center Workload Scheduler — LP demo")

    st.markdown("""
    This tool optimizes the scheduling of deferrable data center workloads to minimize 
    electricity cost and carbon emissions on the CAISO grid. A 500 MW data center is 
    modeled with four task classes — each with a different flexibility window defining 
    how far the task can be shifted from its release time. The LP optimizer 
    redistributes flexible tasks toward hours with low electricity prices (LMP) and 
    low carbon intensity (MOER), subject to capacity constraints.
    
    **How to use:** Toggle Edit mode in the sidebar to adjust parameters — season, 
    the cost-carbon tradeoff weight α, peak multiplier, and task class configurations. 
    Click Run ▶ to solve. Results update across all panels.
    
    **How to interpret results:** Cost saved and carbon avoided are measured relative 
    to a naive baseline where every task runs immediately at its release hour. Load CV 
    measures schedule smoothness — lower values indicate a flatter, more operationally 
    predictable demand profile. The grid signals panel shows what price and carbon 
    conditions the optimizer is responding to. The demand panel shows how the optimized 
    schedule differs from the baseline by task class.
    """)

    if st.session_state.edit_mode:
        st.warning("Parameters changed — click **Run ▶** to update.", icon="ℹ️")

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if st.session_state.normalize_info:
        orig, adj = st.session_state.normalize_info
        st.info(f"Shares normalized: {orig} → {adj}", icon="ℹ️")

    if results is None:
        st.info("No results yet — click Run.")
        return

    statuses = {s: d["status"] for s, d in results.items()}
    bad = {s: st_ for s, st_ in statuses.items() if st_ not in ("optimal", "optimal_inaccurate")}
    if bad:
        if any(st_ == "infeasible" for st_ in bad.values()):
            st.error("No feasible schedule found with these parameters. "
                     "Try increasing the peak multiplier or reducing task shares.")
        else:
            st.error(f"Solver returned non-optimal status: {bad}")
        return

    season_data = results[season]

    # dim visuals slightly in edit mode
    dim = st.session_state.edit_mode
    container = st.container()
    if dim:
        container.markdown(
            "<style>div[data-testid='stVerticalBlock'] "
            ".dimmed { opacity: 0.45; pointer-events: none; }</style>",
            unsafe_allow_html=True,
        )

    with container:
        render_kpi_strip(season_data["kpis"], season)

        st.divider()

        # Panel 1
        st.subheader("Grid signals — LMP & MOER")
        fig1 = plot_grid_signals(season_data, season)
        st.pyplot(fig1)
        plt.close(fig1)

        st.divider()

        # Panel 2
        st.subheader("Demand — baseline vs. optimized (48 h window, hours 72–119)")
        solved = st.session_state.solved_tasks or st.session_state.tasks
        config_tasks = build_config(solved, st.session_state.peak_mult)["tasks"]
        fig2 = plot_demand_48h_main(results, config_tasks, season)
        st.pyplot(fig2)
        plt.close(fig2)

        compare_seasons = [s for s in SEASONS if s != season]
        cmp_cols = st.columns(2)
        for col, s in zip(cmp_cols, compare_seasons):
            with col:
                st.caption(f"Compare — {s.capitalize()}")
                fig_c = plot_demand_48h_compare(results, config_tasks, s)
                st.pyplot(fig_c)
                plt.close(fig_c)

        st.divider()

        # Panel 3
        st.subheader("Load variability — operational smoothness")
        fig3 = plot_load_variability(season_data, season)
        st.pyplot(fig3)
        plt.close(fig3)
        st.caption(
            "Lower CV indicates a flatter, operationally smoother schedule. "
            "Red = hours running above average demand; blue = below average."
        )


# ---------------- entry ----------------


def trigger_solve(selected_season, alpha, peak_mult):
    tasks_snapshot = [dict(t) for t in st.session_state.tasks]
    errors, warnings = validate(tasks_snapshot)
    for w in warnings:
        st.warning(w, icon="⚠️")
    if errors:
        for e in errors:
            st.error(e)
        return False

    tasks_normalized, norm_info = normalize_shares(tasks_snapshot)

    # persist normalized shares back so the sidebar shows what we actually solved
    if norm_info is not None:
        for i, tn in enumerate(tasks_normalized):
            st.session_state.tasks[i]["share_of_demand"] = tn["share_of_demand"]
    st.session_state.normalize_info = norm_info

    config = build_config(tasks_normalized, peak_mult)
    with st.spinner(f"Solving LP for {selected_season} season, α={alpha:.2f}..."):
        try:
            results = run_all_seasons(config, alpha)
        except Exception as e:
            st.session_state.last_error = f"Solver error: {e}"
            st.session_state.results = None
            return False

    st.session_state.last_error = None
    st.session_state.results = results
    st.session_state.solved_tasks = [dict(t) for t in tasks_normalized]
    return True


def main():
    st.set_page_config(
        page_title="Data Center Workload Scheduler",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()

    run_clicked = render_sidebar()

    # First-load: run optimizer once with defaults so the page opens with data.
    if not st.session_state.first_load_done:
        ok = trigger_solve(st.session_state.season, st.session_state.alpha, st.session_state.peak_mult)
        st.session_state.first_load_done = True
        if ok:
            st.rerun()

    if run_clicked:
        ok = trigger_solve(st.session_state.season, st.session_state.alpha, st.session_state.peak_mult)
        if ok:
            st.session_state.edit_mode = False
            st.rerun()

    render_main()


if __name__ == "__main__":
    main()
