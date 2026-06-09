# DC Workload Scheduling Optimizer

A carbon- and cost-aware linear programming optimizer for data center workload scheduling, developed as part of CE 295 (Data Science for Energy) at UC Berkeley.

The optimizer co-schedules flexible compute workloads and a battery energy storage system (BESS) against real-time CAISO locational marginal prices (LMPs) and WattTime marginal operating emissions rates (MOERs), targeting a 500 MW hyperscale data center on the CAISO NP15 node.

---

## Key Results

- **~$16M/yr** in electricity cost savings from optimal workload shifting and BESS dispatch
- **~19K tons CO₂/yr** reduction in carbon emissions
- **~9x** lower carbon intensity compared to a carbon-unaware scheduling baseline
- Pareto frontier analysis reveals the cost-carbon tradeoff across seasonal demand patterns

---

## Problem Formulation

The scheduler solves a multi-objective LP at hourly resolution over a 168-hour (weekly) horizon:

**Minimize:**
$$\alpha \cdot \text{cost} + (1 - \alpha) \cdot \text{carbon}$$

where $\alpha \in [0, 1]$ is a user-controlled cost-carbon tradeoff parameter.

**Subject to:**
- Workload deadline constraints (four flexibility classes: 0h, 4h, 6-12h, 24h deferral windows)
- BESS charge/discharge power limits and state-of-charge bounds
- Round-trip efficiency losses
- Optional carbon cap constraint

Workload classes follow empirical demand profiles from Radovanovic et al. (IEEE TPS 2023), Acun et al. (Carbon Explorer 2023), and Wiesner et al. (ACM Middleware 2021).

---

## Project Structure

```
dc-workload-scheduling/
├── run.py                  # Main entry point: runs LP across seasons and scenarios
├── app.py                  # Streamlit dashboard for interactive exploration
├── inputs/
│   ├── task_config.yaml    # Workload and BESS configuration
│   └── demand_profile.csv  # Hourly demand profile (gitignored)
├── src/
│   ├── optimizer.py        # CVXPY LP formulation (CLARABEL solver)
│   ├── data_loader.py      # CAISO LMP + WattTime MOER data ingestion
│   ├── workload.py         # Workload demand profile generation by class
│   ├── penalty.py          # BESS degradation and ownership cost model
│   └── visualize.py        # Output plots and Pareto frontier figures
└── requirements.txt
```

---

## Quickstart

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the optimizer:**
```bash
python run.py
```

Outputs are written to `outputs/` — schedule CSVs, KPI summary, and Pareto frontier plots.

**Launch the interactive dashboard:**
```bash
streamlit run app.py
```

---

## Data Sources

| Signal | Source | Node / Region |
|---|---|---|
| Locational Marginal Price | CAISO OASIS API | TH_NP15_GEN-APND |
| Marginal Operating Emissions Rate | WattTime API | CAISO NP15 |

Data is fetched at runtime and not included in this repository. You will need a WattTime API account to pull MOER signals.

---

## Configuration

Key parameters in `inputs/task_config.yaml`:

```yaml
battery:
  enabled: true
  capacity_mwh: 500
  max_charge_mw: 200
  max_discharge_mw: 200
  efficiency: 0.92
```

The cost-carbon tradeoff parameter `alpha` and seasonal date ranges are set in `run.py`.

---

## Dependencies

- [CVXPY](https://www.cvxpy.org/) with CLARABEL solver
- pandas, numpy, matplotlib
- Streamlit (dashboard only)

See `requirements.txt` for full list.

---

## References

- Radovanovic et al., "Carbon-Aware Computing for Datacenters," IEEE TPS 2023
- Acun et al., "Carbon Explorer," ACM ASPLOS 2023
- Wiesner et al., "Let's Wait Awhile," ACM Middleware 2021
- NREL Utility-Scale Battery Storage Cost Projections
