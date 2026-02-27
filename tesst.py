import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parser import Parser
from functions_sage import check_analytical_properties

RUNS = [
    "./RunsToPlot/NIHAO_runb_dmo_1_nspe1_bs10000",
    "./RunsToPlot/NIHAO_runb_hydro_1_nspe1_bs10000",
    "./RunsToPlot/NIHAO_runb_dmo_1_nspe2_bs10000",
    "./RunsToPlot/NIHAO_runb_hydro_1_nspe2_bs10000",
    "./RunsToPlot/NIHAO_runb_dmo_1_nspe3_bs10000",
    "./RunsToPlot/NIHAO_runb_hydro_1_nspe3_bs10000",
    "./RunsToPlot/NIHAO_runb_dmo_1_nspe4_bs10000",
    "./RunsToPlot/NIHAO_runb_hydro_1_nspe4_bs10000",
]

input_type = "density"


def reward_to_R2(reward):
    reward = np.asarray(reward, dtype=float)
    return 2.0 / reward - (1.0 / reward) ** 2.0


def parse_run_name(path):
    name = os.path.basename(path.rstrip("/"))
    parts = name.split("_")
    if "dmo" in parts:
        kind = "dmo"
    elif "hydro" in parts:
        kind = "hydro"
    else:
        kind = "unknown"
    nspe = None
    for p in parts:
        if p.startswith("nspe"):
            try:
                nspe = int(p.replace("nspe", ""))
            except ValueError:
                pass
    return kind, nspe, name


def analytic_score_from_df(df):
    if df.empty:
        return 0.0
    # Adapt these to the actual content of df
    wanted = ["Enclosed Mass", "Circular Velocity", "Radial Velocity Dispersion", "Potential", "Surface Density", "Average Surface Density"]
    if "Property" in df.columns:
        df_sub = df[df["Property"].isin(wanted)]
        if df_sub.empty:
            df_sub = df
    else:
        df_sub = df

    if "is_symbolic" in df_sub.columns:
        vals = df_sub["is_symbolic"].astype(int).values
    elif "symb_condition" in df_sub.columns:
        vals = df_sub["symb_condition"].notna().astype(int).values
    else:
        vals = np.zeros(len(df_sub), dtype=int)

    score = int(vals.sum())   # number of analytic quantities (0..N)
    return float(score)


records = []

for run_dir in RUNS:
    kind, nspe, run_name = parse_run_name(run_dir)
    sr_log = os.path.join(run_dir, "SR.log")
    if not os.path.isfile(sr_log):
        print(f"[WARN] SR.log not found in {run_dir}, skipping.")
        continue

    print(f"Processing {sr_log}")
    p = Parser(sr_log, verbose=False)
    best_phys = p.get_physical_expr(n=10)  # adjust n as you like

    # build profiles exactly as in your working code
    physo_profiles = {}
    for idx, entry in enumerate(best_phys):
        name = f"{run_name}_expr{idx}"
        physo_profiles[name] = (entry["expr"], entry["params"])

    for idx, (name, (prof_str, const_names)) in enumerate(physo_profiles.items()):
        # get the corresponding entry again to fetch reward etc.
        entry = best_phys[idx]
        reward = float(entry["reward"])
        R2 = float(reward_to_R2(reward))
        n_free = len(entry["params"])

        try:
            df_sage = check_analytical_properties(
                prof_str, const_names, input_type=input_type
            )
        except Exception as e:
            print(f"  [WARN] Sage failed for {name}: {e}")
            df_sage = pd.DataFrame()
        a_score = analytic_score_from_df(df_sage)

        records.append(
            dict(
                run=run_name,
                kind=kind,
                nspe=nspe,
                expr_idx=idx,
                R2=R2,
                n_free=n_free,
                analytic_score=a_score,
            )
        )

df = pd.DataFrame.from_records(records)
print(df.head())
print("Total rows:", len(df))
print("kinds:", df["kind"].unique())
print("nspe:", df["nspe"].unique())

# ---- 2D plot: R2 vs n_free, coloured by analytic_score ----
kind_to_color = {"hydro": "tab:blue", "dmo": "tab:orange", "unknown": "grey"}

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, sharex=True)
axes_dict = {"dmo": axes[0], "hydro": axes[1]}

for kind in ["dmo", "hydro"]:
    ax = axes_dict[kind]
    sub = df[df["kind"] == kind]
    if sub.empty:
        ax.set_visible(False)
        continue

    sc = ax.scatter(
        sub["n_free"],
        sub["R2"],
        c=sub["analytic_score"],
        cmap="viridis",
        vmin=0.0,
        vmax=df["analytic_score"].max(),
        s=30,
        alpha=0.8,
        edgecolor="none",
    )

    ax.set_title(kind.upper())
    ax.set_xlabel("Number of free parameters")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel(r"$R^2$")

# colour bar to the right of the HYDRO panel only
cbar = fig.colorbar(sc, ax=axes[1], shrink=0.85, pad=0.02)
cbar.set_label("Analytic score")

# integer x-ticks
xmin = int(df["n_free"].min())
xmax = int(df["n_free"].max())
xticks = np.arange(xmin, xmax + 1, 1)
for ax in axes:
    ax.set_xticks(xticks)

fig.suptitle(r"PhySO: $R^2$ vs number of free parameters, coloured by analytic score")
plt.tight_layout()  # leave room on the right and top
plt.show()
