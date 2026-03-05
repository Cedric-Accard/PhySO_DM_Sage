import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from parser import Parser
from functions_sage import check_analytical_properties

# Only this DMO/Hydro pair
RUNS = [
    # "./RunsToPlot/NIHAO_runb_dmo_1_nspe1_bs10000",
    # "./RunsToPlot/NIHAO_runb_hydro_1_nspe1_bs10000",
    "./RunsToPlot/NIHAO_runb_dmo_1_nspe2_bs10000",
    "./RunsToPlot/NIHAO_runb_hydro_1_nspe2_bs10000",
    "./RunsToPlot/NIHAO_runb_dmo_1_nspe3_bs10000",
    "./RunsToPlot/NIHAO_runb_hydro_1_nspe3_bs10000",
    "./RunsToPlot/NIHAO_runb_dmo_1_nspe4_bs10000",
    "./RunsToPlot/NIHAO_runb_hydro_1_nspe4_bs10000",
]

SPE_PREFIXES = ("rho", "rs")   # adapt if needed
CLASS_PREFIXES = ("c",)

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

    wanted = ["Enclosed Mass", "Circular Velocity", "Radial Velocity Dispersion", "Potential", "Surface Density", "Average Surface Density"]

    if "Property" in df.columns:
        df_sub = df[df["Property"].isin(wanted)]
        if df_sub.empty:
            df_sub = df
    else:
        df_sub = df

    # symb_condition is True/False/None -> treat non‑True as non‑analytic
    if "symb_condition" in df_sub.columns:
        vals = df_sub["symb_condition"].astype(float).values
    else:
        vals = np.zeros(len(df_sub), dtype=float)

    if len(vals) == 0:
        return 0.0

    # number of analytic quantities:
    return float(vals.sum())


def count_spe_class(params):
    n_spe = sum(p.startswith(SPE_PREFIXES) for p in params)
    n_class = sum(p.startswith(CLASS_PREFIXES) for p in params)
    return n_spe, n_class


records = []

for run_dir in RUNS:
    kind, nspe, run_name = parse_run_name(run_dir)
    sr_log = os.path.join(run_dir, "SR.log")
    if not os.path.isfile(sr_log):
        print(f"[WARN] SR.log not found in {run_dir}, skipping.")
        continue

    print(f"Processing {sr_log}")
    p = Parser(sr_log, verbose=False)
    best_phys = p.get_physical_expr(n=10)  # top 10 expressions

    physo_profiles = {}
    for idx, entry in enumerate(best_phys):
        name = f"{run_name}_expr{idx}"
        physo_profiles[name] = (entry["expr"], entry["params"])

    for idx, (name, (prof_str, const_names)) in enumerate(physo_profiles.items()):
        entry = best_phys[idx]
        reward = float(entry["reward"])
        R2 = float(reward_to_R2(reward))
        n_spe, n_class = count_spe_class(entry["params"])
        n_free = n_spe + n_class

        try:
            df_sage = check_analytical_properties(
                prof_str, const_names, input_type=input_type
            )
        except Exception as e:
            print(f" [WARN] Sage failed for {name}: {e}")
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
                n_spe_used=n_spe,
                n_class_used=n_class,
                analytic_score=a_score,
            )
        )

df = pd.DataFrame.from_records(records)
# print(df.head())
# print("Total rows:", len(df))
# print("kinds:", df["kind"].unique())
# print("nspe:", df["nspe"].unique())

# print(df[["kind","expr_idx","analytic_score","R2"]])


# ---- 2D plot analytic_score vs R^2, coloured by kind (DMO vs Hydro) ----

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=False)
axes_dict = {"dmo": axes[0], "hydro": axes[1]}

# colour by nspe
nspe_vals = sorted(df["nspe"].unique())
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(min(nspe_vals), max(nspe_vals))

for kind in ["dmo", "hydro"]:
    ax = axes_dict[kind]
    sub_kind = df[df["kind"] == kind]
    if sub_kind.empty:
        ax.set_visible(False)
        continue

    for run_name, sub_run in sub_kind.groupby("run"):
        sub_run = sub_run.sort_values("analytic_score")  # for nicer lines

        # all rows in a run share the same nspe
        nspe = sub_run["nspe"].iloc[0]
        color = cmap(norm(nspe))

        ax.plot(
            sub_run["analytic_score"],
            sub_run["R2"],
            "-o",
            color=color,
            markersize=4,
            linewidth=1,
            alpha=0.8,
            label=f"nspe={nspe}",
        )

    ax.set_title(kind.upper())
    ax.set_xlabel("Analytic score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
axes[0].set_ylabel(r"$R^2$")

# # colour bar for nspe
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
# cbar.set_label(r"$N_{\rm SPE}$")

fig.suptitle(r"PhySO: $R^2$ vs analytic score (10 best physical expressions per run)")
plt.tight_layout()
plt.show()
