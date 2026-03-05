import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import ast
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from cmcrameri import cm
from functions_sage import check_analytical_properties


# one example file; generalise to a list if you have one per run
CSV_PATH = "./RunsToPlot/SageRuns/NIHAO_runb_hydro_0_1_nspe2_nclass1_bs10000/SR_curves_data.csv"
# CSV_PATH = "./RunsToPlot/NIHAO_runb_hydro_1_nspe2_bs10000/SR_curves_data.csv"

input_type = "density"


def extract_parameters(expr: str, allowed_params: set[str]):
    tree = ast.parse(expr, mode="eval")
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names & allowed_params


def reward_to_R2(reward):
    reward = np.asarray(reward, dtype=float)
    return 2.0 / reward - (1.0 / reward) ** 2.0


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


# -------------------------------------------------------
# Read CSV and build records (one per epoch)
# -------------------------------------------------------

df_csv = pd.read_csv(CSV_PATH)

# identify parameter columns for the best_prog_of_epoch block:
param_cols = df_csv.columns[8:]
# allowed parameter *base* names: strip trailing _digits
allowed_parameters = set(re.sub(r"_[0-9]+$", "", c) for c in param_cols)
# param_cols = ["c0"] + [col for col in df_csv.columns if col.startswith("rho0_")]

records = []

for idx, row in df_csv.iterrows():
    expr_str = row["best_prog_of_epoch"]
    const_names = list(extract_parameters(expr_str, allowed_parameters))

    # separate SPE vs CLASS parameters
    spe_params = [p for p in const_names if p.startswith(("rho0", "rs"))]
    class_params = [p for p in const_names if p.startswith("c")]
    reward = float(row["max_R"])
    R2 = float(reward_to_R2(reward))
    n_spe = len(spe_params)
    n_class = len(class_params)
    n_free = n_spe + n_class

    # Sage analytic properties
    try:
        df_sage = check_analytical_properties(expr_str, const_names, input_type=input_type)
    except Exception as e:
        print(f"[WARN] Sage failed for epoch {row['epoch']}: {e}")
        df_sage = pd.DataFrame()

    a_score = analytic_score_from_df(df_sage)

    records.append(
        dict(
            epoch=int(row["epoch"]),
            R2=R2,
            n_free=n_free,
            n_spe=n_spe,
            n_class=n_class,
            analytic_score=a_score,
        )
    )

df = pd.DataFrame.from_records(records)
print(df.to_string())
print("Total epochs:", len(df))

# -------------------------------------------------------
# 3D scatter plot: n_free vs R^2 vs analytic_score, coloured by epoch
# -------------------------------------------------------
df = pd.DataFrame.from_records(records)
df_sorted = df.sort_values("epoch").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7, 5))

sc = ax.scatter(
    df_sorted["epoch"],
    df_sorted["R2"],
    c=df_sorted["analytic_score"],
    cmap=cm.managua,
    s=40,
    alpha=0.9,
    edgecolor="none",
)

# optional line connecting epochs
ax.plot(df_sorted["epoch"], df_sorted["R2"], color="0.7", linewidth=1.0, alpha=0.6)

ax.set_xlabel("Epoch")
ax.set_ylabel(r"$R^2$")

cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Analytic score")

plt.tight_layout()
plt.show()
