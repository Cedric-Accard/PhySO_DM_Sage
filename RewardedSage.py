import os
import re
import sys
import signal
import pandas as pd
from sage.all import *
from parser import Parser
import numpy as np


def is_symbolic(expr) -> bool:
    if expr is None:
        return False
    s = str(expr)
    if 'integrate' in s:
        return False
    if 'undef' in s:
        return False
    return True


def sage_to_python_str(expr):
    """
    Converts a Sage symbolic expression to a valid Python string usable with numpy and scipy.special.
    """
    if expr is None:
        return "None"

    s = str(expr)

    # 1. Simple global replacements that are safe anywhere
    simple_replacements = {
        'pi': 'np.pi',
        '^': '**',
    }
    for old, new in simple_replacements.items():
        s = s.replace(old, new)

    # 2. Function-name replacements, using word boundaries to avoid overlaps
    func_map = {
        'log10': 'np.log10',
        'log': 'np.log',
        'exp': 'np.exp',
        'sqrt': 'np.sqrt',
        'arcsin': 'np.arcsin',
        'arccos': 'np.arccos',
        'arctan': 'np.arctan',
        'sin': 'np.sin',
        'cos': 'np.cos',
        'tan': 'np.tan',
        'Ei': 'scipy.special.expi',
    }
    # Sort by decreasing length so 'arctan' is handled before 'tan'
    for name in sorted(func_map, key=len, reverse=True):
        s = re.sub(r'\b' + name + r'\b', func_map[name], s)

    s = re.sub(r'\bgamma\s*\(', 'scipy.special.gamma(', s)

    return s


# ==========================================
# PHYSICAL PROPS FUNCTIONS
# ==========================================

def get_enclosed_mass(rho, r, rp):
    integrand = 4 * pi * rp**2 * rho.subs({r: rp})
    return integral(integrand, rp, 0, r).simplify_full(), integrand


def get_density_from_mass(M_r, r, rp):
    rho_rp = (diff(M_r, r) / (4 * pi * r**2)).simplify_full()
    return rho_rp.subs({r: rp})


def get_circular_velocity(M_r, r, G):
    return sqrt(G * M_r / r).simplify_full()


def get_potential(M_r, r, rp, G):
    integrand = G * M_r.subs({r: rp}) / rp**2
    return -integral(integrand, rp, r, infinity).simplify_full(), integrand, r, infinity


def get_velocity_dispersion(rho, M_r, r, rp, G):
    integrand = (rho.subs({r: rp}) * G * M_r.subs({r: rp})) / rp**2
    int_res = integral(integrand, rp, r, infinity)
    return (int_res / rho).simplify_full(), integrand


def get_surface_density(rho, r, R, rp):
    integrand = 2 * rho.subs({r: rp}) * rp / sqrt(rp**2 - R**2)
    Sigma = integral(integrand, rp, R, infinity).simplify_full()
    return Sigma, integrand


def get_average_surface_density(Sigma_R, R, Rp):
    integrand = Sigma_R.subs({R: Rp}) * Rp
    avg_Sigma = (2 * integral(integrand, Rp, 0, R) / R**2).simplify_full()
    return avg_Sigma, integrand


# ==========================================
# MAIN CHECK ROUTINE
# ==========================================

def check_analytical_properties(input_str, free_consts_names, input_type='density'):
    """ Given a density profile as str, checks analytical properties."""
    # --- Setup ---
    var('r, rp, G')
    assume(r > 0)
    assume(rp > 0)
    assume(G > 0)
    var('R, Rp')
    assume(R > 0)
    assume(Rp > 0)

    vars_dict = {'r': r, 'rp': rp, 'G': G, 'R': R, 'Rp': Rp}
    for name in free_consts_names:
        v = var(name)
        assume(v > 0)

        if name == 'Rs':
            assume(v > 1)

        if name == 'alpha':
            assume(v > 0)

        if name == 'beta':
            assume(v > 0)

        if name == 'gamma':
            # assume(v > -1)
            assume(v < 3)

        vars_dict[name] = v

    vars_dict.update({'log': log, 'exp': exp, 'sqrt': sqrt, 'pi': pi, 'infinity': infinity})

    try:
        expr = sage_eval(input_str, locals=vars_dict)
        if hasattr(expr, '__call__'):
            expr = expr(r=r)
    except Exception as e:
        print(f"Error parsing {input_str}: {e}")
        return pd.DataFrame()

    results = []

    if input_type == 'density':
        rho = expr
        try:
            M_r, mass_int_expr = get_enclosed_mass(rho, r, rp)
            symb = is_symbolic(M_r)
        except Exception:
            M_r = None
            symb = False
        results.append({'Property': 'Enclosed Mass', 'symb_condition': symb, 'symb_result': M_r})
    elif input_type == 'enclosed_mass':
        M_r = expr
        try:
            rho = get_density_from_mass(M_r, r, rp)
            symb = not str(rho).count('integral')
        except Exception:
            rho = None
            symb = False
        results.append({'Property': 'Density', 'symb_condition': symb, 'symb_result': rho})
    else:
        print("input_type must be 'density' or 'enclosed_mass'")
        return pd.DataFrame()

    # Circular Velocity
    V_r = None
    try:
        if M_r is not None:
            V_r = get_circular_velocity(M_r, r, G)
            symb = is_symbolic(V_r)
        else:
            symb = False
    except Exception:
        symb = False
    results.append({'Property': 'Circular Velocity', 'symb_condition': symb, 'symb_result': V_r})

    # Velocity Dispersion
    Sigma2 = None
    sig_int_expr = None
    try:
        if M_r is not None:
            Sigma2, sig_int_expr = get_velocity_dispersion(rho, M_r, r, rp, G)
            symb = is_symbolic(Sigma2)
        else:
            symb = False
    except Exception:
        symb = False
    results.append({'Property': 'Radial Velocity Dispersion', 'symb_condition': symb, 'symb_result': Sigma2})

    # Potential
    Phi_inf = None
    pot_int_expr = None
    try:
        if M_r is not None:
            Phi_inf, pot_int_expr, low, high = get_potential(M_r, r, rp, G)
            symb = is_symbolic(Phi_inf)
        else:
            symb = False
    except Exception:
        symb = False
    results.append({'Property': 'Potential', 'symb_condition': symb, 'symb_result': Phi_inf})

    # Surface density Σ(R)
    Sigma_R = None
    sigR_int_expr = None
    try:
        if rho is not None:
            Sigma_R, sigR_int_expr = get_surface_density(rho, r, R, rp)
            symb = is_symbolic(Sigma_R)
        else:
            symb = False
    except Exception:
        symb = False
    results.append({'Property': 'Surface Density', 'symb_condition': symb, 'symb_result': Sigma_R})

    # Average surface density \barΣ(<R)
    avg_Sigma_R = None
    avgSigma_int_expr = None
    try:
        if Sigma_R is not None:
            avg_Sigma_R, avgSigma_int_expr = get_average_surface_density(Sigma_R, R, Rp)
            symb = is_symbolic(avg_Sigma_R)
        else:
            symb = False
    except Exception:
        symb = False
    results.append({
        'Property': 'Average Surface Density',
        'symb_condition': symb,
        'symb_result': avg_Sigma_R
    })

    return pd.DataFrame(results)


# ==========================================
# EXECUTION
# ==========================================

input_type = 'density'

sys.path.append(os.path.dirname(__file__))
SR_LOG = "NIHAO_runb_hydro_0_1_nspe2_nclass1_bs10000/SR.log"
p = Parser(SR_LOG, verbose=False)
best_phys = p.get_physical_expr(n=10)

physo_profiles = {}
physo_rewards = {}
for idx, entry in enumerate(best_phys):
    name = f"PhySO_{idx}"
    physo_profiles[name] = (entry["expr"], entry["params"])
    physo_rewards[name] = entry["reward"]

profiles = physo_profiles


# ==========================================
# CHECK PROPS
# ==========================================

print("Generating Analytical Solutions...")
full_data = {}      # Store full DFs with formulas for export
summary_dict = {}   # Store just True/False for printing

for name, (prof_str, const_names) in profiles.items():
    print(f"Processing {name}...")
    df = check_analytical_properties(prof_str, const_names)
    full_data[name] = df
    if not df.empty:
        summary_dict[name] = df.set_index('Property')['symb_condition']

PROPERTY_WEIGHTS = {"Density": 1.0, "Enclosed Mass": 1.0, "Circular Velocity": 1.0, "Radial Velocity Dispersion": 1.0, "Potential": 1.0, "Surface Density": 1.0, "Average Surface Density": 1.0}

analytic_scores = {}
for name, df in full_data.items():
    score = 0.0
    for _, row in df.iterrows():
        w = PROPERTY_WEIGHTS.get(row["Property"], 0.0)
        if bool(row["symb_condition"]):
            score += w
    analytic_scores[name] = score


names = list(summary_dict.keys())
rewards_arr = np.array([physo_rewards[n] for n in names])
ascores_arr = np.array([analytic_scores[n] for n in names])

# Normalise
r_norm = (rewards_arr - rewards_arr.min()) / (rewards_arr.max() - rewards_arr.min() + 1e-9)
a_norm = ascores_arr / (ascores_arr.max() + 1e-9)

alpha = 0.5  # trade-off between reward and analytic completeness
combined = alpha * r_norm + (1.0 - alpha) * a_norm

combined_df = pd.DataFrame({"reward": rewards_arr, "analytic_score": ascores_arr, "combined_score": combined}, index=names)

# Order names by combined score, descending
ordered_names = list(combined_df.sort_values("combined_score", ascending=False).index)

final_table = pd.DataFrame({name: summary_dict[name] for name in ordered_names})
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n=== ANALYTICAL SOLUTIONS SUMMARY (sorted) ===")
print(final_table.T)

# print("\n=== SCORES ===")
# print(combined_df.loc[ordered_names])
