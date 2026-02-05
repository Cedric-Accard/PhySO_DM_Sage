import pandas as pd
from sage.all import *
import signal
import re

"""For now only works when density profiles are given, still need to implement checks for when enclosed mass profiles are given.
"""


class TimeoutError(Exception):
    pass


def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError("Timeout reached")
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
            return result
        return wrapper
    return decorator


# Timeout duration to avoid infinite computations
PROP_TIMEOUT = 10


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
        'e^': 'np.exp',   # crude, see note below
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
        'gamma': 'scipy.special.gamma',
        'Ei': 'scipy.special.expi',
    }

    # Sort by decreasing length so 'arctan' is handled before 'tan'
    for name in sorted(func_map, key=len, reverse=True):
        s = re.sub(r'\b' + name + r'\b', func_map[name], s)

    return s


# ==========================================
# PHYSICAL PROPS FUNCTIONS
# ==========================================
@timeout(PROP_TIMEOUT)
def get_enclosed_mass(rho, r, rp):
    integrand = 4 * pi * rp**2 * rho.subs({r: rp})
    return integral(integrand, rp, 0, r).simplify_full(), integrand


@timeout(PROP_TIMEOUT)
def get_density_from_mass(M_r, r, rp):
    rho_rp = (diff(M_r, r) / (4 * pi * r**2)).simplify_full()
    return rho_rp.subs({r: rp})


@timeout(PROP_TIMEOUT)
def get_circular_velocity(M_r, r, G):
    return sqrt(G * M_r / r).simplify_full()


@timeout(PROP_TIMEOUT)
def get_potential(M_r, r, rp, G):
    integrand = G * M_r.subs({r: rp}) / rp**2
    return -integral(integrand, rp, r, infinity).simplify_full(), integrand, r, infinity


@timeout(PROP_TIMEOUT)
def get_velocity_dispersion(rho, M_r, r, rp, G):
    integrand = (rho.subs({r: rp}) * G * M_r.subs({r: rp})) / rp**2
    int_res = integral(integrand, rp, r, infinity)
    return (int_res / rho).simplify_full(), integrand


@timeout(PROP_TIMEOUT)
def get_surface_density(rho, r, R, rp):
    integrand = 2 * rho.subs({r: rp}) * rp / sqrt(rp**2 - R**2)
    Sigma = integral(integrand, rp, R, infinity).simplify_full()
    return Sigma, integrand


@timeout(PROP_TIMEOUT)
def get_average_surface_density(Sigma_R, R, Rp):
    integrand = Sigma_R.subs({R: Rp}) * Rp
    avg_Sigma = (2 * integral(integrand, Rp, 0, R) / R**2).simplify_full()
    return avg_Sigma, integrand


@timeout(PROP_TIMEOUT)
def get_projected_concentration(Sigma_R, avg_Sigma_R, R):
    # c2(R) = \barΣ(<R) / Σ(R)
    c2 = (avg_Sigma_R / Sigma_R).simplify_full()
    return c2


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
        results.append({'Property': 'Density', 'symb_condition': True, 'symb_result': rho})
        try:
            M_r, mass_int_expr = get_enclosed_mass(rho, r, rp)
            symb = not str(M_r).count('integral')
        except Exception:
            M_r = None
            symb = False
        results.append({'Property': 'Enclosed Mass', 'symb_condition': symb, 'symb_result': M_r})
    elif input_type == 'enclosed_mass':
        M_r = expr
        results.append({'Property': 'Enclosed Mass', 'symb_condition': True, 'symb_result': M_r})
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
            symb = not str(M_r).count('integral')
        else:
            symb = False
    except Exception:
        symb = False
    results.append({'Property': 'Circular Velocity', 'symb_condition': symb, 'symb_result': V_r})

    # Potential
    Phi_inf = None
    pot_int_expr = None
    try:
        if M_r is not None:
            Phi_inf, pot_int_expr, low, high = get_potential(M_r, r, rp, G)
            symb = not str(Phi_inf).count('integral')
        else:
            symb = False
    except Exception:
        symb = False
    results.append({'Property': 'Potential', 'symb_condition': symb, 'symb_result': Phi_inf})

    # Velocity Dispersion
    Sigma2 = None
    sig_int_expr = None
    try:
        if M_r is not None:
            Sigma2, sig_int_expr = get_velocity_dispersion(rho, M_r, r, rp, G)
            symb = not str(Sigma2).count('integral')
        else:
            symb = False
    except Exception:
        symb = False
    results.append({'Property': 'Radial Velocity Dispersion', 'symb_condition': symb, 'symb_result': Sigma2})

    # Surface density Σ(R)
    Sigma_R = None
    sigR_int_expr = None
    try:
        if rho is not None:
            Sigma_R, sigR_int_expr = get_surface_density(rho, r, R, rp)
            symb = not str(Sigma_R).count('integral')
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
            symb = not str(avg_Sigma_R).count('integral')
        else:
            symb = False
    except Exception:
        symb = False
    results.append({
        'Property': 'Average Surface Density',
        'symb_condition': symb,
        'symb_result': avg_Sigma_R
    })

    # Projected concentration c2(R) = \barΣ/Σ
    c2_R = None
    try:
        if (Sigma_R is not None) and (avg_Sigma_R is not None):
            c2_R = get_projected_concentration(Sigma_R, avg_Sigma_R, R)
            symb = not str(c2_R).count('integral')
        else:
            symb = False
    except Exception:
        symb = False
    results.append({
        'Property': 'Projected Concentration',
        'symb_condition': symb,
        'symb_result': c2_R
    })
    return pd.DataFrame(results)


# ==========================================
# EXECUTION
# ==========================================

input_type = 'density'  # 'density' or 'enclosed_mass

if input_type == 'density':
    profiles = {
        "NFW": ("rho0 / ((r / Rs) * (1 + r / Rs)**2)", ['rho0', 'Rs']),
        "superNFW": ("rho0 / ((r / Rs) * (1 + r / Rs)**Rational(5, 2))", ['rho0', 'Rs']),
        "pISO": ("rho0 / (1 + (r / Rs)**2)", ['rho0', 'Rs']),
        "pISO1": ("1 / (1 + (r/1)**2)", []),
        "Burkert": ("rho0 * Rs**3 / ((r + Rs)*(r**2 + Rs**2))", ['rho0', 'Rs']),
        "Lucky13": ("rho0 / (1 + (r/Rs))**3", ['rho0', 'Rs']),
        "Einasto": ("rho0 * exp(-2/alpha * ((r/Rs)**alpha - 1))", ['rho0', 'Rs', 'alpha']),
        "coreEinasto": ("rho0 * exp(-2/alpha * ((r/Rs + rc/Rs)**alpha - 1))", ['rho0', 'Rs', 'alpha', 'rc']),
        "DiCintio": ("rho0/((r/Rs)**alpha * (1+(r/Rs)**(1/beta))**(beta*(gamma-alpha)))", ['rho0', 'Rs', 'alpha', 'beta', 'gamma']),
        "gNFW": ("rho0 / ((r/Rs)**gamma * (1 + r/Rs)**(3-gamma))", ['rho0', 'Rs', 'gamma']),
        "Dekel_Zhao": ("rho0 / ((r/Rs)**alpha * (1 + (r/Rs)**(1/2))**(7-2*alpha))", ['rho0', 'Rs', 'alpha']),
        "Exponential": ("rho0 * exp(-r/Rs)", ['rho0', 'Rs']),
        "Exponential1": ("9.6 * exp(-r/1.4)", []),
        "Exponential2": ("rho0 * exp(-r/(Rs_1 + Rs_2))", ['rho0', 'Rs_1', 'Rs_2']),
    }
elif input_type == 'enclosed_mass':
    profiles = {
        "Plummer_M": ("M0 * r**3 / (r**2 + a**2)**(3/2)", ["M0", "a"],),
        "Hernquist_M": ("M * r**2 / (r + a)**2", ["M", "a"],),
        "BadLinearM": ("M0 * r", ["M0"],),
        # "BadCuspM": ("M0 * r**(1/2)", ["M0"], ),
    }


print("Generating Analytical Solutions...")
full_data = {}      # Store full DFs with formulas for export
summary_dict = {}   # Store just True/False for printing

for name, (prof_str, const_names) in profiles.items():
    print(f"Processing {name}...")
    df = check_analytical_properties(prof_str, const_names, input_type=input_type)
    full_data[name] = df
    if not df.empty:
        summary_dict[name] = df.set_index('Property')['symb_condition']

final_table = pd.DataFrame(summary_dict)
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)
print("\n=== ANALYTICAL SOLUTIONS SUMMARY ===")
print(final_table)

# --- Export to Python File ---
print("\nWriting 'derived_profiles.py'...")

with open("derived_profiles.py", "w") as f:
    f.write("import numpy as np\n")
    f.write("import scipy.special\n\n")
    f.write("# Helper wrapper for Sage compatibility\n")
    f.write("def dilog(x):\n    return scipy.special.spence(1 - x)\n\n")

    for name, df in full_data.items():
        f.write(f"# ================= {name} =================\n")

        for _, row in df.iterrows():
            if row['symb_condition'] and row['symb_result'] is not None:
                prop_name = row['Property'].replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").lower()
                # Special naming for potentials
                if prop_name == "potential":
                    func_name = f"{name}_potential"
                elif prop_name == "radial_velocity_dispersion":
                    func_name = f"{name}_sigma2"
                elif prop_name == "enclosed_mass":
                    func_name = f"{name}_mass"
                else:
                    func_name = f"{name}_{prop_name}"

                expr_str = sage_to_python_str(row['symb_result'])

                f.write(f"def {func_name}(r, rho0, Rs, G=4.301e-6):\n")
                f.write(f"    # {row['Property']}\n")
                f.write(f"    return {expr_str}\n\n")

print("Done. Check 'derived_profiles.py' for usable code.")
