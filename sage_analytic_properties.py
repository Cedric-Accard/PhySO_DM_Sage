import pandas as pd
from sage.all import *
import signal

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
    Converts a Sage symbolic expression to a valid Python string usable with numpy and scipy.special. Only used in the final export step.
    """
    if expr is None:
        return "None"

    # 1. Get string representation
    s = str(expr)

    # 2. Map mathematical functions to numpy/scipy equivalents
    replacements = {
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
        'pi': 'np.pi',
        'e^': 'np.exp',                 # Sage sometimes outputs e^x
        '^': '**',                      # Python power
        'gamma': 'scipy.special.gamma',
        'Ei': 'scipy.special.expi',     # Exponential integral
    }

    for sage_func, py_func in replacements.items():
        s = s.replace(sage_func, py_func)

    return s


# ==========================================
# PHYSICAL PROPS FUNCTIONS
# ==========================================
@timeout(PROP_TIMEOUT)
def get_enclosed_mass(rho, r, rp):
    integrand = 4 * pi * rp**2 * rho.subs({r: rp})
    return integral(integrand, rp, 0, r).simplify_full(), integrand


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


# ==========================================
# MAIN CHECK ROUTINE
# ==========================================
def check_analytical_properties(density_str, free_consts_names):
    """ Given a density profile as str, checks analytical properties."""
    # --- Setup ---
    var('r, rp, G')
    assume(r > 0)
    assume(rp > 0)
    assume(G > 0)

    vars_dict = {'r': r, 'rp': rp, 'G': G}
    for name in free_consts_names:
        v = var(name)
        assume(v > 0)

        if name == 'Rs':
            assume(v > 1)

        # Specific Constraints for Stability

        # Shape Parameters (alpha, beta, gamma)
        if name == 'alpha':
            assume(v > 0)  # alpha usually controls sharpness of transition (must be > 0)

        if name == 'beta':
            assume(v > 0)  # beta controls outer slope, let's assume > 0 to start

        if name == 'gamma':
            # assume(v > -1)
            assume(v < 3)  # gamma controls inner slope (cusp), typically < 3 to avoid infinite mass

        vars_dict[name] = v

    vars_dict.update({'log': log, 'exp': exp, 'sqrt': sqrt, 'pi': pi, 'infinity': infinity})

    try:
        rho = sage_eval(density_str, locals=vars_dict)
        if hasattr(rho, '__call__'):
            rho = rho(r=r)
    except Exception as e:
        print(f"Error parsing {density_str}: {e}")
        return pd.DataFrame()

    results = []

    # 1. Density for completeness
    results.append({'Property': 'Density', 'symb_condition': True, 'symb_result': rho})

    # 2. Enclosed Mass
    M_r = None
    mass_int_expr = None
    try:
        M_r, mass_int_expr = get_enclosed_mass(rho, r, rp)
        symb = not str(M_r).count('integral')
    except Exception:
        symb = False

    results.append({'Property': 'Enclosed Mass', 'symb_condition': symb, 'symb_result': M_r})

    # 3. Circular Velocity
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

    # 4. Potential
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

    # 5. Velocity Dispersion
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

    return pd.DataFrame(results)


# ==========================================
# EXECUTION
# ==========================================

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
    "Dekel-Zhao": ("rho0 / ((r/Rs)**alpha * (1 + (r/Rs)**(1/2))**(7-2*alpha))", ['rho0', 'Rs', 'alpha']),
    "Exponential": ("rho0 * exp(-r/Rs)", ['rho0', 'Rs']),
    "Exponential1": ("9.6 * exp(-r/1.4)", []),
    "Exponential2": ("rho0 * exp(-r/(Rs_1 + Rs_2))", ['rho0', 'Rs_1', 'Rs_2']),
}

print("Generating Analytical Solutions...")
full_data = {}      # Store full DFs with formulas for export
summary_dict = {}   # Store just True/False for printing

for name, (prof_str, const_names) in profiles.items():
    print(f"Processing {name}...")
    df = check_analytical_properties(prof_str, const_names)
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
