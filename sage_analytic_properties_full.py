import pandas as pd
import numpy as np
import scipy.integrate
from sage.all import *
import signal

"""For now only works when density profiles are given, still need to implement checks for when enclosed mass profiles are given.
It appears that Einasto profile is not working properly, need to check that later."""


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


# Timeout durations
PROP_TIMEOUT = 10   # seconds
NUM_TIMEOUT = 5     # seconds


def sage_to_python_str(expr):
    """
    Converts a Sage symbolic expression to a valid Python string
    usable with numpy and scipy.special.
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
        'sin': 'np.sin',
        'cos': 'np.cos',
        'tan': 'np.tan',
        'arctan': 'np.arctan',
        'arcsin': 'np.arcsin',
        'arccos': 'np.arccos',
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
@timeout(NUM_TIMEOUT)
def check_numeric_integrability(integrand_expr, r_val, lower_lim, upper_lim, vars_dict, integration_var):
    """Checks if the integral converges numerically using Scipy quad."""
    try:
        # 1. Substitute physical constants (Rs=1, rho0=1, G=1)
        # Creation of a substitution dict where all constants are 1.0
        # We use the symbols from vars_dict to key the substitution
        subs_dict = {}
        for name, sym in vars_dict.items():
            if name not in ['r', 'rp', 'log', 'exp', 'sqrt', 'pi', 'infinity']:
                subs_dict[sym] = 1.0

        # Also set G=1 if present
        if 'G' in vars_dict:
            subs_dict[vars_dict['G']] = 1.0

        # Substitute r (the outer variable) with a fixed value (e.g. 1.0)
        # This leaves only the integration variable (rp)
        subs_dict[vars_dict['r']] = float(r_val)

        # Get the 1D function to integrate
        func_1d = integrand_expr.subs(subs_dict)

        # Convert to a python lambda for scipy
        # fast_callable is Sage's efficient way to do this
        f_fast = fast_callable(func_1d, vars=[integration_var], domain=float)

        # Handle limits
        low = float(lower_lim) if lower_lim != -infinity else -np.inf
        up = float(upper_lim) if upper_lim != infinity else np.inf

        # Integrate
        val, err = scipy.integrate.quad(f_fast, low, up, limit=50)
        return not np.isnan(val) and not np.isinf(val)

    except Exception:
        return False


@timeout(NUM_TIMEOUT)
def check_numeric_limit(expr, r_sym, vars_dict):
    """ Checks if limit r->inf is finite numerically """
    try:
        subs_dict = {}
        for name, sym in vars_dict.items():
            if name not in ['r', 'rp', 'log', 'exp', 'sqrt', 'pi', 'infinity']:
                subs_dict[sym] = 1.0

        # Substitute large r
        subs_dict[r_sym] = 1e10

        val = expr.subs(subs_dict).n()  # with .n() being Sage's numerical evaluation
        return not np.isnan(val) and not np.isinf(val)
    except Exception:
        return False


# ==========================================

@timeout(PROP_TIMEOUT)
def get_enclosed_mass(rho, r, rp):
    integrand = 4 * pi * rp**2 * rho.subs({r: rp})
    return integral(integrand, rp, 0, r).simplify_full(), integrand


@timeout(PROP_TIMEOUT)
def get_mass_limit(M_r, r):
    return limit(M_r, r=infinity)


def get_circular_velocity(M_r, r, G):
    return sqrt(G * M_r / r).simplify_full()


@timeout(PROP_TIMEOUT)
def get_potential(M_r, r, rp, G, use_finite_ref=False):
    integrand = G * M_r.subs({r: rp}) / rp**2
    if use_finite_ref:
        return integral(integrand, rp, 0, r).simplify_full(), integrand, 0, r
    else:
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
            assume(v > 0)           # alpha usually controls sharpness of transition (must be > 0)

        if name == 'beta':
            assume(v > 0)          # beta controls outer slope, let's assume > 0 to start

        if name == 'gamma':
            # assume(v > -1)
            assume(v < 3)          # gamma controls inner slope (cusp), typically < 3 to avoid infinite mass

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

    # 1. Density
    results.append({'Property': 'Density', 'symb_condition': True, 'num_condition': True, 'symb_result': rho})

    # 2. Enclosed Mass
    M_r = None
    mass_int_expr = None
    try:
        M_r, mass_int_expr = get_enclosed_mass(rho, r, rp)
        symb = not str(M_r).count('integral')
    except Exception:
        symb = False

    # Numeric Check: Integrate density from 0 to 1
    if mass_int_expr is not None:
        num = check_numeric_integrability(mass_int_expr, 1.0, 0, 1.0, vars_dict, rp)
    else:
        num = False

    results.append({'Property': 'Enclosed Mass', 'symb_condition': symb, 'num_condition': num, 'symb_result': M_r})

    # 3. Mass Limit
    M_lim = None
    try:
        if M_r is not None:
            M_lim = get_mass_limit(M_r, r)
            symb = not str(M_lim).count('limit')
            # Numeric: Check if M_r(1e10) is finite
            num = check_numeric_limit(M_r, r, vars_dict)
        else:
            symb, num = False, False
    except Exception:
        symb, num = False, False

    results.append({'Property': 'Enclosed Mass Limit', 'symb_condition': symb, 'num_condition': num, 'symb_result': M_lim})

    # 4. Circular Velocity
    V_r = None
    if M_r is not None:
        V_r = get_circular_velocity(M_r, r, G)
        symb = True
        num = True  # Algebraic, always true if Mass exists
    else:
        symb, num = False, False
    results.append({'Property': 'Circular Velocity', 'symb_condition': symb, 'num_condition': num, 'symb_result': V_r})

    # # 5. Potential (Inf Ref)
    # Phi_inf = None
    # pot_int_expr = None
    # try:
    #     if M_r is not None:
    #         Phi_inf, pot_int_expr, low, high = get_potential(M_r, r, rp, G, False)
    #         symb = not str(Phi_inf).count('integral')
    #         # Num: Integrate force from 1 to infinity
    #         num = check_numeric_integrability(pot_int_expr, 1.0, 1.0, infinity, vars_dict, rp)
    #     else:
    #         symb, num = False, False
    # except Exception:
    #     symb, num = False, False
    # results.append({'Property': 'Potential (at Inf)', 'symb_condition': symb, 'num_condition': num, 'symb_result': Phi_inf})

    # 6. Potential (Zero Ref)
    Phi_z = None
    pot_z_int_expr = None
    try:
        if M_r is not None:
            Phi_z, pot_z_int_expr, low, high = get_potential(M_r, r, rp, G, True)
            symb = not str(Phi_z).count('integral')
            # Num: Integrate force from 0 to 1
            num = check_numeric_integrability(pot_z_int_expr, 1.0, 0, 1.0, vars_dict, rp)
        else:
            symb, num = False, False
    except Exception:
        symb, num = False, False
    results.append({'Property': 'Potential', 'symb_condition': symb, 'num_condition': num, 'symb_result': Phi_z})

    # 7. Velocity Dispersion
    Sigma2 = None
    sig_int_expr = None
    try:
        if M_r is not None:
            Sigma2, sig_int_expr = get_velocity_dispersion(rho, M_r, r, rp, G)
            symb = not str(Sigma2).count('integral')
            # Num: Integrate Jeans from 1 to infinity
            num = check_numeric_integrability(sig_int_expr, 1.0, 1.0, infinity, vars_dict, rp)
        else:
            symb, num = False, False
    except Exception:
        symb, num = False, False
    results.append({'Property': 'Radial Velocity Dispersion', 'symb_condition': symb, 'num_condition': num, 'symb_result': Sigma2})

    return pd.DataFrame(results)


# ==========================================
# 5. EXECUTION
# ==========================================

# profiles = {
#     'NFW': "rho0 / ((r / Rs) * (1 + r / Rs)**2)",
#     'pISO': "rho0 / (1 + (r / Rs)**2)",
#     'pISO1': "1 / (1 + (r / 1)**2)",
#     'Lucky13': "rho0 / (1 + (r / Rs))**3",
#     'Burkert': "rho0 / ((1 + (r / Rs))*(1 + (r / Rs)**2))",
#     'superNFW': "rho0 / ((r / Rs) * (1 + r / Rs)**Rational(5, 2))",
#     'Exponential': "rho0*exp(-r/(Rs))",
#     'Exponential1': "9.6*exp(-r/(1.4))",
#     'Exponential2': "rho0*exp(-r/(Rs_1+Rs_2))",
# }
# const_names = ["Rs", "rho0", "Rs_1", "Rs_2"]

profiles = {
    "NFW": ("rho0 / ((r / Rs) * (1 + r / Rs)**2)", ['rho0', 'Rs']),
    "superNFW": ("rho0 / ((r / Rs) * (1 + r / Rs)**Rational(5, 2))", ['rho0', 'Rs']),
    "pISO": ("rho0 / (1 + (r / Rs)**2)", ['rho0', 'Rs']),
    "pISO1": ("1 / (1 + (r/1)**2)", []),
    "Burkert": ("rho0 * Rs**3 / ((r + Rs)*(r**2 + Rs**2))", ['rho0', 'Rs']),
    "Lucky13": ("rho0 / (1 + (r/Rs))**3", ['rho0', 'Rs']),
    # "Einasto": ("rho0 * exp(-2/alpha * ((r/Rs)**alpha - 1))", ['rho0', 'Rs', 'alpha']),
    "coreEinasto": ("rho0 * exp(-2/alpha * ((r/Rs + rc/Rs)**alpha - 1))", ['rho0', 'Rs', 'alpha', 'rc']),
    "DiCintio": ("rho0/((r/Rs)**alpha * (1+(r/Rs)**(1/beta))**(beta*(gamma-alpha)))", ['rho0', 'Rs', 'alpha', 'beta', 'gamma']),
    "gNFW": ("rho0 / ((r/Rs)**gamma * (1 + r/Rs)**(3-gamma))", ['rho0', 'Rs', 'gamma']),
    "Dekel-Zhao": ("rho0 / ((r/Rs)**alpha * (1 + (r/Rs)**(1/2))**(7-2*alpha))", ['rho0', 'Rs', 'alpha']),
    "Exponential": ("rho0 * exp(-r/Rs)", ['rho0', 'Rs']),
    "Exponential1": ("9.6 * exp(-r/1.4)", []),
    "Exponential2": ("rho0 * exp(-r/(Rs_1 + Rs_2))", ['rho0', 'Rs_1', 'Rs_2']),
}

print("Generating Analytical Solutions...")
all_results = {}

# for name, prof_str in profiles.items():
for name, (prof_str, const_names) in profiles.items():
    print(f"\n=== {name} ===")
    df = check_analytical_properties(prof_str, const_names)
    print(df[['Property', 'symb_condition', 'num_condition']])
    all_results[name] = df

# --- Export to Python File ---
print("\nWriting 'derived_profiles.py'...")

with open("derived_profiles.py", "w") as f:
    f.write("import numpy as np\n")
    f.write("import scipy.special\n\n")
    f.write("# Helper wrapper for Sage compatibility\n")
    f.write("def dilog(x):\n    return scipy.special.spence(1 - x)\n\n")

    for name, df in all_results.items():
        f.write(f"# ================= {name} =================\n")

        # Iterate over properties
        for _, row in df.iterrows():
            if row['symb_condition'] and row['symb_result'] is not None:
                prop_name = row['Property'].replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").lower()

                # Special naming for potentials
                if "wo_ref" in prop_name:
                    func_name = f"{name}_potential_inf"
                elif prop_name == "potential":
                    func_name = f"{name}_potential_zero"
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
