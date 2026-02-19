import os
import re
import sys
import pandas as pd
from sage.all import *
from parser import Parser
import numpy as np
import multiprocessing as mp
import traceback


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
    s = re.sub(r'hypergeometric\(\(([^,()]+)\s*,\s*([^,()]+)\),\s*\(([^,()]+),?\),\s*([^)]+)\)', r'scipy.special.hyp2f1(\1, \2, \3, \4)', s)

    return s


def run_with_timeout(func, args=(), kwargs=None, timeout=10):
    """
    Run func(*args, **kwargs) in a separate process and kill it after `timeout` seconds.
    Returns (success, result_or_error_string).
    """
    if kwargs is None:
        kwargs = {}

    def worker(q, *a, **k):
        try:
            res = func(*a, **k)
            q.put(("ok", res))
        except Exception:
            q.put(("err", traceback.format_exc()))

    q = mp.Queue()
    p = mp.Process(target=worker, args=(q, *args), kwargs=kwargs)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return False, f"Timeout after {timeout} s"

    if q.empty():
        return False, "No result returned (crash or kill)."

    status, payload = q.get()
    return (True, payload) if status == "ok" else (False, payload)


# ==========================================
# PHYSICAL PROPS FUNCTIONS
# ==========================================

def get_enclosed_mass(rho, r, rp):
    integrand = 4 * pi * rp**2 * rho.subs({r: rp})
    return integral(integrand, rp, 0, r).simplify_full(), integrand


def get_density(M_r, r, rp):
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

def check_analytical_properties(input_str, free_consts_names, input_type='density', timeout=10):
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
        ok, res = run_with_timeout(get_enclosed_mass, args=(rho, r, rp), timeout=timeout)
        if ok:
            M_r, _ = res
            symb = is_symbolic(M_r)
        else:
            M_r = None
            symb = False
        results.append({'Property': 'Enclosed Mass', 'symb_condition': symb, 'symb_result': M_r})
    elif input_type == 'enclosed_mass':
        M_r = expr
        ok, res = run_with_timeout(get_density, args=(M_r, r, rp), timeout=timeout)
        if ok:
            rho = res
            symb = is_symbolic(rho)
        else:
            rho = None
            symb = False
        results.append({'Property': 'Density', 'symb_condition': symb, 'symb_result': rho})
    else:
        print("input_type must be 'density' or 'enclosed_mass'")
        return pd.DataFrame()

    # Circular Velocity
    V_r = None
    if M_r is not None:
        ok, res = run_with_timeout(get_circular_velocity, args=(M_r, r, G), timeout=timeout)
        if ok:
            V_r = res
            symb = is_symbolic(V_r)
        else:
            V_r = None
            symb = False
    else:
        symb = False
    results.append({'Property': 'Circular Velocity', 'symb_condition': symb, 'symb_result': V_r})

    # Velocity Dispersion
    Sigma2 = None
    if M_r is not None:
        ok, res = run_with_timeout(get_velocity_dispersion, args=(rho, M_r, r, rp, G), timeout=timeout)
        if ok:
            Sigma2, _ = res
            symb = is_symbolic(Sigma2)
        else:
            Sigma2 = None
            symb = False
    else:
        symb = False
    results.append({'Property': 'Radial Velocity Dispersion', 'symb_condition': symb, 'symb_result': Sigma2})

    # Potential
    Phi_inf = None
    if M_r is not None:
        ok, res = run_with_timeout(get_potential, args=(M_r, r, rp, G), timeout=timeout)
        if ok:
            Phi_inf, _, _, _ = res
            symb = is_symbolic(Phi_inf)
        else:
            Phi_inf = None
            symb = False
    else:
        symb = False
    results.append({'Property': 'Potential', 'symb_condition': symb, 'symb_result': Phi_inf})

    # Surface density Σ(R)
    Sigma_R = None
    if rho is not None:
        ok, res = run_with_timeout(get_surface_density, args=(rho, r, R, rp), timeout=timeout)
        if ok:
            Sigma_R, _ = res
            symb = is_symbolic(Sigma_R)
        else:
            Sigma_R = None
            symb = False
    else:
        symb = False
    results.append({'Property': 'Surface Density', 'symb_condition': symb, 'symb_result': Sigma_R})

    # Average surface density \barΣ(<R)
    avg_Sigma_R = None
    if Sigma_R is not None:
        ok, res = run_with_timeout(get_average_surface_density, args=(Sigma_R, R, Rp), timeout=timeout)
        if ok:
            avg_Sigma_R, _ = res
            symb = is_symbolic(avg_Sigma_R)
        else:
            avg_Sigma_R = None
            symb = False
    else:
        symb = False

    results.append({'Property': 'Average Surface Density', 'symb_condition': symb, 'symb_result': avg_Sigma_R})

    return pd.DataFrame(results)
