from __future__ import annotations
import sympy as sym
from dataclasses import dataclass, field
from typing import Optional
from sympy.parsing.sympy_parser import parse_expr

@dataclass
class CheckResult:
    """ Stores simple test results.
    """
    name: str
    passed: Optional[bool]          # True / False / None (undetermined)
    details: str = ""

    def __repr__(self):
        status = {True: "PASS", False: "FAIL", None: "UNDETERMINED"}[self.passed]
        s = f"[{status}] {self.name}"
        if self.details:
            s += f"\n         {self.details}"
        return s

# Helpers

def _safe_limit(expr: sym.Expr, var: sym.Symbol, point: any, dir="+"):
    """ Safe limit helper.
    """
    try:
        return sym.limit(expr, var, point, dir=dir)
    except Exception:
        return None

def _safe_simplify(expr: sym.Expr):
    """ Safe simplify helper.
    """
    try:
        return sym.simplify(expr)
    except Exception:
        return expr
    
def _safe_derivative(expr: sym.Expr, var: sym.Symbol):
    """ Safe derivative helper.
    """
    try:
        return sym.diff(expr, var)
    except Exception:
        return None
    
def _is_nonneg(expr, var):
    """ Tries to determine whether expr >= 0 for all r > 0.
        Returns True / False / None.
    """
    simplified = _safe_simplify(expr)
    pos = sym.ask(sym.Q.nonnegative(expr.subs(var, sym.Symbol('var', positive=True))))
    if pos is not None:
        return pos
    # check if expression is a sum of squares or manifestly positive
    if simplified.is_nonnegative:
        return True
    if simplified.is_nonpositive:
        return False
    return None

# Positive rho check

def check_non_negativity(rho: sym.Expr, r: sym.Symbol, vals_dict: dict = None) -> CheckResult:
    """ Checks that the density is positive.
    """
    rho_sub = rho
    if vals_dict is not None:
        rho_sub = rho.subs(vals_dict)
        
    result = _is_nonneg(rho_sub, r)
    return CheckResult(
        name="Non-negativity  rho >= 0",
        passed=result,
        details="Could not determine sign symbolically." if result is None else "",
    )

# Finite rho check

def check_finiteness_at_origin(rho: sym.Expr, r: sym.Symbol, vals_dict: dict = None) -> CheckResult:
    """ Checks that rho is finite or at least with mass integrable at the origin.
    """
    rho_sub = rho
    if vals_dict is not None:
        rho_sub = rho.subs(vals_dict)

    # STEP 1 - Check lim_0 rho(r) is finite

    L = _safe_limit(rho_sub, r, 0, dir="+")

    if L is None:
        return CheckResult(
            name="Finiteness at origin",
            passed=None,
            details="Could not evaluate limit at r=0.",
        )
    
    if L.is_finite and L.is_real:
        if L < 0:
            return CheckResult(
                name="Finiteness at origin",
                passed=False,
                details=f"rho(0) = {L} < 0.",
            )
        return CheckResult(
            name="Finiteness at origin",
            passed=True,
            details=f"rho(0) = {L} (finite, non-negative).",
        )
    
    # STEP 2 - Check lim_0 r^2 rho(r) = 0
    
    L = _safe_limit(rho_sub*r**2, r, 0, dir="+")

    if L is not None:
        if L.is_finite and L.is_real:
            return CheckResult(
                name="Finiteness at origin",
                passed=True,
                details="lim_0 rho(r)r^2 converges at the origin."
            )

    # STEP 3 - Compute for rho(r) ~ r^{-\alpha}
    
    try:
        slope = sym.limit(sym.log(sym.Abs(rho_sub)) / sym.log(r), r, 0, dir="+")
    except Exception:
        return CheckResult(
            name="Finiteness at origin",
            passed=None,
            details="rho diverges at origin; could not determine slope.",
        )

    if slope == sym.nan:
        return CheckResult(
            name="Finiteness at origin",
            passed=None, 
            details="rho diverges at origin; could not determine slope.",
        )

    alpha = -slope
    cond = sym.StrictLessThan(alpha, sym.Integer(3))
    cond = _safe_simplify(cond)

    if cond == True:
        return CheckResult(
            name="Finiteness at origin",
            passed=True,
            details=f"Integrable cusp: rho ~ r^(-{alpha}) with {alpha} < 3."
        )
    if cond == False:
        return CheckResult(
            name="Finiteness at origin",
            passed=False,
            details=f"Non-integrable cusp: rho ~ r^(-{alpha}) with {alpha} >= 3."
        )
    return CheckResult(
        name="Finiteness at origin",
        passed=None,
        details=f"Cusp exponent = {alpha}; integrability undetermined. Found condition : {cond}."
    )

# Finite enclosed mass check

def check_finite_mass_at_infinity(rho: sym.Expr, r: sym.Symbol, vals_dict: dict = None) -> CheckResult:
    """ Check that the mass is finite at infinity from a given density profile.
    """
    rho_sub = rho
    if vals_dict is not None:
        rho_sub = rho.subs(vals_dict)

    # STEP 1 - Check lim_inf r^2 rho(r) = 0

    L = _safe_limit(rho_sub*r**2, r, sym.oo, dir="+")

    if L is not None:
        if L.is_infinite:
            return CheckResult(
                name="Finite enclosed mass  M(inf) < inf",
                passed=False,
                details="lim_inf r^2 rho(r) diverges.")
        elif L.is_finite and L.is_real:
            if L != 0 and L.is_number:
                return CheckResult(
                    name="Finite enclosed mass  M(inf) < inf",
                    passed=False,
                    details=f"lim_inf r^2 rho(r) = {L} > 0."
                )
        
    # STEP 2 - Compute for rho(r) ~ r^{-\alpha}

    try:
        slope = sym.limit(sym.log(sym.Abs(rho_sub)) / sym.log(r), r, sym.oo, dir="+")
    except Exception:
        return CheckResult(
            name="Finite enclosed mass  M(inf) < inf",
            passed=None,
            details="rho diverges at infinity; could not determine slope.",
        )
    
    if slope == sym.nan:
        return CheckResult(
            name="Finite enclosed mass  M(inf) < inf", 
            passed=None, 
            details="Failed to determine asymptotic slope."
        )

    alpha = -slope
    cond = sym.StrictGreaterThan(alpha, sym.Integer(3))
    cond = _safe_simplify(cond)

    if cond == True:
        return CheckResult(
            name="Finite enclosed mass  M(inf) < inf", 
            passed=True,
            details=f"Found slope = {alpha}"
        )
    if cond == False:
        return CheckResult(
            name="Finite enclosed mass  M(inf) < inf", 
            passed=False,
            details=f"Found slope = {alpha}"
        )
    return CheckResult(
        name="Finite enclosed mass  M(inf) < inf",
        passed=None,
        details=f"Asymptotic slope = {alpha}; integrability undetermined. Found condition : {cond}.",
        )

# Check mass positivity

def check_mass_positivity_and_monotonicity(rho: sym.Expr, r: sym.Symbol, vals_dict: dict = None) -> list[CheckResult]:
    """ M(r) > 0  for r > 0  (follows from rho >= 0, but checked via dM/dr = 4pi r^2 rho)
        dM/dr >= 0           (mass is non-decreasing)
        Not very reliable for parametric profiles if no assumption are made on the parameters.
    """
    rho_sub = rho
    if vals_dict is not None:
        rho_sub = rho.subs(vals_dict)
        
    dMdr = 4 * sym.pi * r**2 * rho_sub

    pos = _is_nonneg(dMdr, r)
    monotone = CheckResult(
        name="Mass monotonicity  dM/dr >= 0",
        passed=pos,
        details="dM/dr = 4pi r^2 rho — sign inherits from non-negativity of rho.",
    )
    positive = CheckResult(
        name="Mass positivity  M(r) > 0",
        passed=pos,
        details="Follows from dM/dr >= 0 and M(0) = 0.",
    )
    return [positive, monotone]

# Check potential

def check_potential_convergence(rho: sym.Expr, r: sym.Symbol, vals_dict: dict = None) -> CheckResult:
    """ Check that the potential is defined from the expression of the density.
        Test result might be invalid if density positivity was not verified.
    """

    rho_sub = rho
    if vals_dict is not None:
        rho_sub = rho.subs(vals_dict)

    # STEP 1 - Check for finite enclosed mass

    finite_origin = check_finiteness_at_origin(rho, r, vals_dict)
    finite_infinity = check_finite_mass_at_infinity(rho, r, vals_dict)
    
    if finite_origin.passed and finite_infinity.passed:
        return CheckResult(
            name="Potential convergence Phi(r) finite",
            passed=True,
            details="Convergence is defined for finite mass."
        )
    
    if finite_origin.passed is False:
        return CheckResult(
            name="Potential convergence Phi(r) finite",
            passed=False,
            details="Potential requires finite mass at origin."
        )
    
    if finite_infinity.passed and finite_origin.passed is None:
        details = "Convergence is defined for finite mass."
        details += f" WARNING - was unable to determine finiteness at origin : {finite_origin.details}" 
        return CheckResult(
            name="Potential convergence Phi(r) finite",
            passed=True,
            details=details
        )
    
    # STEP 2 - Compute for rho(r) ~ r^{-\alpha}

    try:
        slope = sym.limit(
            sym.log(sym.Abs(rho_sub)) / sym.log(r), r, sym.oo, dir="+"
        )
    except Exception:
        return CheckResult(
            name="Potential convergence  Phi(r) finite",
            passed=None,
            details="Could not determine asymptotic slope."
        )
    
    if slope == sym.nan:
        return CheckResult(
            name="Potential convergence  Phi(r) finite",
            passed=None,
            details="Could not determine asymptotic slope."
        )
    
    alpha = -slope # If alpha = 3 or if alpha > 2, knowing alpha <= 3.
    interval = sym.Interval(2, 3, left_open=True, right_open=False)
    cond = interval.contains(alpha)
    cond = sym.simplify(cond)

    if cond == True:
        details = f"rho ~ r^(-{alpha}); {alpha} in (2, 3] ensures convergence."
        if finite_origin.passed is None:
            details += f" WARNING - was unable to determine finiteness at origin : {finite_origin.details}"
        return CheckResult(
            name="Potential convergence  Phi(r) finite",
            passed=True,
            details=details,
        )

    if cond == False:
        return CheckResult(
            name="Potential convergence  Phi(r) finite",
            passed=False,
            details=f"rho ~ r^(-{alpha}); potential integral diverges (need {alpha} in (2, 3]).",
        )
    return CheckResult(
        name="Potential convergence  Phi(r) finite",
        passed=None,
        details=f"Asymptotic slope alpha = {alpha}; convergence undetermined. Found condition : {cond}",
    )

# Check radial velocity dispersion

def check_jeans_dispersion(rho: sym.Expr, r: sym.Symbol, vals_dict: dict = None) -> CheckResult:
    """ Check that sigma_r^2(r) is defined from the expression of rho.
    """
    rho_sub = rho
    if vals_dict is not None:
        rho_sub = rho.subs(vals_dict)

    # STEP 1 - Check for finite enclosed mass

    finite_origin = check_finiteness_at_origin(rho, r, vals_dict)
    finite_infinity = check_finite_mass_at_infinity(rho, r, vals_dict)

    if finite_origin.passed and finite_infinity.passed:
        return CheckResult(
            name="Jeans dispersion - sigma_r^2 finite",
            passed=True,
            details="Defined from finite enclosed mass."
        )
    
    if finite_origin.passed is False:
        return CheckResult(
            name="Jeans dispersion - sigma_r^2 finite",
            passed=False,
            details="Dispersion requires finite mass at origin."
        )
    
    if finite_infinity.passed and finite_origin.passed is None:
        details = "Defined from finite enclosed mass."
        details += f" WARNING - was unable to determine finiteness at origin : {finite_origin.details}" 
        return CheckResult(
            name="Jeans dispersion - sigma_r^2 finite",
            passed=True,
            details=details
        )

    # STEP 2 - Compute for rho(r) ~ r^{-\alpha}

    try:
        slope = sym.limit(
            sym.log(sym.Abs(rho_sub)) / sym.log(r), r, sym.oo, dir="+"
        )
    except Exception:
        return CheckResult(
            name="Jeans dispersion - sigma_r^2 finite",
            passed=None,
            details="Could not determine asymptotic slope.",
        )
    
    if slope == sym.nan:
        return CheckResult(
            name="Jeans dispersion - sigma_r^2 finite",
            passed=None,
            details="Could not determine asymptotic slope."
        )
    
    alpha = -slope
    interval = sym.Interval(1, 3, left_open=True, right_open=False)
    cond = interval.contains(alpha)
    cond = _safe_simplify(cond)

    if cond == True:
        details = f"rho ~ r^(-{alpha}); {alpha} in (2, 3] ensures convergence."
        if finite_origin.passed is None:
            details += f" WARNING - was unable to determine finiteness at origin : {finite_origin.details}"
        return CheckResult(
            name="Jeans dispersion  sigma_r^2 finite",
            passed=True,
            details=details,
        )
    if cond == False:
        return CheckResult(
            name="Jeans dispersion  sigma_r^2 finite",
            passed=False,
            details=f"rho ~ r^(-{alpha}); potential integral diverges (need {alpha} in (1, 3]).",
        )
    return CheckResult(
        name="Jeans dispersion  sigma_r^2 finite",
        passed=None,
        details=f"Asymptotic slope alpha = {alpha}; convergence undetermined. Found condition : {cond}",
        )

def check_jeans_dispersion_from_mass(M: sym.Expr, r: sym.Symbol, vals_dict: dict) -> CheckResult:
    """ Check that sigma_r^2(r) is defined from the expression of M.
    """
    M_sub = M
    if vals_dict is not None:
        M_sub = M.subs(vals_dict)
    
    dMdr = _safe_derivative(M, r)
    if dMdr is None:
        return CheckResult(
            name="Jeans dispersion  sigma_r^2 finite",
            passed=None,
            details="Failed to compute density."
        )

    if vals_dict is not None:
        dMdr = dMdr.subs(vals_dict)
    
    M_rho = _safe_simplify(dMdr/(4*sym.pi*r**2) * M_sub)

    # STEP 1 - Check lim_inf rho(r)M(r)/r^2 = 0

    L = _safe_limit(M_rho/r**2, r, sym.oo, dir="+")

    if L is not None:
        if L.is_infinite:
            return CheckResult(
                name="Jeans dispersion - sigma_r^2 finite",
                passed=False,
                details="lim_inf M(r)rho(r)/r^2 diverges.")
        elif L.is_finite and L.is_real:
            if L != 0 and L.is_number:
                return CheckResult(
                    name="Jeans dispersion - sigma_r^2 finite",
                    passed=False,
                    details="lim_inf M(r)rho(r)/r^2 != 0."
                )
    
    # STEP 2 - Compute for rho(r) ~ r^{-\alpha}

    try:
        slope = sym.limit(
            sym.log(sym.Abs(M_rho)) / sym.log(r), r, sym.oo, dir="+"
        )
    except Exception:
        return CheckResult(
            name="Jeans dispersion  sigma_r^2 finite",
            passed=None,
            details="Could not determine asymptotic slope.",
        )
    
    if slope == sym.nan:
        return CheckResult(
            name="Jeans dispersion - sigma_r^2 finite",
            passed=None,
            details="Could not determine asymptotic slope."
        )
    
    alpha = -slope
    cond = sym.StrictGreaterThan(alpha, sym.Integer(-1))
    cond = _safe_simplify(cond)

    if cond == True:
        return CheckResult(
            name="Jeans dispersion  sigma_r^2 finite",
            passed=True,
            details=f"rho ~ r^(-{alpha}); alpha < 1 ensures convergence.",
        )
    if cond == False:
        return CheckResult(
            name="Jeans dispersion  sigma_r^2 finite",
            passed=False,
            details=f"rho ~ r^(-{alpha}); potential integral diverges (need alpha < 1).",
        )
    return CheckResult(
        name="Jeans dispersion  sigma_r^2 finite",
        passed=None,
        details=f"Asymptotic slope alpha = {alpha}; convergence undetermined. Found condition : {cond}",
        )

def check_surface_density(rho: sym.Expr, r: sym.Symbol, vals_dict: dict = None) -> CheckResult:
    """ Check that that the surface density Sigma(R) is defined.
    """
    rho_sub = rho
    if vals_dict is not None:
        rho_sub = rho.subs(vals_dict)

    # STEP 1 - Compute for rho(r) ~ r^{-\alpha}

    try:
        slope = sym.limit(
            sym.log(sym.Abs(rho_sub)) / sym.log(r), r, sym.oo, dir="+"
        )
    except Exception:
        return CheckResult(
            name="Surface density - Sigma(R) finite",
            passed=None,
            details="Could not determine asymptotic slope.",
        )
    
    if slope == sym.nan:
        return CheckResult(
            name="Surface density - Sigma(R) finite",
            passed=None,
            details="Could not determine asymptotic slope."
        )
    
    alpha = -slope
    cond = sym.StrictGreaterThan(alpha, sym.Integer(1))
    cond = _safe_simplify(cond)

    if cond == True:
        return CheckResult(
            name="Surface density - Sigma(R) finite",
            passed=True,
            details=f"rho ~ r^(-{alpha}) with {alpha} > 1; surface density integral converges.",
        )
    if cond == False:
        return CheckResult(
            name="Surface density - Sigma(R) finite",
            passed=False,
            details=f"rho ~ r^(-{alpha}); surface density integral integral diverges (need {alpha} > 1).",
        )
    return CheckResult(
        name="Surface density - Sigma(R) finite",
        passed=None,
        details=f"Asymptotic slope alpha = {alpha}; convergence undetermined. Found condition : {cond}",
        )

def check_mean_surface_density(rho: sym.Expr, r: sym.Symbol, vals_dict: dict = None) -> CheckResult:
    """ Check that that the mean surface density is defined.
    """
    rho_sub = rho
    if vals_dict is not None:
        rho_sub = rho.subs(vals_dict)

    try:
        slope = sym.limit(sym.log(sym.Abs(rho_sub)) / sym.log(r), r, 0, dir="+")
    except Exception:
        return CheckResult(
            name="Mean surface density finite",
            passed=None,
            details="rho diverges at origin; could not determine slope.",
        )

    if slope == sym.nan:
        return CheckResult(
            name="Mean surface density finite",
            passed=None, 
            details="rho diverges at origin; could not determine slope.",
        )

    alpha = -slope
    cond = sym.StrictLessThan(alpha, sym.Integer(3))
    cond = _safe_simplify(cond)

    if cond == True:
        return CheckResult(
            name="Mean surface density finite",
            passed=True,
            details=f"Integrable cusp: rho ~ r^(-{alpha}) with {alpha} < 3."
        )
    if cond == False:
        return CheckResult(
            name="Mean surface density finite",
            passed=False,
            details=f"Non-integrable cusp: rho ~ r^(-{alpha}) with {alpha} >= 3."
        )
    return CheckResult(
        name="Mean surface density finite",
        passed=None,
        details=f"Cusp exponent = {alpha}; integrability undetermined. Found condition : {cond}."
    )

    

if __name__=="__main__":

    profiles = {
        "NFW": ("rho0 / ((r / Rs) * (1 + r / Rs)**2)", ['rho0', 'Rs']),
        "superNFW": ("rho0 / ((r / Rs) * (1 + r / Rs)**Rational(5, 2))", ['rho0', 'Rs']),
        "pISO": ("rho0 / (1 + (r / Rs)**2)", ['rho0', 'Rs']),
        "pISO1": ("1 / (1 + (r/1)**2)", []),
        "Burkert": ("rho0 * Rs**3 / ((r + Rs)*(r**2 + Rs**2))", ['rho0', 'Rs']),
        "Lucky13": ("rho0 / (1 + (r/Rs))**3", ['rho0', 'Rs']),
        "Einasto": ("rho0 * exp(-2/a * ( (r/Rs)**a - 1 ) )", ['rho0', 'Rs', 'a']),
        "coreEinasto": ("rho0 * exp(-2/a * ((r/Rs + rc/Rs)**a - 1))", ['rho0', 'Rs', 'a', 'rc']),
        "DiCintio": ("rho0 / ( (r/Rs)**a * (1+(r/Rs)**(1/b))**(b*(g-a)))", ['rho0', 'Rs', 'a', 'b', 'g']),
        "gNFW": ("rho0 / ((r/Rs)**g * (1 + r/Rs)**(3-g))", ['rho0', 'Rs', 'g']),
        "Dekel-Zhao": ("rho0 / ((r/Rs)**a * (1 + (r/Rs)**(1/2))**(7-2*a))", ['rho0', 'Rs', 'a']),
        "Exponential": ("rho0 * exp(-r/Rs)", ['rho0', 'Rs']),
        "Exponential1": ("9.6 * exp(-r/1.4)", []),
        "Exponential2": ("rho0 * exp(-r/(Rs_1 + Rs_2))", ['rho0', 'Rs_1', 'Rs_2']),
    }

    r = sym.symbols('r', positive=True, real=True)
    rho0 = sym.symbols('rho0', positive=True, real=True)
    Rs, Rs_1, Rs_2 = sym.symbols('Rs Rs_1 Rs_2', positive=True, real=True)
    rc = sym.symbols('rc', positive=True, real=True)
    a, b, g = sym.symbols('a b g', positive=True, real=True)

    local_dict = {
        'r': r, 'rho0': rho0, 'Rs': Rs, 'Rs_1': Rs_1,
        'Rs_2': Rs_2, 'rc': rc, 'a': a, 'b': b, 'g': g
    }

    with open("results.txt", "w") as f:

        for rho_name in profiles:
            f.write(f"### {rho_name} ###")
            f.write('\n')
            rho_expr = sym.simplify(parse_expr(profiles[rho_name][0], local_dict=local_dict))

            f.write(str(check_non_negativity(rho_expr, r)))
            f.write('\n')
            f.write(str(check_finiteness_at_origin(rho_expr, r)))
            f.write('\n')
            f.write(str(check_finite_mass_at_infinity(rho_expr, r)))
            f.write('\n')
            for i in range(2):
                f.write(str(check_mass_positivity_and_monotonicity(rho_expr, r)[i]))
                f.write('\n')
            f.write(str(check_potential_convergence(rho_expr, r)))
            f.write('\n')
            f.write(str(check_jeans_dispersion(rho_expr, r)))
            f.write('\n')
            f.write(str(check_surface_density(rho_expr, r)))
            f.write('\n')
            f.write(str(check_mean_surface_density(rho_expr, r)))
            f.write('\n')
            f.write('\n')