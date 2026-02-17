# sage_checker.py
from RewardedSage import check_analytical_properties


def check_profile(expr_str: str, params: list[str]) -> dict:
    """
    expr_str : PhySO expression string, e.g. '((rs0+-(r))*...)'
    params   : list of parameter names in expr_str, e.g. ['rho0', 'rs1', 'rs0']

    Returns a dict: { 'Density': True/False, 'Enclosed Mass': ..., ... }.
    """
    df = check_analytical_properties(expr_str, free_consts_names=params)
    if df.empty:
        return {}
    return {row["Property"]: bool(row["symb_condition"]) for _, row in df.iterrows()}


expr_str = "((rs0+-(r))*-(((-((rs1/r))/r)/-(((rho0)**(-1))))))"
params = ["rho0", "rs1", "rs0"]

flags = check_profile(expr_str, params)
print(flags)
