import numpy as np
import pandas as pd
import json
import ast


def extract_parameters(expr: str, allowed_params: set[str]):
    tree = ast.parse(expr, mode="eval")
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names & allowed_params


def print_(verbose: bool, arg: any):
    if verbose:
        print(arg)


class Parser:
    def __init__(self, in_path: str, verbose: bool = True):
        """ Load SR.log into memory and format it into Python dict."""

        # Class parameters
        self.sr_log: dict = {}
        self.sorted_rewards: np.ndarray = None
        self.physical_expr: np.ndarray = None

        # Load .csv
        print_(verbose, f"Loading {in_path}...")
        csv_log = pd.read_csv(in_path)
        print_(verbose, "Finished loading.")
        allowed_parameters = set([c for c in csv_log.columns[8:]] + ['rho0', 'rs0'])
        all_epochs = csv_log['epoch']
        all_expressions = csv_log['program']
        all_rewards = csv_log['reward']
        all_physicals = csv_log['is_physical']

        arg_rewards = np.empty(len(all_expressions))
        physical_mask = np.zeros(len(all_expressions), dtype=np.int8)

        for i, expr in enumerate(all_expressions):
            exp_dict = {}
            name = f"{i}"
            params = list(extract_parameters(expr, allowed_parameters))
            exp_dict['expr'] = expr
            exp_dict['params'] = params
            exp_dict['epoch'] = int(all_epochs[i])
            exp_dict['is_physical'] = bool(all_physicals[i])
            exp_dict['reward'] = float(all_rewards[i])

            self.sr_log[name] = exp_dict
            arg_rewards[i] = all_rewards[i]
            if all_physicals[i]:
                physical_mask[i] = 1.0
        # Argsort rewards
        self.sorted_rewards = np.argsort(arg_rewards)
        print_(verbose, f">>> Highest reward = {all_rewards[self.sorted_rewards[-1]]}")

        # Get physical indices
        physical_indices = np.arange(len(all_expressions))[physical_mask.astype(np.bool)]
        self.physical_expr = np.array([id for id in self.sorted_rewards if id in physical_indices])
        print_(verbose, f">>> Found {len(self.physical_expr)} physical expressions")

    def get_most_rewarded(self, n: int) -> list[dict]:
        """ Returns a list of n-most rewarded expressions as a list of dict:

            {
                'expr': str,
                'params': list[str],
                'epoch': int,
                'is_physical': bool,
                'reward': float
            }
        """
        sorted_list = []
        for i in self.sorted_rewards[-n:]:
            sorted_list.append(self.sr_log[str(i)])
        return sorted_list

    def get_physical_expr(self, n: int = None) -> dict:
        """ Returns a sorted list of all physical expressions by reward:

            {
                'expr': str,
                'params': list[str],
                'epoch': int,
                'is_physical': bool = True,
                'reward': float
            }

            If n is not None, truncates to the n-most rewarded physical expressions.
        """
        sorted_physical_expr = self.physical_expr
        if n is not None:
            sorted_physical_expr = sorted_physical_expr[-n:]

        sorted_list = []
        for i in sorted_physical_expr:
            sorted_list.append(self.sr_log[str(i)])
        return sorted_list

    def dump_log(self, out_path: str):
        """ Dump dict into json."""
        assert self.sr_log is not None

        with open(out_path, "w") as f:
            json.dump(self.sr_log, f, indent=4)


if __name__ == "__main__":

    IN_PATH = "./SR.log"
    OUT_PATH = "./SR.json"

    parser = Parser(IN_PATH)
    parser.dump_log(OUT_PATH)
