"""
Solution evaluation and repair logic for Tabu Search.
Handles production simulation, inventory, backlog, and learning curves.
"""

import copy
from collections import defaultdict
from typing import Dict, Set, Any


class SolutionEvaluator:
    """Evaluates and repairs solutions, computing costs and feasibility."""

    def __init__(
        self, input_data, cap_map: Dict[str, Set[str]], discount_alpha: float
    ):
        """
        Parameters
        ----------
        input_data : InputData
            Problem data
        cap_map : Dict[str, Set[str]]
            Line -> allowed styles mapping
        discount_alpha : float
            Discount rate for time value
        """
        self.input = input_data
        self.cap_map = cap_map
        self.alpha = discount_alpha
        self.precomputed = self._precompute_data()

    def _discount(self, t: int) -> float:
        """Calculate discount factor for period t."""
        return 1.0 / (1.0 + self.alpha) ** t

    def _random_allowed_style(self, line: str) -> str:
        """Get a random allowed style for a line."""
        import random
        return random.choice(list(self.cap_map[line]))

    def _is_allowed(self, line: str, style: str) -> bool:
        """Check if line can produce style."""
        return style in self.cap_map[line]

    def _precompute_data(self) -> Dict:
        """Precompute frequently accessed data."""
        precomputed = {"style_sam": {}, "line_capacity": {}}

        for s in self.input.set["setS"]:
            precomputed["style_sam"][s] = self.input.param["paramSAM"][s]

        for l in self.input.set["setL"]:
            precomputed["line_capacity"][l] = [
                self.input.param["paramH"][(l, t)]
                * 60
                * self.input.param["paramN"][l]
                for t in self.input.set["setT"]
            ]

        return precomputed

    def initialize_solution(self) -> Dict:
        """
        Create initial solution by assigning each line to its
        highest-demand allowed style.
        """
        solution = {"assignment": {}}

        for l in self.input.set["setL"]:
            allowed = self.cap_map[l]
            demands = {
                s: sum(
                    self.input.param["paramD"].get((s, t), 0)
                    for t in self.input.set["setT"]
                )
                for s in allowed
            }
            initial_style = max(demands, key=demands.get)
            
            for t in self.input.set["setT"]:
                solution["assignment"][(l, t)] = initial_style

        return self.repair_and_evaluate(solution)

    def get_efficiency(self, exp_days: float) -> float:
        """
        Get efficiency from learning curve via linear interpolation.

        Parameters
        ----------
        exp_days : float
            Experience in days

        Returns
        -------
        float
            Efficiency factor (0 to 1)
        """
        curve = [
            (self.input.param["paramXp"][p], self.input.param["paramFp"][p])
            for p in self.input.set["setBP"]
        ]

        if exp_days <= curve[0][0]:
            return curve[0][1]
        if exp_days >= curve[-1][0]:
            return curve[-1][1]

        for i in range(len(curve) - 1):
            x1, y1 = curve[i]
            x2, y2 = curve[i + 1]
            if x1 <= exp_days <= x2:
                return y1 + (y2 - y1) * (exp_days - x1) / (x2 - x1)

        return curve[-1][1]

    def repair_and_evaluate(self, solution: Dict) -> Dict:
        """
        Main evaluation function: repairs infeasible assignments and
        simulates production to compute total cost.

        Parameters
        ----------
        solution : Dict
            Solution with 'assignment' key (line, time) -> style

        Returns
        -------
        Dict
            Complete solution with production, costs, backlog, etc.
        """
        # Repair capability violations
        for (l, t), s in list(solution["assignment"].items()):
            if not self._is_allowed(l, s):
                solution["assignment"][(l, t)] = self._random_allowed_style(l)

        move_type = solution.get("type")

        # Initialize solution tracking
        solution.update(
            {
                "production": {},
                "shipment": {},
                "changes": {},
                "experience": {},
                "efficiency": {},
            }
        )

        # Initialize inventories and backlog
        inv_fab = defaultdict(
            float, copy.deepcopy(self.input.param["paramI0fabric"])
        )
        inv_prod = defaultdict(
            float, copy.deepcopy(self.input.param["paramI0product"])
        )
        backlog = copy.deepcopy(self.input.param["paramB0"])

        setup_cost = late_cost = exp_reward = 0.0

        # Line states (current style, experience)
        line_states = {
            l: dict(
                current_style=self._get_initial_style(l),
                exp=self.input.param["paramExp0"].get(l, 0),
                up_exp=0,
            )
            for l in self.input.set["setL"]
        }

        # Production history for lead time lookback
        daily_prod_history = defaultdict(lambda: defaultdict(float))

        # Main simulation loop
        for t in sorted(self.input.set["setT"]):
            # Fabric arrivals
            for s in self.input.set["setS"]:
                LT_f = self.input.param["paramTfabprocess"][s]
                inv_fab[s] += self.input.param["paramF"].get(
                    (s, t - LT_f), 0
                )

            # Production decisions
            pot_prod = {s: [] for s in self.input.set["setS"]}

            for l in self.input.set["setL"]:
                st = line_states[l]
                st["exp"] += st["up_exp"]  # Carry over yesterday's update

                new_style = solution["assignment"][(l, t)]
                work_day = self.input.param["paramH"].get((l, t), 0) > 0

                # Style change & setup cost
                if st["current_style"] != new_style:
                    solution["changes"][
                        (l, st["current_style"], new_style, t)
                    ] = 1
                    setup_cost += (
                        self.input.param["Csetup"] * self._discount(t)
                    )

                    # Reset experience if not same family
                    if (
                        st["current_style"],
                        new_style,
                    ) not in self.input.set["setSsame"]:
                        st["exp"] = self.input.param["paramLexp"][
                            l, new_style
                        ]

                # Record experience & efficiency
                solution["experience"][(l, t)] = st["exp"]
                eff = self.get_efficiency(st["exp"])
                solution["efficiency"][(l, t)] = eff

                # Experience reward
                exp_reward += st["exp"] * self.input.param["Rexp"]

                # Potential capacity on working day
                if (
                    work_day
                    and self.precomputed["style_sam"].get(new_style, 0) > 0
                ):
                    cap_min = self.precomputed["line_capacity"][l][t - 1]
                    sam = self.precomputed["style_sam"][new_style]
                    max_p = (cap_min * eff) / sam
                    pot_prod[new_style].append({"line": l, "max_p": max_p})
                    st["up_exp"] = 0  # Default, may change
                else:
                    st["up_exp"] = 0

                st["current_style"] = new_style

            # Realize production
            for s, items in pot_prod.items():
                total_cap = sum(i["max_p"] for i in items)
                actual_p = min(total_cap, inv_fab[s])

                daily_prod_history[s][t] = actual_p
                inv_fab[s] -= actual_p

                # Split across lines proportionally
                if total_cap > 0:
                    for i in items:
                        share = actual_p * i["max_p"] / total_cap
                        solution["production"][(i["line"], s, t)] = share

                        # Experience bump if worked enough
                        if share >= 0.5 * i["max_p"]:
                            line_states[i["line"]]["up_exp"] = 1

            # Shipments & backlog
            for s in self.input.set["setS"]:
                LT_p = self.input.param["paramTprodfinish"][s]
                finished = daily_prod_history[s].get(t - LT_p, 0.0)
                inv_prod[s] += finished

                to_ship = backlog[s] + self.input.param["paramD"].get(
                    (s, t), 0
                )
                ship_qty = min(inv_prod[s], to_ship)

                solution["shipment"][(s, t)] = ship_qty
                inv_prod[s] -= ship_qty
                backlog[s] = to_ship - ship_qty

                if backlog[s] > 1e-6:
                    late_cost += (
                        backlog[s]
                        * self.input.param["Plate"][s]
                        * self._discount(t)
                    )

        # Finalize solution
        solution.update(
            {
                "final_backlog": backlog,
                "total_setup": setup_cost,
                "total_late": late_cost,
                "total_exp": exp_reward,
                "total_cost": setup_cost + late_cost - exp_reward,
            }
        )

        if move_type:
            solution["type"] = move_type

        return solution

    def _get_initial_style(self, line: str) -> str | None:
        """Get initial style assignment for a line from paramY0."""
        if "paramY0" in self.input.param:
            for s in self.input.set["setS"]:
                if self.input.param["paramY0"].get((line, s), 0) == 1:
                    return s
        return None
