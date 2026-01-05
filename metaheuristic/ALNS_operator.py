import copy
from collections import defaultdict
import random
import numpy as np

class ALNSOperator:
    """
    Evaluator & Repairer tích hợp Bitmask và Fast Fail.
    """

    def __init__(self, input_data, cap_map, discount_alpha):
        self.input = input_data
        self.raw_cap_map = cap_map  # Giữ bản gốc để tham chiếu nếu cần
        self.alpha = discount_alpha
        
        # --- BITMASK SETUP ---
        # 1. Map Style name <-> Integer ID
        all_styles = sorted(list(self.input.set['setS']))
        self.style_to_id = {name: i for i, name in enumerate(all_styles)}
        self.id_to_style = {i: name for i, name in enumerate(all_styles)}
        
        # 2. Tạo Bitmask và Cached List cho từng Line
        self.line_masks = {}       # Dùng để check _is_allowed cực nhanh
        self.line_allowed_ids = {} # Dùng để random cực nhanh
        
        for l in self.input.set['setL']:
            mask = 0
            allowed_ids = []
            for s_name in self.raw_cap_map[l]:
                if s_name in self.style_to_id:
                    s_id = self.style_to_id[s_name]
                    mask |= (1 << s_id) # Bật bit tại vị trí ID
                    allowed_ids.append(s_id)
            
            self.line_masks[l] = mask
            self.line_allowed_ids[l] = allowed_ids

        # --- FAST FAIL SETUP ---
        self.pruning_cutoff = float('inf')

        self.precomputed = self._precompute_data()

    def set_pruning_best(self, best_cost):
        """
        Cập nhật chi phí tốt nhất hiện tại để làm ngưỡng cắt tỉa.
        Cho phép nới lỏng 20% (factor 1.2) để giữ tính đa dạng.
        """
        if best_cost == float('inf'):
            self.pruning_cutoff = float('inf')
        else:
            self.pruning_cutoff = best_cost * 1.2

    def _precompute_data(self):
        precomputed = {'style_sam': {}, 'line_capacity': {}}
        for s in self.input.set['setS']:
            precomputed['style_sam'][s] = self.input.param['paramSAM'][s]
        for l in self.input.set['setL']:
            precomputed['line_capacity'][l] = [
                self.input.param['paramH'].get((l, t), 0) * 60 * self.input.param['paramN'][l]
                for t in self.input.set['setT']
            ]
        return precomputed

    def _discount(self, t: int) -> float:
        return 1.0 / (1.0 + self.alpha) ** t

    def _is_allowed(self, line, style):
        """Kiểm tra Bitwise: Nhanh hơn Set lookup."""
        s_id = self.style_to_id.get(style)
        if s_id is None: return False
        return (self.line_masks[line] & (1 << s_id)) != 0

    def _random_allowed_style(self, line):
        """Random trên list ID số nguyên: Nhanh hơn random trên list string."""
        allowed_ids = self.line_allowed_ids.get(line)
        if not allowed_ids: return None
        s_id = random.choice(allowed_ids)
        return self.id_to_style[s_id]

    def _get_initial_style(self, line):
        if 'paramY0' in self.input.param:
            for s in self.input.set['setS']:
                if self.input.param['paramY0'].get((line, s), 0) == 1:
                    return s
        return None

    def get_efficiency(self, exp_days):
        curve = [(self.input.param['paramXp'][p], self.input.param['paramFp'][p]) 
                 for p in self.input.set['setBP']]
        if exp_days <= curve[0][0]:
            return curve[0][1]
        if exp_days >= curve[-1][0]:
            return curve[-1][1]
        for i in range(len(curve) - 1):
            x1, y1 = curve[i]
            x2, y2 = curve[i+1]
            if x1 <= exp_days <= x2:
                return y1 + (y2 - y1) * (exp_days - x1) / (x2 - x1)
        return curve[-1][1]

    def initialize_solution(self):
        solution = {'assignment': {}}
        for l in self.input.set['setL']:
            # Dùng list ID đã cache
            allowed_ids = self.line_allowed_ids[l]
            allowed_styles = [self.id_to_style[i] for i in allowed_ids]
            
            demands = {
                s: sum(self.input.param['paramD'].get((s, t), 0)
                for t in self.input.set['setT'])
                for s in allowed_styles
            }
            if not demands:
                initial_style = self._random_allowed_style(l)
            else:
                initial_style = max(demands, key=demands.get)
            
            for t in self.input.set['setT']:
                solution['assignment'][(l, t)] = initial_style
        
        return self.repair_and_evaluate(solution)

    def repair_and_evaluate(self, solution):
        """
        Logic cũ + Tối ưu Bitmask Repair + Tối ưu Fast Fail Evaluation.
        """
        assignment = solution.get('assignment', {})
        
        # --- REPAIR: Fix invalid assignments (Bitmask Optimized) ---
        for (l, t), s in list(assignment.items()):
            if not self._is_allowed(l, s):
                assignment[(l, t)] = self._random_allowed_style(l)
        
        solution['assignment'] = assignment

        # --- EVALUATE: Initialize Simulation State ---
        move_type = solution.get("type")
        solution.update({
            "production":  {},
            "shipment":    {},
            "changes":     {},
            "experience":  {},
            "efficiency":  {}
        })

        inv_fab  = defaultdict(float, copy.deepcopy(self.input.param["paramI0fabric"]))
        inv_prod = defaultdict(float, copy.deepcopy(self.input.param["paramI0product"]))
        backlog  = copy.deepcopy(self.input.param["paramB0"])

        setup_cost = late_cost = exp_reward = 0.0

        line_states = {
            l: dict(current_style=self._get_initial_style(l),
                    exp=self.input.param["paramExp0"].get(l, 0),
                    up_exp=0)
            for l in self.input.set["setL"]
        }

        daily_prod_history = defaultdict(lambda: defaultdict(float))
        
        # Cache local variables for speed loop
        get_sam = self.precomputed["style_sam"].get
        get_line_cap = self.precomputed["line_capacity"]
        param_h = self.input.param["paramH"]
        param_csetup = self.input.param["Csetup"]
        param_lexp = self.input.param["paramLexp"]
        param_rexp = self.input.param["Rexp"]
        param_ssame = self.input.set["setSsame"]
        set_s = self.input.set["setS"]
        set_l = self.input.set["setL"]
        
        # --- MAIN SIMULATION LOOP ---
        for t in sorted(self.input.set["setT"]):
            
            # --- FAST FAIL CHECK ---
            # Nếu chi phí tích lũy đã vượt quá ngưỡng cắt tỉa -> Dừng ngay
            current_est_cost = setup_cost + late_cost - exp_reward
            if current_est_cost > self.pruning_cutoff:
                solution['total_cost'] = float('inf')
                return solution

            # 1. Fabric Receipts
            for s in set_s:
                LT_f = self.input.param["paramTfabprocess"][s]
                inv_fab[s] += self.input.param["paramF"].get((s, t - LT_f), 0)

            # 2. Decide Production
            pot_prod = {s: [] for s in set_s}
            
            # Cache discount factor for this day
            disc_factor = self._discount(t)

            for l in set_l:
                st = line_states[l]
                st["exp"] += st["up_exp"]
                
                new_style = assignment.get((l, t))
                work_day  = param_h.get((l, t), 0) > 0
                
                if new_style is None: 
                    st["up_exp"] = 0
                    continue

                # Style change & setup cost
                if st["current_style"] != new_style:
                    solution["changes"][(l, st["current_style"], new_style, t)] = 1
                    setup_cost += (param_csetup * disc_factor)

                    if (st["current_style"], new_style) not in param_ssame:
                        st["exp"] = param_lexp.get((l, new_style), 0)

                # Record experience & efficiency
                solution["experience"][(l, t)] = st["exp"]
                eff = self.get_efficiency(st["exp"])
                solution["efficiency"][(l, t)] = eff

                # Experience reward
                exp_reward += st["exp"] * param_rexp

                # Potential capacity
                if work_day:
                    sam = get_sam(new_style, 0)
                    if sam > 0:
                        cap_min = get_line_cap[l][t - 1]
                        max_p   = (cap_min * eff) / sam
                        pot_prod[new_style].append({"line": l, "max_p": max_p})
                        st["up_exp"] = 0 
                    else:
                        st["up_exp"] = 0
                else:
                    st["up_exp"] = 0

                st["current_style"] = new_style

            # 3. Realise Production
            for s, items in pot_prod.items():
                if not items: continue # Skip if empty
                total_cap = sum(i["max_p"] for i in items)
                actual_p  = min(total_cap, inv_fab[s])

                daily_prod_history[s][t] = actual_p
                inv_fab[s] -= actual_p

                if total_cap > 0:
                    for i in items:
                        share = actual_p * i["max_p"] / total_cap
                        solution["production"][(i["line"], s, t)] = share
                        if share >= 0.5 * i["max_p"]:
                            line_states[i["line"]]["up_exp"] = 1

            # 4. Shipments & Backlog
            for s in set_s:
                LT_p = self.input.param["paramTprodfinish"][s]
                finished = daily_prod_history[s].get(t - LT_p, 0.0)
                inv_prod[s] += finished

                to_ship = backlog[s] + self.input.param["paramD"].get((s, t), 0)
                ship_qty = min(inv_prod[s], to_ship)

                solution["shipment"][(s, t)] = ship_qty
                inv_prod[s] -= ship_qty
                backlog[s]  = to_ship - ship_qty

                if backlog[s] > 1e-6:
                    late_cost += (backlog[s] * 
                                  self.input.param["Plate"][s] * 
                                  disc_factor)

        # --- FINALIZE ---
        solution.update({
            "final_backlog": backlog,
            "total_setup":   setup_cost,
            "total_late":    late_cost,
            "total_exp":     exp_reward,
            "total_cost":    setup_cost + late_cost - exp_reward
        })

        if move_type:
            solution["type"] = move_type
            
        return solution
