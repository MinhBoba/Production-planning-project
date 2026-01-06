import copy
from collections import defaultdict
import random
import numpy as np

class ALNSOperator:
    """
    Evaluator & Repairer tối ưu:
    - Integer ID Mapping: Tăng tốc độ so sánh và random.
    - Fast Fail: Cắt tỉa các nhánh kém chất lượng sớm.
    - Look-ahead Logic: Xử lý thông minh việc chuyển đổi khi thiếu nguyên liệu.
    """

    def __init__(self, input_data, cap_map, discount_alpha):
        self.input = input_data
        self.alpha = discount_alpha
        
        # --- 1. Map String <-> Integer ID ---
        all_styles = sorted(list(self.input.set['setS']))
        self.style_to_id = {name: i for i, name in enumerate(all_styles)}
        self.id_to_style = {i: name for i, name in enumerate(all_styles)}
        
        # --- 2. Cache Capability dưới dạng ID ---
        self.line_allowed_sets = {} # Dùng để check O(1)
        self.line_allowed_lists = {} # Dùng để random O(1)
        
        for l in self.input.set['setL']:
            ids = [self.style_to_id[s] for s in cap_map[l] if s in self.style_to_id]
            self.line_allowed_sets[l] = set(ids)
            self.line_allowed_lists[l] = ids

        # --- 3. Setup Fast Fail ---
        self.pruning_cutoff = float('inf')

        # Precompute dữ liệu
        self.precomputed = self._precompute_data()

    def set_pruning_best(self, best_cost):
        """
        Thiết lập ngưỡng cắt tỉa. 
        Nếu chi phí tạm tính vượt quá (best_cost * 1.2), dừng mô phỏng.
        """
        if best_cost == float('inf'):
            self.pruning_cutoff = float('inf')
        else:
            self.pruning_cutoff = best_cost * 1.2

    def _precompute_data(self):
        # Convert paramSAM sang key là ID
        precomputed = {'style_sam': {}, 'line_capacity': {}}
        for s_name, s_id in self.style_to_id.items():
            precomputed['style_sam'][s_id] = self.input.param['paramSAM'][s_name]
            
        for l in self.input.set['setL']:
            precomputed['line_capacity'][l] = [
                self.input.param['paramH'].get((l, t), 0) * 60 * self.input.param['paramN'][l]
                for t in self.input.set['setT']
            ]
        return precomputed

    def _discount(self, t: int) -> float:
        return 1.0 / (1.0 + self.alpha) ** t

    def _is_allowed(self, line, style_id):
        return style_id in self.line_allowed_sets[line]

    def _random_allowed_style_id(self, line):
        options = self.line_allowed_lists.get(line)
        if not options: return None
        return random.choice(options)

    def _get_initial_style_id(self, line):
        if 'paramY0' in self.input.param:
            for s_name, s_id in self.style_to_id.items():
                if self.input.param['paramY0'].get((line, s_name), 0) == 1:
                    return s_id
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
        
        demand_by_id_time = {}
        for (s_name, t), val in self.input.param['paramD'].items():
            if s_name in self.style_to_id:
                demand_by_id_time[(self.style_to_id[s_name], t)] = val

        for l in self.input.set['setL']:
            allowed_ids = self.line_allowed_lists[l]
            demands = {
                s_id: sum(demand_by_id_time.get((s_id, t), 0) for t in self.input.set['setT'])
                for s_id in allowed_ids
            }
            if not demands:
                initial_style_id = self._random_allowed_style_id(l)
            else:
                initial_style_id = max(demands, key=demands.get)
            
            for t in self.input.set['setT']:
                solution['assignment'][(l, t)] = initial_style_id
        
        return self.repair_and_evaluate(solution)

    def repair_and_evaluate(self, solution):
        """
        Main simulation loop: Calculate cost, inventory, and handle constraints.
        """
        assignment = solution.get('assignment', {})
        
        # --- REPAIR PHASE ---
        # Ensure all assignments are valid based on capability
        for (l, t), s_id in list(assignment.items()):
            if isinstance(s_id, str): s_id = self.style_to_id.get(s_id)
            
            if s_id is None or not self._is_allowed(l, s_id):
                assignment[(l, t)] = self._random_allowed_style_id(l)
        
        solution['assignment'] = assignment

        # --- EVALUATE INITIALIZATION ---
        move_type = solution.get("type")
        solution.update({
            "production":  {},
            "shipment":    {},
            "changes":     {},
            "experience":  {},
            "efficiency":  {}
        })

        # Initialize Inventory & Backlog (ID based)
        inv_fab = defaultdict(float)
        inv_prod = defaultdict(float)
        backlog = defaultdict(float)
        
        for s_name, val in self.input.param["paramI0fabric"].items():
            if s_name in self.style_to_id: inv_fab[self.style_to_id[s_name]] = val
        for s_name, val in self.input.param["paramI0product"].items():
            if s_name in self.style_to_id: inv_prod[self.style_to_id[s_name]] = val
        for s_name, val in self.input.param["paramB0"].items():
            if s_name in self.style_to_id: backlog[self.style_to_id[s_name]] = val

        setup_cost = late_cost = exp_reward = 0.0

        # Line States:
        # up_exp: Flag = 1 nếu hôm nay làm việc hiệu quả -> hôm sau exp tăng
        # exp: Kinh nghiệm tích lũy hiện tại
        line_states = {
            l: dict(current_style=self._get_initial_style_id(l),
                    exp=self.input.param["paramExp0"].get(l, 0),
                    up_exp=0)
            for l in self.input.set["setL"]
        }

        daily_prod_history = defaultdict(lambda: defaultdict(float))
        
        # Cache Lookups for speed
        get_sam = self.precomputed["style_sam"].get
        get_line_cap = self.precomputed["line_capacity"]
        param_h = self.input.param["paramH"]
        param_csetup = self.input.param["Csetup"]
        param_rexp = self.input.param["Rexp"]
        
        param_lexp = {}
        for (l, s_name), val in self.input.param["paramLexp"].items():
            if s_name in self.style_to_id: param_lexp[(l, self.style_to_id[s_name])] = val
            
        param_plate = {}
        for s_name, val in self.input.param["Plate"].items():
             if s_name in self.style_to_id: param_plate[self.style_to_id[s_name]] = val
             
        param_tfab = {}
        for s_name, val in self.input.param["paramTfabprocess"].items():
            if s_name in self.style_to_id: param_tfab[self.style_to_id[s_name]] = val

        param_tprod = {}
        for s_name, val in self.input.param["paramTprodfinish"].items():
            if s_name in self.style_to_id: param_tprod[self.style_to_id[s_name]] = val
            
        param_F = defaultdict(float)
        for (s_name, t), val in self.input.param["paramF"].items():
             if s_name in self.style_to_id: param_F[(self.style_to_id[s_name], t)] = val
             
        param_D = defaultdict(float)
        for (s_name, t), val in self.input.param["paramD"].items():
             if s_name in self.style_to_id: param_D[(self.style_to_id[s_name], t)] = val
        
        set_ssame = set()
        for (s1, s2) in self.input.set["setSsame"]:
            if s1 in self.style_to_id and s2 in self.style_to_id:
                set_ssame.add((self.style_to_id[s1], self.style_to_id[s2]))
                
        all_style_ids = list(self.style_to_id.values())
        set_l = self.input.set["setL"]
        
        # Helper for look-ahead
        sorted_times = sorted(self.input.set["setT"])
        t_index_map = {t: i for i, t in enumerate(sorted_times)}

        # --- TIME LOOP ---
        for t in sorted_times:
            
            # --- FAST FAIL CHECK ---
            current_est_cost = setup_cost + late_cost - exp_reward
            if current_est_cost > self.pruning_cutoff:
                solution['total_cost'] = float('inf')
                return solution

            disc_factor = self._discount(t)

            # 1. Fabric Receipts
            for s_id in all_style_ids:
                LT_f = param_tfab.get(s_id, 0)
                inv_fab[s_id] += param_F.get((s_id, t - LT_f), 0)

            # 2. Decide Production
            pot_prod = {s_id: [] for s_id in all_style_ids}

            for l in set_l:
                st = line_states[l]
                
                # [UPDATE EXP] 
                # Cộng kinh nghiệm từ kết quả làm việc của ngày hôm trước
                st["exp"] += st["up_exp"]
                
                new_style_id = assignment.get((l, t))
                work_day = param_h.get((l, t), 0) > 0
                
                if new_style_id is None:
                    st["up_exp"] = 0
                    continue

                # ==========================================================
                # [LOOK-AHEAD LOGIC] CHẶN CHUYỂN ĐỔI NẾU KHÔNG CÓ HÀNG
                # ==========================================================
                # Nếu định chuyển đổi sang mã mới
                if st["current_style"] is not None and st["current_style"] != new_style_id:
                    
                    # Nếu kho vải mã mới <= 0 (Sẽ bị Idle)
                    if inv_fab[new_style_id] <= 1e-6:
                        allow_switch = False
                        
                        # Kiểm tra ngày mai (t+1)
                        current_t_idx = t_index_map[t]
                        if current_t_idx < len(sorted_times) - 1:
                            next_t = sorted_times[current_t_idx + 1]
                            next_style_id = assignment.get((l, next_t))
                            
                            # Nếu ngày mai vẫn làm mã này -> CHO PHÉP (Setup trước)
                            if next_style_id == new_style_id:
                                allow_switch = True
                        
                        # Nếu ngày mai đổi mã khác -> CẤM CHUYỂN
                        if not allow_switch:
                            new_style_id = st["current_style"] # Ép giữ nguyên mã cũ
                            assignment[(l, t)] = new_style_id  # Cập nhật vào lịch
                
                # ==========================================================
                # END LOOK-AHEAD
                # ==========================================================

                # Setup Logic
                if st["current_style"] != new_style_id:
                    solution["changes"][(l, st["current_style"], new_style_id, t)] = 1
                    setup_cost += (param_csetup * disc_factor)

                    if (st["current_style"], new_style_id) not in set_ssame:
                        st["exp"] = param_lexp.get((l, new_style_id), 0)

                solution["experience"][(l, t)] = st["exp"]
                eff = self.get_efficiency(st["exp"])
                solution["efficiency"][(l, t)] = eff
                exp_reward += st["exp"] * param_rexp

                if work_day:
                    sam = get_sam(new_style_id, 0)
                    if sam > 0:
                        cap_min = get_line_cap[l][t - 1]
                        max_p = (cap_min * eff) / sam
                        pot_prod[new_style_id].append({"line": l, "max_p": max_p})
                        
                        # Mặc định reset flag tăng exp về 0
                        # Flag này chỉ bật lên 1 nếu bên dưới thực sự sản xuất được
                        st["up_exp"] = 0 
                    else:
                        st["up_exp"] = 0
                else:
                    st["up_exp"] = 0

                st["current_style"] = new_style_id

            # 3. Realise Production
            for s_id, items in pot_prod.items():
                if not items: continue
                total_cap = sum(i["max_p"] for i in items)
                actual_p = min(total_cap, inv_fab[s_id])

                daily_prod_history[s_id][t] = actual_p
                inv_fab[s_id] -= actual_p

                if total_cap > 0:
                    for i in items:
                        share = actual_p * i["max_p"] / total_cap
                        solution["production"][(i["line"], s_id, t)] = share
                        
                        # [EXP GAIN RULE]
                        # Chỉ tăng kinh nghiệm nếu sản xuất được >= 50% năng lực
                        # Nếu hết vải (actual_p=0) -> share=0 -> up_exp=0 -> Exp không tăng
                        if share >= 0.5 * i["max_p"]:
                            line_states[i["line"]]["up_exp"] = 1

            # 4. Shipments & Backlog
            for s_id in all_style_ids:
                LT_p = param_tprod.get(s_id, 0)
                finished = daily_prod_history[s_id].get(t - LT_p, 0.0)
                inv_prod[s_id] += finished

                to_ship = backlog[s_id] + param_D.get((s_id, t), 0)
                ship_qty = min(inv_prod[s_id], to_ship)

                solution["shipment"][(s_id, t)] = ship_qty
                inv_prod[s_id] -= ship_qty
                backlog[s_id] = to_ship - ship_qty

                if backlog[s_id] > 1e-6:
                    late_cost += (backlog[s_id] * param_plate.get(s_id, 0) * disc_factor)

        # --- FINALIZE ---
        # Convert Backlog to String keys for compatibility
        final_backlog_str = {self.id_to_style[k]: v for k, v in backlog.items() if k is not None}
        
        solution.update({
            "final_backlog": final_backlog_str,
            "total_setup":   setup_cost,
            "total_late":    late_cost,
            "total_exp":     exp_reward,
            "total_cost":    setup_cost + late_cost - exp_reward
        })

        if move_type:
            solution["type"] = move_type
            
        return solution

    def convert_solution_to_string_keys(self, solution):
        """Helper để convert ID sang String khi cần xuất báo cáo/visualize"""
        new_sol = copy.deepcopy(solution)
        
        new_assign = {}
        for (l, t), s_id in new_sol['assignment'].items():
            new_assign[(l, t)] = self.id_to_style.get(s_id)
        new_sol['assignment'] = new_assign
        
        new_prod = {}
        for (l, s_id, t), val in new_sol['production'].items():
             new_prod[(l, self.id_to_style.get(s_id), t)] = val
        new_sol['production'] = new_prod
        
        return new_sol