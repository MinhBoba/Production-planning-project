# metaheuristic/oscillation_strategy.py
import random
import copy

class StrategicOscillationHandler:
    def __init__(self, input_data, evaluator):
        self.input = input_data
        self.evaluator = evaluator # Cần evaluator để check ID và ràng buộc
        self.lines = list(self.input.set['setL'])
        self.times = sorted(list(self.input.set['setT']))

    def explore_infeasible_region(self, current_solution):
        """
        Bước 1: Relax - Tạo ra giải pháp không khả thi.
        Chiến thuật: Lấy các Style đang bị trễ hàng (Backlog cao) và ép buộc gán vào 
        các Line ngẫu nhiên, BẤT CHẤP việc Line đó có làm được (enable) hay không.
        """
        shaken_solution = copy.deepcopy(current_solution)
        assignment = shaken_solution['assignment']
        
        # Tìm các style đang bị backlog nặng nhất
        backlog_map = current_solution.get('final_backlog', {})
        # Chuyển backlog key từ tên Style sang ID (nếu đang là string)
        high_risk_ids = []
        for s_name, qty in backlog_map.items():
            if qty > 0:
                s_id = self.evaluator.style_to_id.get(s_name)
                if s_id is not None:
                    high_risk_ids.append(s_id)
        
        if not high_risk_ids:
            # Nếu không có backlog, thử random đảo lộn một vùng thời gian
            return self._random_perturbation(shaken_solution)

        # Thực hiện "ép" gán (Infeasible Move)
        # Chọn ngẫu nhiên 5-10% số slot thời gian để ép style vào
        num_changes = max(5, int(len(self.lines) * len(self.times) * 0.05))
        
        for _ in range(num_changes):
            l = random.choice(self.lines)
            t = random.choice(self.times)
            s_forced = random.choice(high_risk_ids)
            
            # Gán trực tiếp không cần kiểm tra _is_allowed
            assignment[(l, t)] = s_forced
            
        return shaken_solution

    def _random_perturbation(self, solution):
        """Hàm phụ: Nếu không có backlog thì đảo lộn ngẫu nhiên để tạo dao động"""
        assignment = solution['assignment']
        for _ in range(10):
            l = random.choice(self.lines)
            t = random.choice(self.times)
            # Random bất kỳ style nào có trong hệ thống
            s_id = random.choice(list(self.evaluator.style_to_id.values()))
            assignment[(l, t)] = s_id
        return solution

    def aggressive_repair(self, infeasible_solution):
        """
        Bước 2: Repair - Sửa chữa để đưa về vùng khả thi.
        Chiến thuật: Duyệt qua các slot vi phạm capability.
        Thử swap style vi phạm sang một Line khác CÓ THỂ làm được style đó.
        """
        repaired_assign = copy.deepcopy(infeasible_solution['assignment'])
        
        # Duyệt tìm vi phạm
        for l in self.lines:
            for t in self.times:
                s_id = repaired_assign.get((l, t))
                
                # Nếu Line l không được phép may Style s_id (Vi phạm Infeasible)
                if s_id is not None and not self.evaluator._is_allowed(l, s_id):
                    
                    # Tìm cứu viện: Tìm line khác (l_target) tại cùng thời điểm t có thể may s_id
                    candidates = [
                        candidate_l for candidate_l in self.lines 
                        if candidate_l != l and self.evaluator._is_allowed(candidate_l, s_id)
                    ]
                    
                    fixed = False
                    if candidates:
                        # Chọn một line cứu viện ngẫu nhiên
                        l_target = random.choice(candidates)
                        s_target_current = repaired_assign.get((l_target, t))
                        
                        # Swap: Đưa s_id sang l_target
                        repaired_assign[(l_target, t)] = s_id
                        
                        # Xử lý style bị đẩy ra (s_target_current)
                        # Nếu mang về l mà l làm được thì tốt, không thì gán random hợp lệ
                        if s_target_current is not None and self.evaluator._is_allowed(l, s_target_current):
                            repaired_assign[(l, t)] = s_target_current
                        else:
                            # Nếu hoán đổi xong mà vẫn lỗi ở vị trí cũ, gán lại cái hợp lệ cho l
                            repaired_assign[(l, t)] = self.evaluator._random_allowed_style_id(l)
                        fixed = True
                    
                    if not fixed:
                        # Nếu không tìm được ai cứu, đành reset vị trí này về random hợp lệ
                        repaired_assign[(l, t)] = self.evaluator._random_allowed_style_id(l)

        # Sau khi sửa xong cấu trúc, gọi evaluator để tính toán lại chỉ số (Cost, Inventory...)
        return self.evaluator.repair_and_evaluate({'assignment': repaired_assign})
