import random
import copy

class StrategicOscillationHandler:
    def __init__(self, input_data, evaluator):
        self.input = input_data
        self.evaluator = evaluator
        self.lines = list(self.input.set['setL'])
        self.times = sorted(list(self.input.set['setT']))

    def explore_infeasible_region(self, current_solution):
        """
        [RELAX] Tạo ra giải pháp vi phạm ràng buộc.
        Mục tiêu: Đẩy các style đang bị trễ (backlog) vào lịch sản xuất bất chấp capability.
        """
        shaken_solution = copy.deepcopy(current_solution)
        assignment = shaken_solution['assignment']
        
        # Lấy danh sách style đang bị backlog (đổi tên sang ID)
        backlog_map = current_solution.get('final_backlog', {})
        high_risk_ids = []
        for s_name, qty in backlog_map.items():
            if qty > 0:
                s_id = self.evaluator.style_to_id.get(s_name)
                if s_id is not None:
                    high_risk_ids.append(s_id)
        
        # Nếu không có backlog thì quậy ngẫu nhiên
        if not high_risk_ids:
            return self._random_perturbation(shaken_solution)

        # Ép style bị trễ vào các vị trí ngẫu nhiên (Infeasible Injection)
        # Số lượng thay đổi khoảng 5-8% tổng số slot
        num_changes = max(5, int(len(self.lines) * len(self.times) * 0.08))
        
        for _ in range(num_changes):
            l = random.choice(self.lines)
            t = random.choice(self.times)
            s_forced = random.choice(high_risk_ids)
            
            # Gán trực tiếp, bỏ qua kiểm tra _is_allowed
            assignment[(l, t)] = s_forced
            
        return shaken_solution

    def _random_perturbation(self, solution):
        """Đảo lộn ngẫu nhiên khi không có backlog để phá vỡ cấu trúc hiện tại."""
        assignment = solution['assignment']
        all_style_ids = list(self.evaluator.style_to_id.values())
        
        for _ in range(15):
            l = random.choice(self.lines)
            t = random.choice(self.times)
            s_id = random.choice(all_style_ids)
            assignment[(l, t)] = s_id
        return solution

    def aggressive_repair(self, infeasible_solution):
        """
        [REPAIR] Sửa chữa quyết liệt để đưa giải pháp về khả thi.
        Logic: Nếu Line A giữ Style X (sai), tìm Line B (đúng) để swap X sang,
        kể cả khi phải đẩy Style Y của Line B ra ngoài.
        """
        repaired_assign = copy.deepcopy(infeasible_solution['assignment'])
        
        # Duyệt qua toàn bộ lưới
        for l in self.lines:
            for t in self.times:
                s_id = repaired_assign.get((l, t))
                
                # Nếu gặp vị trí vi phạm (Line l không may được Style s_id)
                if s_id is not None and not self.evaluator._is_allowed(l, s_id):
                    
                    # Tìm "cứu viện": Các line khác có thể may s_id tại thời điểm t
                    candidates = [
                        cl for cl in self.lines 
                        if cl != l and self.evaluator._is_allowed(cl, s_id)
                    ]
                    
                    fixed = False
                    if candidates:
                        # Chọn ngẫu nhiên một người cứu viện
                        l_target = random.choice(candidates)
                        s_target_current = repaired_assign.get((l_target, t))
                        
                        # -- SWAP --
                        # 1. Đưa hàng sai (s_id) sang chỗ đúng (l_target)
                        repaired_assign[(l_target, t)] = s_id
                        
                        # 2. Xử lý hàng bị đẩy ra (s_target_current)
                        # Nếu l làm được s_target_current thì đổi chéo, không thì random
                        if s_target_current is not None and self.evaluator._is_allowed(l, s_target_current):
                            repaired_assign[(l, t)] = s_target_current
                        else:
                            repaired_assign[(l, t)] = self.evaluator._random_allowed_style_id(l)
                        
                        fixed = True
                    
                    if not fixed:
                        # Nếu không ai cứu được, đành xóa style vi phạm đi, gán style random hợp lệ cho l
                        repaired_assign[(l, t)] = self.evaluator._random_allowed_style_id(l)

        # Tính toán lại chi phí sau khi đã sửa xong cấu trúc
        return self.evaluator.repair_and_evaluate({'assignment': repaired_assign})
