import time
import copy
from collections import deque, defaultdict
import random

# Import các module nội bộ
from .neighbor_generator import NeighborGenerator
from .ALNS_operator import ALNSOperator
from .oscillation_strategy import StrategicOscillationHandler

class TabuSearchSolver:
    def __init__(self, input_data, discount_alpha=0.05, initial_line_df=None, max_iter=1000, 
                 tabu_tenure=15, max_time=1200, min_tenure=5, max_tenure=40, 
                 increase_threshold=50, decrease_threshold=10, verbose=True):
        
        self.input = input_data
        self.alpha = discount_alpha
        self.max_iter = max_iter
        self.max_time = max_time
        self.verbose = verbose
        
        # --- TABU PARAMETERS ---
        self.current_tenure = tabu_tenure
        self.min_tenure = min_tenure
        self.max_tenure = max_tenure
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.tabu_list = deque(maxlen=self.current_tenure)
        
        # --- ADAPTIVE TRACKING ---
        self.no_improvement_counter = 0
        self.consecutive_improvements_counter = 0
        self.mo_probability = 0.5  # Xác suất chạy Multi-Objective move
        self.mo_moves_attempted = 0
        self.mo_moves_accepted_as_best = 0

        # --- SETUP CAPABILITY MAP ---
        # Map này dùng để NeighborGenerator biết Line nào làm được gì
        param_enable = self.input.param.get("paramYenable", {})
        self.cap_map = defaultdict(set)
        for (l, s), val in param_enable.items():
            if val: self.cap_map[l].add(s)

        # Kiểm tra dữ liệu đầu vào cơ bản
        for l in self.input.set['setL']:
            if not self.cap_map[l]:
                print(f"WARNING: Line {l} không có khả năng may mã nào (paramYenable toàn 0).")

        # --- INITIALIZE COMPONENTS ---
        # 1. Evaluator: Tính toán chi phí, check ràng buộc (Core logic)
        self.evaluator = ALNSOperator(input_data, self.cap_map, discount_alpha)
        
        # 2. Generator: Sinh láng giềng
        self.neighbor_gen = NeighborGenerator(input_data, self.cap_map)
        
        # 3. Oscillation: Xử lý phá vỡ rào cản (Infeasible -> Feasible)
        self.oscillation_handler = StrategicOscillationHandler(input_data, self.evaluator)

        # --- INITIAL SOLUTION ---
        print("Đang tạo giải pháp ban đầu...")
        self.current_solution = self.evaluator.initialize_solution()
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_cost = self.current_solution['total_cost']
        self.costs = [self.best_cost]
        self.start_time = time.time()

    def solve(self):
        print(f"\n--- BẮT ĐẦU TỐI ƯU HÓA ---")
        print(f"Chi phí ban đầu: {self.best_cost:,.2f}")
        print(f"Tham số: Max Iter={self.max_iter}, Max Time={self.max_time}s")
        print("-" * 60)
        
        last_iter = 0
        
        for i in range(1, self.max_iter + 1):
            last_iter = i
            
            # 1. Kiểm tra thời gian
            if time.time() - self.start_time > self.max_time:
                print(f"\n[STOP] Đã đạt giới hạn thời gian tại vòng lặp {i}.")
                break

            # 2. Cập nhật Fast Fail cho Evaluator
            self.evaluator.set_pruning_best(self.best_cost)

            # ==========================================================
            # A. CHIẾN LƯỢC DAO ĐỘNG (STRATEGIC OSCILLATION)
            # ==========================================================
            # Kích hoạt khi bế tắc (no_improve > 150) hoặc định kỳ (mỗi 250 vòng)
            oscillation_triggered = False
            if i > 50 and (self.no_improvement_counter > 150 or i % 250 == 0):
                oscillation_triggered = self._perform_oscillation(i)
            
            # Nếu oscillation đã tìm ra hướng đi mới và reset bộ đếm, 
            # ta có thể skip phần tìm neighbor truyền thống ở vòng này để tiết kiệm thời gian
            if oscillation_triggered and self.no_improvement_counter == 0:
                continue

            # ==========================================================
            # B. TÌM KIẾM LÁNG GIỀNG (NEIGHBORHOOD SEARCH)
            # ==========================================================
            neighbors = self.neighbor_gen.generate_neighbors(
                self.current_solution, 
                self.mo_probability, 
                self.evaluator
            )
            
            if not neighbors:
                continue

            # Sắp xếp để ưu tiên các giải pháp tốt (Best Improvement strategy)
            # Tuy nhiên, Tabu Search thường duyệt hết, ở đây ta sort để dễ check Aspiration
            neighbors.sort(key=lambda s: s['total_cost'])
            
            best_neighbor = None
            found_valid_move = False
            chosen_move_is_mo = False

            # Duyệt qua các láng giềng
            for neighbor in neighbors:
                move_signature = self._get_move_signature(self.current_solution['assignment'], neighbor['assignment'])
                cost = neighbor['total_cost']
                
                # Aspiration Criteria: Nếu tốt hơn Best Global -> Bỏ qua Tabu
                is_aspiration = cost < self.best_cost
                is_tabu = move_signature in self.tabu_list
                
                if is_aspiration or not is_tabu:
                    best_neighbor = neighbor
                    found_valid_move = True
                    chosen_move_is_mo = (neighbor.get('type') == 'mo_move')
                    
                    # Cập nhật Tabu List
                    self.tabu_list.append(move_signature)
                    
                    # Cập nhật Best Global nếu cần
                    if is_aspiration:
                        self.best_solution = copy.deepcopy(neighbor)
                        self.best_cost = cost
                        self._on_improvement(i, cost, source="TabuSearch")
                    else:
                        self._on_no_improvement()
                    
                    break # Chọn được nước đi tốt nhất khả dĩ rồi thì dừng (Best Fit)

            # Cập nhật Current Solution
            if found_valid_move:
                self.current_solution = best_neighbor
            else:
                # Nếu tất cả đều bị Tabu (hiếm gặp), chọn cái tốt nhất bất chấp Tabu
                # để thuật toán không bị kẹt chết
                best_neighbor = neighbors[0]
                self.current_solution = best_neighbor
                # Vẫn tính là không cải thiện global
                self._on_no_improvement()

            self.costs.append(self.current_solution['total_cost'])

            # ==========================================================
            # C. CẬP NHẬT CHIẾN THUẬT (ADAPTIVE STRATEGY)
            # ==========================================================
            self._update_mo_strategy(chosen_move_is_mo, found_valid_move and best_neighbor['total_cost'] < self.costs[-2] if len(self.costs)>1 else False)
            self._update_tenure()

            # Logging định kỳ
            if i % 100 == 0:
                print(f"Iter {i:5d} | Current: {self.current_solution['total_cost']:12,.0f} | Best: {self.best_cost:12,.0f} | Tenure: {self.current_tenure:2d} | MO Prob: {self.mo_probability:.2f}")

        # --- KẾT THÚC ---
        return self._finalize_solution(last_iter)

    # --------------------------------------------------------------------------
    #  HELPER METHODS
    # --------------------------------------------------------------------------

    def _perform_oscillation(self, iter_idx):
        """Thực hiện logic dao động chiến lược: Relax -> Repair."""
        if self.verbose:
            print(f"  >> [Oscillation] Kích hoạt tại vòng {iter_idx}. Đang thăm dò vùng Infeasible...")

        # 1. Relax: Tạo giải pháp vi phạm
        relaxed_sol = self.oscillation_handler.explore_infeasible_region(self.best_solution)
        
        # 2. Repair: Sửa chữa quyết liệt
        feasible_sol = self.oscillation_handler.aggressive_repair(relaxed_sol)
        
        cost_new = feasible_sol['total_cost']
        improved = False
        
        # 3. Đánh giá
        if cost_new < self.best_cost:
            # Tìm thấy kỷ lục mới nhờ Oscillation
            self.best_solution = copy.deepcopy(feasible_sol)
            self.best_cost = cost_new
            self.current_solution = feasible_sol
            self._on_improvement(iter_idx, cost_new, source="Oscillation")
            
            # Reset Tabu List để tự do khai thác vùng đất mới này
            self.tabu_list.clear()
            improved = True
            
        elif self.no_improvement_counter > 200:
            # Chế độ "Tuyệt vọng": Nếu bế tắc quá lâu, chấp nhận giải pháp từ Oscillation
            # kể cả khi nó không tốt hơn Best Global, miễn là nó khác biệt để thoát hố.
            # (Ở đây ta check nếu nó không quá tệ so với current)
            if cost_new < self.current_solution['total_cost'] * 1.1:
                if self.verbose:
                    print(f"  >> [Oscillation] Chấp nhận giải pháp thay thế để thoát bế tắc (Cost: {cost_new:,.0f}).")
                self.current_solution = feasible_sol
                self.tabu_list.clear()
                self.no_improvement_counter = 50 # Reset một phần
                improved = True # Trả về True để báo main loop skip neighbor search vòng này

        return improved

    def _on_improvement(self, iter_idx, new_cost, source="Tabu"):
        """Xử lý khi tìm thấy giải pháp tốt hơn."""
        print(f"[{source}] Vòng {iter_idx}: Kỷ lục mới! Chi phí: {new_cost:,.2f}")
        self.consecutive_improvements_counter += 1
        self.no_improvement_counter = 0

    def _on_no_improvement(self):
        """Xử lý khi không tìm thấy giải pháp tốt hơn toàn cục."""
        self.no_improvement_counter += 1
        self.consecutive_improvements_counter = 0

    def _update_tenure(self):
        """Điều chỉnh độ dài danh sách cấm (Tabu Tenure) động."""
        if self.consecutive_improvements_counter >= self.decrease_threshold:
            # Đang thuận lợi -> Giảm tenure để khai thác sâu (Intensification)
            if self.current_tenure > self.min_tenure:
                self.current_tenure -= 1
                self.tabu_list = deque(self.tabu_list, maxlen=self.current_tenure)
            self.consecutive_improvements_counter = 0
            
        elif self.no_improvement_counter >= self.increase_threshold:
            # Đang bế tắc -> Tăng tenure để đi xa hơn (Diversification)
            if self.current_tenure < self.max_tenure:
                self.current_tenure += 2
                self.tabu_list = deque(self.tabu_list, maxlen=self.current_tenure)
            self.no_improvement_counter = 0 # Reset để tránh tăng liên tục quá nhanh

    def _update_mo_strategy(self, move_was_mo, move_was_improvement):
        """Điều chỉnh xác suất sử dụng Multi-Objective moves."""
        if move_was_mo:
            self.mo_moves_attempted += 1
            if move_was_improvement:
                self.mo_moves_accepted_as_best += 1
        
        # Điều chỉnh mỗi 50 lần thử
        if self.mo_moves_attempted > 50:
            rate = self.mo_moves_accepted_as_best / self.mo_moves_attempted
            if rate > 0.15: # Nếu hiệu quả khá
                self.mo_probability = min(0.9, self.mo_probability + 0.05)
            else:
                self.mo_probability = max(0.2, self.mo_probability - 0.05)
            
            # Reset
            self.mo_moves_attempted = 0
            self.mo_moves_accepted_as_best = 0

    def _get_move_signature(self, old_assign, new_assign):
        """Tạo 'chữ ký' cho nước đi để lưu vào Tabu List.
        Chữ ký là tập hợp các thay đổi: ((line, date, old_style, new_style), ...)
        """
        changes = []
        for key, val in old_assign.items():
            if new_assign[key] != val:
                changes.append((key, val, new_assign[key]))
        # Sort để đảm bảo tính nhất quán của tuple
        return tuple(sorted(changes))

    def _finalize_solution(self, iterations_run):
        print("\n" + "="*50)
        print("TỐI ƯU HÓA HOÀN TẤT")
        print(f"Chi phí tốt nhất: {self.best_cost:,.2f}")
        print(f"Tổng số vòng lặp: {iterations_run}")
        print(f"Thời gian chạy: {time.time() - self.start_time:.2f}s")
        print("="*50)
        
        # Tắt cắt tỉa để tính toán chính xác lần cuối
        self.evaluator.set_pruning_best(float('inf'))
        
        # 1. Tính toán lại đầy đủ các chỉ số
        final_sol_id = self.evaluator.repair_and_evaluate(self.best_solution)
        
        # 2. Convert ID -> String (quan trọng để xuất Excel)
        final_sol_str = self.evaluator.convert_solution_to_string_keys(final_sol_id)
        
        return final_sol_str

    def print_solution_summary(self, solution=None):
        sol = solution or self.best_solution # solution này nên là dạng String keys (sau khi finalize)
        if not sol: 
            print("Chưa có giải pháp nào.")
            return
        
        # Lưu ý: Nếu gọi hàm này trước khi finalize, sol có thể đang dùng ID
        # Nên check an toàn
        total = sol.get('total_cost', 0)
        setup = sol.get('total_setup', 0)
        late = sol.get('total_late', 0)
        exp = sol.get('total_exp', 0)
        
        print(f"Tổng chi phí: {total:,.2f}")
        print(f"  - Chi phí Setup: {setup:,.2f}")
        print(f"  - Phạt trễ hạn:  {late:,.2f}")
        print(f"  - Thưởng kinh nghiệm: {exp:,.2f}")
