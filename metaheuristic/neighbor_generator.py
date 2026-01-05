import copy
import random
import numpy as np

class NeighborGenerator:
    """
    RL-based Neighbor Generator (Adaptive Operator Selection).
    Sử dụng Q-Learning (Multi-Armed Bandit) để tự động chọn chiến lược sinh láng giềng tốt nhất
    tại mỗi thời điểm thay vì dùng xác suất cố định.
    """

    def __init__(self, input_data, cap_map, alpha=0.1, gamma=0.0, epsilon=0.3):
        self.input = input_data
        self.cap_map = cap_map
        self.lines = list(self.input.set['setL'])
        self.times = sorted(list(self.input.set['setT']))

        # --- RL CONFIGURATION ---
        # Danh sách các "cánh tay" (Arms/Operators) mà Agent có thể chọn
        self.operators = [
            'swap',             # Trao đổi vị trí (Explore)
            'reassign_single',  # Gán lại 1 vị trí (Explore)
            'reassign_block',   # Gán lại 1 khối (Explore/Exploit)
            'setup_reduction',  # Giảm chi phí chuyển đổi (Exploit)
            'late_reduction',   # Giảm chi phí phạt trễ (Exploit)
            'balanced'          # Cân bằng (Exploit)
        ]

        # Q-Table: Lưu giá trị kỳ vọng (điểm uy tín) của mỗi toán tử
        # Khởi tạo bằng 10.0 để khuyến khích khám phá ban đầu (Optimistic Initialization)
        self.q_values = {op: 10.0 for op in self.operators}
        
        # Đếm số lần chọn để thống kê
        self.selection_counts = {op: 0 for op in self.operators}

        # RL Hyperparameters
        self.alpha = alpha      # Learning rate (Tốc độ học: 0.1)
        self.gamma = gamma      # Discount factor (0.0 vì bài toán này thưởng tức thì quan trọng hơn)
        self.epsilon = epsilon  # Epsilon-Greedy (Tỷ lệ khám phá ngẫu nhiên)
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    # =================================================================
    #  RL CORE METHODS
    # =================================================================

    def select_operator(self):
        """Chọn toán tử dựa trên chiến lược Epsilon-Greedy."""
        # 1. Khám phá (Exploration): Chọn ngẫu nhiên
        if random.random() < self.epsilon:
            selected = random.choice(self.operators)
        # 2. Khai thác (Exploitation): Chọn toán tử có Q-value cao nhất
        else:
            # Lấy danh sách các op có Q cao nhất (để random nếu có nhiều cái bằng nhau)
            max_q = max(self.q_values.values())
            best_ops = [op for op, q in self.q_values.items() if q == max_q]
            selected = random.choice(best_ops)
        
        return selected

    def update_reward(self, operator, reward):
        """
        Cập nhật Q-Table dựa trên kết quả thực tế.
        Reward > 0: Cải thiện cost.
        Reward <= 0: Không cải thiện.
        """
        # Nếu reward âm (tệ đi), ta phạt nhẹ để nó ít chọn lại, nhưng không phạt quá nặng
        # để tránh việc nó "sợ" không dám đi nữa.
        # Ta dùng hàm tanh hoặc clip để chuẩn hóa reward nếu cần, ở đây dùng trực tiếp.
        
        old_q = self.q_values[operator]
        
        # Công thức Q-Learning: Q(s,a) = Q(s,a) + alpha * (r + gamma * maxQ(s',a') - Q(s,a))
        # Vì gamma=0 (One-step bandit), công thức rút gọn:
        new_q = old_q + self.alpha * (reward - old_q)
        
        self.q_values[operator] = new_q
        self.selection_counts[operator] += 1
        
        # Giảm dần độ ngẫu nhiên theo thời gian
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def generate_neighbors(self, base_solution, evaluator):
        """
        Quy trình chính:
        1. Agent chọn 1 Toán tử (Operator).
        2. Sinh ra một nhóm (Batch) các láng giềng chỉ dùng toán tử đó.
        3. Đánh dấu (Tag) để Tabu Search biết đường trả thưởng.
        """
        selected_op = self.select_operator()
        neighbors = []
        
        # Số lượng láng giềng sinh ra trong 1 lần (Batch size)
        # Sinh nhiều một chút để tăng xác suất tìm được bước đi tốt từ toán tử đó
        batch_size = 10 

        # --- Dispatcher ---
        if selected_op == 'swap':
            neighbors = self._gen_swap(base_solution, evaluator, batch_size)
        elif selected_op == 'reassign_single':
            neighbors = self._gen_reassign_single(base_solution, evaluator, batch_size)
        elif selected_op == 'reassign_block':
            neighbors = self._gen_reassign_block(base_solution, evaluator, batch_size)
        elif selected_op == 'setup_reduction':
            neighbors = self._gen_setup_reduction(base_solution, evaluator) # Logic cũ tự quyết định số lượng
        elif selected_op == 'late_reduction':
            neighbors = self._gen_late_reduction(base_solution, evaluator)
        elif selected_op == 'balanced':
            neighbors = self._gen_balanced(base_solution, evaluator)

        # Gắn thẻ (Tagging) để feedback
        valid_neighbors = []
        for n in neighbors:
            if n is not None:
                n['origin_operator'] = selected_op
                valid_neighbors.append(n)
                
        return valid_neighbors

    # =================================================================
    #  OPERATOR IMPLEMENTATIONS (Logic cũ được chia nhỏ)
    # =================================================================

    def _is_allowed(self, line, style):
        return style in self.cap_map[line]

    def _random_allowed_style(self, line):
        if not self.cap_map[line]: return None
        return random.choice(list(self.cap_map[line]))

    # --- 1. Swap ---
    def _gen_swap(self, base_solution, evaluator, n_attempts):
        neighbors = []
        base_assign = base_solution['assignment']
        if len(self.times) < 2: return []

        for _ in range(n_attempts):
            new_assign = copy.copy(base_assign) # Shallow copy dict is fast enough
            l = random.choice(self.lines)
            t1, t2 = random.sample(self.times, 2)
            
            if new_assign[(l, t1)] != new_assign[(l, t2)]:
                new_assign[(l, t1)], new_assign[(l, t2)] = new_assign[(l, t2)], new_assign[(l, t1)]
                neighbors.append(evaluator.repair_and_evaluate({'assignment': new_assign}))
        return neighbors

    # --- 2. Reassign Single ---
    def _gen_reassign_single(self, base_solution, evaluator, n_attempts):
        neighbors = []
        base_assign = base_solution['assignment']
        
        for _ in range(n_attempts):
            new_assign = copy.copy(base_assign)
            l = random.choice(self.lines)
            t = random.choice(self.times)
            
            new_style = self._random_allowed_style(l)
            if new_style and new_style != new_assign.get((l, t)):
                new_assign[(l, t)] = new_style
                neighbors.append(evaluator.repair_and_evaluate({'assignment': new_assign}))
        return neighbors

    # --- 3. Reassign Block ---
    def _gen_reassign_block(self, base_solution, evaluator, n_attempts):
        neighbors = []
        base_assign = base_solution['assignment']
        if len(self.times) < 5: return []

        for _ in range(n_attempts):
            new_assign = copy.copy(base_assign)
            l = random.choice(self.lines)
            block_size = random.randint(2, max(2, len(self.times) // 4))
            start_idx = random.randint(0, len(self.times) - block_size)
            
            new_style = self._random_allowed_style(l)
            if new_style:
                changed = False
                for i in range(block_size):
                    t = self.times[start_idx + i]
                    if new_assign[(l, t)] != new_style:
                        new_assign[(l, t)] = new_style
                        changed = True
                
                if changed:
                    neighbors.append(evaluator.repair_and_evaluate({'assignment': new_assign}))
        return neighbors

    # --- 4. Setup Reduction (Logic thông minh cũ) ---
    def _gen_setup_reduction(self, base_solution, evaluator):
        moves = []
        current_assign = base_solution['assignment']

        # Cố gắng tìm tối đa 5 move giảm setup
        attempts = 0
        for l in self.lines:
            # Tìm các đoạn ngắn <= 3 ngày
            segments = self._find_short_segments(l, current_assign)
            if not segments: continue
            
            # Chỉ lấy ngẫu nhiên 2 segment để xử lý mỗi lần gọi để tiết kiệm time
            for segment in random.sample(segments, min(len(segments), 2)):
                dominant = self._get_dominant_neighbor_style(l, segment, current_assign)
                if dominant and self._is_allowed(l, dominant):
                    new_assign = copy.copy(current_assign)
                    for t in segment['periods']:
                        new_assign[(l, t)] = dominant
                    moves.append(evaluator.repair_and_evaluate({'assignment': new_assign}))
                    attempts += 1
            if attempts >= 5: break
            
        return moves

    # --- 5. Late Cost Reduction (Logic thông minh cũ) ---
    def _gen_late_reduction(self, base_solution, evaluator):
        moves = []
        current_assign = base_solution['assignment']
        
        # Tìm style đang bị backlog cao nhất
        high_risk = self._identify_high_risk_styles(base_solution)[:3] # Top 3
        if not high_risk: return []

        for style in high_risk:
            # Tìm các slot trống hoặc slot của style khác để chèn vào
            # Thử chèn vào 3 vị trí ngẫu nhiên
            valid_slots = [
                (l, t) for l in self.lines for t in self.times
                if self._is_allowed(l, style) and current_assign.get((l, t)) != style
            ]
            
            if valid_slots:
                for _ in range(min(3, len(valid_slots))):
                    l, t = random.choice(valid_slots)
                    new_assign = copy.copy(current_assign)
                    new_assign[(l, t)] = style
                    moves.append(evaluator.repair_and_evaluate({'assignment': new_assign}))
        return moves

    # --- 6. Balanced Move ---
    def _gen_balanced(self, base_solution, evaluator):
        # Swap có chủ đích: Chỉ swap nếu style khác nhau (tránh swap vô nghĩa)
        moves = []
        current_assign = base_solution['assignment']
        
        for _ in range(5):
            l = random.choice(self.lines)
            if len(self.times) < 2: continue
            t1, t2 = random.sample(self.times, 2)
            
            if current_assign[(l, t1)] != current_assign[(l, t2)]:
                new_assign = copy.copy(current_assign)
                new_assign[(l, t1)], new_assign[(l, t2)] = new_assign[(l, t2)], new_assign[(l, t1)]
                moves.append(evaluator.repair_and_evaluate({'assignment': new_assign}))
        return moves

    # --- Helpers ---
    def _find_short_segments(self, line, assignment):
        segments = []
        current_style = None
        current_segment = []
        
        for t in self.times:
            style = assignment.get((line, t))
            if style != current_style:
                if current_segment: segments.append({'style': current_style, 'periods': current_segment})
                current_style = style
                current_segment = [t]
            else:
                current_segment.append(t)
        if current_segment: segments.append({'style': current_style, 'periods': current_segment})
        return [s for s in segments if len(s['periods']) <= 3]

    def _get_dominant_neighbor_style(self, line, segment, assignment):
        start_t = min(segment['periods'])
        end_t = max(segment['periods'])
        try:
            start_idx = self.times.index(start_t)
            end_idx = self.times.index(end_t)
        except ValueError: return None

        prev_style = assignment.get((line, self.times[start_idx-1])) if start_idx > 0 else None
        next_style = assignment.get((line, self.times[end_idx+1])) if end_idx < len(self.times) - 1 else None
        
        if prev_style and prev_style == next_style: return prev_style
        return prev_style or next_style

    def _identify_high_risk_styles(self, solution):
        backlog = solution.get('final_backlog', {})
        if not backlog: return []
        sorted_styles = sorted(backlog.keys(), key=lambda s: backlog[s], reverse=True)
        return [s for s in sorted_styles if backlog[s] > 0]
