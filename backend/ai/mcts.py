"""
蒙特卡洛树搜索 (MCTS) 模块
实现 AlphaZero 风格的 MCTS 算法
"""

import numpy as np
import copy
import math
import random
import torch
from typing import Optional, Tuple, List

from .network import PolicyValueNetwork, board_to_tensor, get_network_prediction, BOARD_SIZE


# 训练配置常量
TRAIN_SIMULATION = 30
TRAIN_BUFF = 0.8


def check_winner(board: list) -> float:
    """
    检查棋盘是否有获胜方
    
    Args:
        board: 15x15 棋盘，1=当前方，-1=对方，0=空
        
    Returns:
        1: 当前方赢，-1: 对方赢，0: 未结束
    """
    num_used = sum(1 for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] != 0)
    
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != 0:
                for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    count = 0
                    for d in range(5):
                        ni, nj = i + d * dx, j + d * dy
                        if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE and board[i][j] == board[ni][nj]:
                            count += 1
                        else:
                            break
                    if count == 5:
                        # 略微惩罚较长的游戏
                        if board[i][j] == 1:
                            return 1 - num_used * 3e-4
                        else:
                            return -(1 - num_used * 3e-4)
    return 0


class Node:
    """MCTS 树节点"""
    
    def __init__(self, board: list, parent: Optional['Node'] = None, move: Optional[Tuple[int, int]] = None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: dict = {}  # move -> (child_node, prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.value: Optional[float] = None  # 神经网络评估值
        self.val = 0.0  # 用于 get_train_data
    
    @property
    def q_value(self) -> float:
        """平均价值"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def update_value(self):
        """更新节点值 (用于训练数据生成)"""
        if len(self.children) == 0:
            self.val = self.value if self.value else 0
        else:
            self.val = self.value_sum / self.visit_count if self.visit_count > 0 else 0


class MCTS:
    """蒙特卡洛树搜索"""
    
    def __init__(self, model_path: str = None, model: PolicyValueNetwork = None,
                 c_puct: float = 0.8, use_noise: float = 0.01):
        """
        Args:
            model_path: 模型文件路径
            model: 已加载的模型
            c_puct: 探索系数
            use_noise: 噪声强度
        """
        self.c_puct = c_puct
        self.use_noise = use_noise
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = PolicyValueNetwork()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            raise ValueError("必须提供 model_path 或 model")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 用于收集训练数据的节点列表（每次搜索前清空）
        self.visited_nodes: List[Node] = []
    
    def clear_cache(self):
        """清理缓存，释放内存"""
        self.visited_nodes.clear()
    
    def _is_board_full(self, board: list) -> bool:
        """检查棋盘是否已满"""
        return all(board[i][j] != 0 for i in range(BOARD_SIZE) for j in range(BOARD_SIZE))
    
    def _is_terminal(self, board: list) -> bool:
        """检查是否为终止状态"""
        return self._is_board_full(board) or check_winner(board) != 0
    
    def search(self, root_board: list, num_simulations: int, 
               existing_root: Optional[Node] = None) -> Tuple[Tuple[float, np.ndarray], Node]:
        """
        执行 MCTS 搜索
        
        Args:
            root_board: 当前棋盘状态
            num_simulations: 模拟次数
            existing_root: 已有的根节点（用于复用搜索树）
            
        Returns:
            ((value, probs), root): 搜索结果和根节点
        """
        # 如果没有复用根节点，清空之前的缓存
        if existing_root is None:
            self.visited_nodes.clear()
            root = Node(root_board)
            self.visited_nodes.append(root)
        else:
            root = existing_root
        
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            
            # 选择阶段：沿树下降直到叶子节点
            while node.children:
                node = self._select_child(node)
                search_path.append(node)
            
            # 扩展和评估
            if not self._is_terminal(node.board):
                self._expand_node(node)
            else:
                node.value = self._evaluate_terminal(node.board)
            
            # 回溯更新
            value = node.value if node.value is not None else 0
            for n in reversed(search_path):
                n.visit_count += 1
                n.value_sum += value
                n.update_value()
                value = -value
        
        return self._get_results(root), root
    
    def _select_child(self, node: Node) -> Node:
        """UCB选择子节点"""
        total_visits = sum(
            (child.visit_count if child else 0) 
            for child, _ in node.children.values()
        )
        sqrt_total = math.sqrt(total_visits + 1)
        
        # 计算平均值用于未访问节点
        visited_value_sum = sum(
            child.q_value * child.visit_count 
            for child, _ in node.children.values() if child
        )
        visited_count = sum(
            child.visit_count for child, _ in node.children.values() if child
        )
        avg_value = visited_value_sum / (visited_count + 1e-5)
        
        best_score = -float('inf')
        best_move = None
        
        for move, (child, prior) in node.children.items():
            if child and child.visit_count > 0:
                exploit = child.q_value
                explore = self.c_puct * prior * sqrt_total / (child.visit_count + 1)
            else:
                exploit = avg_value
                explore = self.c_puct * prior * sqrt_total
            
            score = explore - exploit
            if score > best_score:
                best_score = score
                best_move = move
        
        # 如果子节点不存在，创建它
        child, prior = node.children[best_move]
        if child is None:
            i, j = best_move
            new_board = copy.deepcopy(node.board)
            new_board[i][j] = 1
            # 翻转视角
            for x in range(BOARD_SIZE):
                for y in range(BOARD_SIZE):
                    new_board[x][y] *= -1
            child = Node(new_board, parent=node, move=best_move)
            self.visited_nodes.append(child)
            node.children[best_move] = (child, prior)
        
        return child
    
    def _expand_node(self, node: Node):
        """扩展节点"""
        value, policy = get_network_prediction(self.model, node.board, self.device)
        node.value = value
        
        # 归一化策略概率
        total_prior = sum(
            policy[i][j] for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)
            if node.board[i][j] == 0
        )
        if total_prior < 1e-10:
            total_prior = 1e-10
        
        # 添加子节点
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if node.board[i][j] == 0:
                    prior = policy[i][j] / total_prior
                    if self.use_noise > 0:
                        prior += random.gauss(0, self.use_noise)
                    node.children[(i, j)] = (None, max(prior, 0))
    
    def _evaluate_terminal(self, board: list) -> float:
        """评估终止状态"""
        if self._is_board_full(board):
            return 0
        return check_winner(board)
    
    def _get_results(self, root: Node) -> Tuple[float, np.ndarray]:
        """获取搜索结果"""
        probs = np.zeros((BOARD_SIZE, BOARD_SIZE))
        total_visits = sum(
            (child.visit_count if child else 0) 
            for child, _ in root.children.values()
        )
        
        if total_visits > 0:
            for move, (child, _) in root.children.items():
                if child:
                    i, j = move
                    probs[i, j] = child.visit_count / total_visits
        
        avg_value = root.value_sum / root.visit_count if root.visit_count > 0 else 0
        return avg_value, probs
    
    def get_best_move(self, root: Node) -> Optional[Tuple[int, int]]:
        """获取最佳落子位置"""
        best_visits = -1
        best_move = None
        
        for move, (child, _) in root.children.items():
            if child and child.visit_count > best_visits:
                best_visits = child.visit_count
                best_move = move
        
        return best_move
    
    def get_statistics(self, root: Node) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取搜索统计信息
        
        Returns:
            (visit_matrix, value_matrix): 访问次数矩阵和价值矩阵
        """
        visit_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        value_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        
        for move, (child, _) in root.children.items():
            if child:
                i, j = move
                visit_matrix[i, j] = child.visit_count
                value_matrix[i, j] = -child.q_value
        
        return visit_matrix, value_matrix
    
    def get_train_data(self) -> Tuple[List, List, List, List]:
        """
        获取训练数据 (与 nn_012.py 一致)
        
        Returns:
            (boards, policies, values, weights): 训练数据
        """
        boards, policies, values, weights = [], [], [], []
        
        for root in self.visited_nodes:
            # 过滤条件：有多个子节点或访问次数足够
            child_count = sum(1 for child, _ in root.children.values() if child is not None)
            if not (child_count > 1 or root.visit_count >= TRAIN_SIMULATION / 2):
                continue
            
            probs = np.zeros((BOARD_SIZE, BOARD_SIZE))
            total_visits = sum(
                (child.visit_count if child else 0) 
                for child, _ in root.children.values()
            )
            
            # 获取原始网络预测
            org_value, org_policy = get_network_prediction(self.model, root.board, self.device)
            
            # 归一化原始策略
            good = sum(
                org_policy[i][j] 
                for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)
                if root.board[i][j] == 0
            )
            if good < 1e-10:
                good = 1e-10
            
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if root.board[i][j] != 0 or org_policy[i][j] == 0:
                        probs[i, j] = 0
                    else:
                        org_policy[i][j] /= good
                        probs[i, j] = org_policy[i][j]
            
            # 收集已访问的子节点
            sum_used = 0
            moves = []
            for move, (child, prior) in root.children.items():
                if child is not None and child.visit_count > 0:
                    sum_used += probs[move]
                    moves.append((move[0], move[1], child, probs[move]))
                    probs[move] = 0
            
            # 按原始策略排序
            moves.sort(key=lambda x: x[3], reverse=True)
            
            # 计算加权值
            value_sum = 0
            for i in range(len(moves)):
                cnt = moves[i][3]
                if i + 1 < len(moves):
                    cnt -= moves[i + 1][3]
                if cnt == 0:
                    continue
                
                # 找最佳值
                best_val = -1e9
                best_pos = i
                for j in range(i + 1):
                    val = -moves[j][2].val
                    if val > best_val:
                        best_val = val
                        best_pos = j
                
                probs[moves[best_pos][0], moves[best_pos][1]] += cnt * (i + 1)
                if sum_used > 0:
                    value_sum += best_val * cnt * (i + 1) / sum_used
            
            boards.append(board_to_tensor(copy.deepcopy(root.board)))
            policies.append(torch.FloatTensor(probs))
            values.append(value_sum)
            weights.append(math.sqrt(total_visits / TRAIN_SIMULATION) * TRAIN_BUFF)
        
        return boards, policies, values, weights
