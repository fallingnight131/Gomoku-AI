"""
蒙特卡罗树搜索 (MCTS) 模块
实现AlphaZero风格的MCTS算法
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from game.board import Board
    from ai.network import PolicyValueNetwork


class Node:
    """
    MCTS树节点
    
    Attributes:
        N: 访问次数
        W: 累计价值
        Q: 平均价值 = W/N
        P: 先验概率（来自策略网络）
        children: 动作 -> 子节点的映射
        parent: 父节点
        action: 到达此节点的动作
    """
    
    def __init__(self, prior: float = 0.0, parent: Optional['Node'] = None, action: int = -1):
        """
        Args:
            prior: 先验概率
            parent: 父节点
            action: 到达此节点的动作
        """
        self.N = 0  # 访问次数
        self.W = 0.0  # 累计价值
        self.Q = 0.0  # 平均价值
        self.P = prior  # 先验概率
        self.children: Dict[int, Node] = {}
        self.parent = parent
        self.action = action
    
    def is_leaf(self) -> bool:
        """是否为叶节点（未展开）"""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """是否为根节点"""
        return self.parent is None
    
    def select_child(self, c_puct: float) -> Tuple[int, 'Node']:
        """
        选择UCB值最高的子节点
        
        UCB = Q + c_puct * P * sqrt(sum(parent.N)) / (1 + N)
        
        Args:
            c_puct: 探索常数
        
        Returns:
            (action, child_node)
        """
        total_visits = sum(child.N for child in self.children.values())
        sqrt_total = math.sqrt(total_visits + 1)
        
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in self.children.items():
            # UCB公式
            ucb = child.Q + c_puct * child.P * sqrt_total / (1 + child.N)
            
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, probs: np.ndarray, legal_actions: List[int]) -> None:
        """
        展开节点，创建子节点
        
        Args:
            probs: 策略网络输出的概率分布 (225,)
            legal_actions: 合法动作列表
        """
        for action in legal_actions:
            if action not in self.children:
                self.children[action] = Node(prior=probs[action], parent=self, action=action)
    
    def update(self, value: float) -> None:
        """
        更新节点统计信息
        
        Args:
            value: 回传的价值
        """
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
    
    def backup(self, value: float) -> None:
        """
        回溯更新，从当前节点到根节点
        
        Args:
            value: 叶节点评估值
        """
        node = self
        while node is not None:
            node.update(value)
            value = -value  # 交替玩家，价值取反
            node = node.parent


class MCTS:
    """
    蒙特卡罗树搜索
    
    实现AlphaZero风格的MCTS，使用神经网络引导搜索
    """
    
    def __init__(
        self,
        network: 'PolicyValueNetwork',
        simulations: int = 800,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.02
    ):
        """
        Args:
            network: 策略-价值网络
            simulations: 每次搜索的模拟次数
            c_puct: UCB探索常数
            dirichlet_alpha: Dirichlet噪声参数
            dirichlet_epsilon: Dirichlet噪声混合比例
        """
        self.network = network
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root: Optional[Node] = None
    
    def search(self, board: 'Board', add_noise: bool = True) -> Node:
        """
        执行MCTS搜索
        
        Args:
            board: 当前棋盘状态
            add_noise: 是否在根节点添加Dirichlet噪声
        
        Returns:
            搜索完成后的根节点
        """
        # 初始化根节点
        self.root = Node()
        
        # 获取根节点策略和价值
        state = board.encode_state()
        probs, _ = self.network.predict(state)
        legal_actions = board.get_legal_actions()
        
        # 归一化合法动作的概率
        legal_probs = np.zeros_like(probs)
        legal_probs[legal_actions] = probs[legal_actions]
        prob_sum = legal_probs.sum()
        if prob_sum > 0:
            legal_probs /= prob_sum
        else:
            # 均匀分布
            legal_probs[legal_actions] = 1.0 / len(legal_actions)
        
        # 添加Dirichlet噪声到根节点
        if add_noise and len(legal_actions) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for i, action in enumerate(legal_actions):
                legal_probs[action] = (1 - self.dirichlet_epsilon) * legal_probs[action] + \
                                       self.dirichlet_epsilon * noise[i]
        
        # 展开根节点
        self.root.expand(legal_probs, legal_actions)
        
        # 执行模拟
        for _ in range(self.simulations):
            self._simulate(board.copy())
        
        return self.root
    
    def _simulate(self, board: 'Board') -> None:
        """
        执行一次模拟
        
        包括Selection, Expansion, Evaluation, Backpropagation
        """
        node = self.root
        
        # Selection: 选择到叶节点
        while not node.is_leaf():
            action, node = node.select_child(self.c_puct)
            board.move_by_action(action)
        
        # 检查游戏是否结束
        if board.is_game_over():
            # 游戏结束，直接获取结果
            if board.winner == 0:
                value = 0.0  # 平局
            else:
                # 当前玩家视角的价值
                value = -1.0  # 对手赢了（因为最后一步是对手下的）
        else:
            # Expansion & Evaluation: 展开并评估
            state = board.encode_state()
            probs, value = self.network.predict(state)
            
            legal_actions = board.get_legal_actions()
            
            # 归一化合法动作的概率
            legal_probs = np.zeros_like(probs)
            legal_probs[legal_actions] = probs[legal_actions]
            prob_sum = legal_probs.sum()
            if prob_sum > 0:
                legal_probs /= prob_sum
            else:
                legal_probs[legal_actions] = 1.0 / len(legal_actions)
            
            # 展开节点
            node.expand(legal_probs, legal_actions)
        
        # Backpropagation: 回溯更新
        node.backup(-value)  # 取反因为是从对手视角
    
    def get_action_probs(self, board: 'Board', temperature: float = 1.0, add_noise: bool = True) -> np.ndarray:
        """
        获取动作概率分布
        
        Args:
            board: 当前棋盘状态
            temperature: 温度参数，控制探索程度
                - temp=1.0: 按访问次数比例采样
                - temp→0: 选择访问次数最多的动作
            add_noise: 是否添加Dirichlet噪声
        
        Returns:
            动作概率分布 (225,)
        """
        # 执行搜索
        self.search(board, add_noise=add_noise)
        
        # 计算概率分布
        action_visits = np.zeros(board.size * board.size)
        for action, child in self.root.children.items():
            action_visits[action] = child.N
        
        if temperature == 0:
            # 选择访问次数最多的
            best_action = np.argmax(action_visits)
            probs = np.zeros_like(action_visits)
            probs[best_action] = 1.0
        else:
            # 按温度缩放
            visits_temp = action_visits ** (1.0 / temperature)
            total = visits_temp.sum()
            if total > 0:
                probs = visits_temp / total
            else:
                probs = action_visits
        
        return probs
    
    def get_best_action(self, board: 'Board', add_noise: bool = False) -> int:
        """
        获取最佳动作
        
        Args:
            board: 当前棋盘状态
            add_noise: 是否添加噪声
        
        Returns:
            最佳动作编号
        """
        probs = self.get_action_probs(board, temperature=0, add_noise=add_noise)
        return int(np.argmax(probs))
    
    def get_visit_counts(self) -> Dict[int, int]:
        """获取根节点子节点的访问次数"""
        if self.root is None:
            return {}
        return {action: child.N for action, child in self.root.children.items()}
    
    def get_root_q_value(self) -> float:
        """获取根节点的Q值（AI对当前局面的评估）"""
        if self.root is None:
            return 0.0
        return self.root.Q


class RandomPlayer:
    """随机玩家，用于测试"""
    
    def get_action(self, board: 'Board') -> int:
        """随机选择一个合法动作"""
        legal_actions = board.get_legal_actions()
        return np.random.choice(legal_actions)


class MCTSPlayer:
    """MCTS玩家"""
    
    def __init__(
        self,
        network: 'PolicyValueNetwork',
        simulations: int = 800,
        c_puct: float = 2.0,
        temperature: float = 0.0
    ):
        """
        Args:
            network: 策略-价值网络
            simulations: MCTS模拟次数
            c_puct: UCB探索常数
            temperature: 动作选择温度
        """
        self.mcts = MCTS(network, simulations=simulations, c_puct=c_puct)
        self.temperature = temperature
    
    def get_action(self, board: 'Board') -> int:
        """选择动作"""
        probs = self.mcts.get_action_probs(board, temperature=self.temperature, add_noise=False)
        return int(np.argmax(probs))
    
    def get_action_with_probs(self, board: 'Board', temperature: float = 1.0) -> Tuple[int, np.ndarray]:
        """选择动作并返回概率分布"""
        probs = self.mcts.get_action_probs(board, temperature=temperature, add_noise=True)
        
        if temperature == 0:
            action = int(np.argmax(probs))
        else:
            action = np.random.choice(len(probs), p=probs)
        
        return action, probs


if __name__ == '__main__':
    # 测试MCTS
    import sys
    sys.path.insert(0, '..')
    
    from game.board import Board
    from network import PolicyValueNetwork
    
    # 创建网络和棋盘
    network = PolicyValueNetwork(num_res_blocks=5)
    board = Board()
    
    # 创建MCTS
    mcts = MCTS(network, simulations=100, c_puct=2.0)
    
    # 获取动作概率
    probs = mcts.get_action_probs(board, temperature=1.0)
    
    print("动作概率分布（前10个）:", probs[:10])
    print("概率和:", probs.sum())
    
    # 获取最佳动作
    best_action = mcts.get_best_action(board)
    x, y = best_action // 15, best_action % 15
    print(f"最佳动作: ({x}, {y})")
    
    # 访问次数
    visits = mcts.get_visit_counts()
    print(f"总访问次数: {sum(visits.values())}")
