"""
自我对弈模块
生成训练数据
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time
import sys
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board
from ai.mcts import MCTS, MCTSPlayer
from ai.network import PolicyValueNetwork


@dataclass
class GameRecord:
    """一局游戏的记录"""
    states: List[np.ndarray]  # 棋盘状态列表
    mcts_probs: List[np.ndarray]  # MCTS概率分布列表
    winner: int  # 获胜者 1=黑 2=白 0=平局
    
    def to_training_data(self, augment: bool = True) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        转换为训练数据
        
        Args:
            augment: 是否进行数据增强
        
        Returns:
            [(state, probs, value), ...] 训练数据列表
        """
        data = []
        
        for i, (state, probs) in enumerate(zip(self.states, self.mcts_probs)):
            # 计算当前玩家视角的结果
            # 偶数步是黑方(1)，奇数步是白方(2)
            current_player = 1 if i % 2 == 0 else 2
            
            if self.winner == 0:
                value = 0.0  # 平局
            elif self.winner == current_player:
                value = 1.0  # 当前玩家赢
            else:
                value = -1.0  # 当前玩家输
            
            if augment:
                # 数据增强：8种对称变换
                for aug_state, aug_probs in self._get_symmetries(state, probs):
                    data.append((aug_state, aug_probs, value))
            else:
                data.append((state, probs, value))
        
        return data
    
    def _get_symmetries(self, state: np.ndarray, probs: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """获取棋盘状态的对称变换"""
        size = state.shape[1]
        probs_2d = probs.reshape(size, size)
        
        symmetries = []
        
        for i in range(4):
            # 旋转
            rotated_state = np.array([np.rot90(s, i) for s in state])
            rotated_probs = np.rot90(probs_2d, i).flatten()
            symmetries.append((rotated_state, rotated_probs))
            
            # 水平翻转后旋转
            flipped_state = np.array([np.fliplr(np.rot90(s, i)) for s in state])
            flipped_probs = np.fliplr(np.rot90(probs_2d, i)).flatten()
            symmetries.append((flipped_state, flipped_probs))
        
        return symmetries


class SelfPlayWorker:
    """
    自我对弈工作器
    """
    
    def __init__(
        self,
        network: PolicyValueNetwork,
        simulations: int = 800,
        c_puct: float = 2.0,
        temp_threshold: int = 10
    ):
        """
        Args:
            network: 策略-价值网络
            simulations: MCTS模拟次数
            c_puct: UCB探索常数
            temp_threshold: 温度阈值，前N步使用温度1.0，之后使用0.1
        """
        self.network = network
        self.simulations = simulations
        self.c_puct = c_puct
        self.temp_threshold = temp_threshold
    
    def self_play_one_game(self, verbose: bool = False, pbar=None) -> GameRecord:
        """
        进行一局自我对弈
        
        Args:
            verbose: 是否打印详细信息
            pbar: tqdm进度条对象，用于实时更新步数
        
        Returns:
            游戏记录
        """
        board = Board()
        mcts = MCTS(
            self.network,
            simulations=self.simulations,
            c_puct=self.c_puct
        )
        
        states = []
        mcts_probs_list = []
        step = 0
        
        while not board.is_game_over():
            # 确定温度
            if step < self.temp_threshold:
                temperature = 1.0
            else:
                temperature = 0.1
            
            # 获取MCTS概率分布
            probs = mcts.get_action_probs(board, temperature=temperature, add_noise=True)
            
            # 保存状态和概率
            states.append(board.encode_state())
            mcts_probs_list.append(probs)
            
            # 选择动作
            if temperature == 0:
                action = np.argmax(probs)
            else:
                action = np.random.choice(len(probs), p=probs)
            
            # 执行动作
            x, y = action // board.size, action % board.size
            board.move(x, y)
            
            step += 1
            
            # 实时更新进度条显示当前步数
            if pbar is not None:
                pbar.set_postfix_str(f"第{step}步")
            elif verbose:
                print(f"Step {step}: Player {3 - board.current_player} -> ({x}, {y})")
            
        
        winner = board.get_winner()
        
        if verbose:
            print(f"Game over! Winner: {'Black' if winner == 1 else 'White' if winner == 2 else 'Draw'}")
            print(f"Total steps: {step}")
        
        return GameRecord(
            states=states,
            mcts_probs=mcts_probs_list,
            winner=winner
        )
    
    def self_play_games(
        self,
        num_games: int,
        augment: bool = True,
        verbose: bool = False
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        进行多局自我对弈
        
        Args:
            num_games: 游戏局数
            augment: 是否数据增强
            verbose: 是否打印详细信息
        
        Returns:
            所有游戏的训练数据
        """
        all_data = []
        
        pbar = tqdm(range(num_games), desc="  自我对弈", leave=False, unit="局")
        for i in pbar:
            record = self.self_play_one_game(verbose=False, pbar=pbar)
            game_data = record.to_training_data(augment=augment)
            all_data.extend(game_data)
            
            winner_str = '黑胜' if record.winner == 1 else '白胜' if record.winner == 2 else '平'
            pbar.set_postfix_str(f"{winner_str} {len(record.states)}步")
        
        pbar.close()
        return all_data
    
    def self_play_games_parallel(
        self,
        num_games: int,
        num_workers: int = None,
        augment: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        并行进行多局自我对弈
        
        Args:
            num_games: 游戏局数
            num_workers: 并行进程数，默认为CPU核数的一半
            augment: 是否数据增强
        
        Returns:
            所有游戏的训练数据
        """
        if num_workers is None:
            num_workers = max(1, cpu_count() // 2)
        
        # 限制worker数量不超过游戏数
        num_workers = min(num_workers, num_games)
        
        if num_workers <= 1:
            # 单进程模式
            return self.self_play_games(num_games, augment=augment, verbose=False)
        
        print(f"  🔄 并行自我对弈: {num_workers} 进程, {num_games} 局")
        
        # 准备参数
        args = []
        for i in range(num_games):
            args.append((
                self.network.state_dict(),
                type(self.network).__name__,
                self.simulations,
                self.c_puct,
                self.temp_threshold,
                augment
            ))
        
        # 并行执行
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(_play_one_game_worker, args),
                total=num_games,
                desc="  自我对弈",
                leave=False,
                unit="局"
            ))
        
        # 合并所有数据
        all_data = []
        for game_data in results:
            all_data.extend(game_data)
        
        return all_data


def _play_one_game_worker(args) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    单个自我对弈的worker函数（用于多进程）
    
    Args:
        args: (network_state_dict, network_class_name, simulations, c_puct, temp_threshold, augment)
    
    Returns:
        一局游戏的训练数据
    """
    state_dict, network_class, simulations, c_puct, temp_threshold, augment = args
    
    # 在子进程中重建网络
    from ai.network import PolicyValueNetwork, PolicyValueNetworkSmall
    
    if network_class == 'PolicyValueNetworkSmall':
        network = PolicyValueNetworkSmall()
    else:
        network = PolicyValueNetwork()
    
    network.load_state_dict(state_dict)
    network.eval()
    
    # 创建worker并执行一局游戏
    worker = SelfPlayWorker(network, simulations=simulations, c_puct=c_puct, temp_threshold=temp_threshold)
    record = worker.self_play_one_game(verbose=False)
    
    return record.to_training_data(augment=augment)


def _evaluate_game_worker(args) -> Tuple[int, int]:
    """
    评估对局的worker函数（用于多进程）
    
    Args:
        args: (network1_state_dict, network2_state_dict, network1_class, network2_class,
               simulations, game_idx, is_vs_random)
        network2_state_dict: None表示对手是随机玩家
    
    Returns:
        (player1结果, game_idx)
        结果: 1=胜, 0=平, -1=负
    """
    (network1_state, network2_state, network1_class, network2_class,
     simulations, game_idx, is_vs_random) = args
    
    from ai.network import PolicyValueNetwork, PolicyValueNetworkSmall
    from ai.mcts import MCTSPlayer, RandomPlayer
    from game.board import Board
    
    # 创建网络1
    if network1_class == 'PolicyValueNetworkSmall':
        network1 = PolicyValueNetworkSmall()
    else:
        network1 = PolicyValueNetwork()
    network1.load_state_dict(network1_state)
    network1.eval()
    
    player1 = MCTSPlayer(network1, simulations=simulations, temperature=0)
    
    # 创建对手
    if is_vs_random:
        player2 = RandomPlayer()
    else:
        if network2_class == 'PolicyValueNetworkSmall':
            network2 = PolicyValueNetworkSmall()
        else:
            network2 = PolicyValueNetwork()
        network2.load_state_dict(network2_state)
        network2.eval()
        player2 = MCTSPlayer(network2, simulations=simulations, temperature=0)
    
    # 开始对局
    board = Board()
    
    # 交替先后手
    if game_idx % 2 == 0:
        players = [player1, player2]  # player1先手
        p1_color = 1
    else:
        players = [player2, player1]  # player1后手
        p1_color = 2
    
    current = 0
    while not board.is_game_over():
        action = players[current].get_action(board)
        x, y = action // 15, action % 15
        board.move(x, y)
        current = 1 - current
    
    winner = board.get_winner()
    if winner == p1_color:
        result = 1  # 胜
    elif winner == 0:
        result = 0  # 平
    else:
        result = -1  # 负
    
    return (result, game_idx)


def evaluate_games_parallel(
    network1_state_dict: dict,
    network2_state_dict: Optional[dict],
    network1_class: str,
    network2_class: str = None,
    num_games: int = 10,
    num_workers: int = 1,
    simulations: int = 100,
    desc: str = "评估"
) -> Tuple[int, int, int]:
    """
    并行执行评估对局
    
    Args:
        network1_state_dict: 网络1参数
        network2_state_dict: 网络2参数，None表示对手是随机玩家
        network1_class: 网络1类名
        network2_class: 网络2类名（如果不同于网络1）
        num_games: 对局数
        num_workers: 进程数
        simulations: MCTS模拟次数
        desc: 进度条描述
    
    Returns:
        (胜, 负, 平)
    """
    is_vs_random = network2_state_dict is None
    
    # 如果没指定 network2_class，默认与 network1_class 相同
    if network2_class is None:
        network2_class = network1_class
    
    # 准备参数
    args = []
    for i in range(num_games):
        args.append((
            network1_state_dict,
            network2_state_dict,
            network1_class,
            network2_class,
            simulations,
            i,
            is_vs_random
        ))
    
    wins = 0
    losses = 0
    draws = 0
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_evaluate_game_worker, args),
            total=num_games,
            desc=desc,
            leave=False,
            unit="局"
        ))
    
    for result, _ in results:
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
    
    return wins, losses, draws


class ReplayBuffer:
    """
    经验回放缓冲区
    """
    
    def __init__(self, max_size: int = 50000):
        """
        Args:
            max_size: 最大容量
        """
        self.max_size = max_size
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []
    
    def add(self, data: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        """添加数据"""
        self.buffer.extend(data)
        
        # 保持大小限制
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        采样一个批次
        
        Returns:
            (states, probs, values)
        """
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        
        states = np.array([self.buffer[i][0] for i in indices])
        probs = np.array([self.buffer[i][1] for i in indices])
        values = np.array([self.buffer[i][2] for i in indices])
        
        return states, probs, values
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.buffer.clear()
    
    def save(self, path: str) -> None:
        """
        保存经验池到文件
        
        Args:
            path: 保存路径
        """
        import pickle
        
        # 将数据转换为更紧凑的格式
        data = {
            'max_size': self.max_size,
            'buffer': self.buffer
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ 经验池已保存: {path} ({len(self.buffer)} 条数据)")
    
    def load(self, path: str) -> bool:
        """
        从文件加载经验池
        
        Args:
            path: 加载路径
        
        Returns:
            是否加载成功
        """
        import pickle
        import os
        
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.max_size = data.get('max_size', self.max_size)
            self.buffer = data.get('buffer', [])
            
            print(f"✓ 经验池已加载: {path} ({len(self.buffer)} 条数据)")
            return True
        except Exception as e:
            print(f"⚠ 加载经验池失败: {e}")
            return False


if __name__ == '__main__':
    # 测试自我对弈
    from network import PolicyValueNetworkSmall
    
    print("创建网络...")
    network = PolicyValueNetworkSmall()
    
    print("开始自我对弈测试...")
    worker = SelfPlayWorker(network, simulations=50)  # 减少模拟次数加快测试
    
    record = worker.self_play_one_game(verbose=True)
    print(f"\n游戏记录: {len(record.states)} 步")
    
    training_data = record.to_training_data(augment=True)
    print(f"训练数据: {len(training_data)} 样本")
