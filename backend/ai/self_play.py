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
    
    def self_play_one_game(self, verbose: bool = False) -> GameRecord:
        """
        进行一局自我对弈
        
        Args:
            verbose: 是否打印详细信息
        
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
            
            if verbose:
                print(f"Step {step + 1}: Player {3 - board.current_player} -> ({x}, {y})")
            
            step += 1
        
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
        
        for i in range(num_games):
            start_time = time.time()
            
            record = self.self_play_one_game(verbose=False)
            game_data = record.to_training_data(augment=augment)
            all_data.extend(game_data)
            
            elapsed = time.time() - start_time
            
            if verbose:
                winner_str = 'Black' if record.winner == 1 else 'White' if record.winner == 2 else 'Draw'
                print(f"Game {i + 1}/{num_games}: {len(record.states)} moves, "
                      f"Winner: {winner_str}, Time: {elapsed:.1f}s, "
                      f"Data: {len(game_data)} samples")
        
        return all_data


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
