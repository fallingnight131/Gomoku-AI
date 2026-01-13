"""
训练模块
实现自我对弈、网络训练、模型评估的完整循环
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import argparse
import json
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board
from ai.network import PolicyValueNetwork, PolicyValueNetworkSmall
from ai.mcts import MCTS, MCTSPlayer, RandomPlayer
from ai.self_play import SelfPlayWorker, ReplayBuffer


class Trainer:
    """
    训练器
    管理整个训练流程
    """
    
    def __init__(
        self,
        model_dir: str = 'models',
        use_small_network: bool = False,
        device: str = 'auto'
    ):
        """
        Args:
            model_dir: 模型保存目录
            use_small_network: 是否使用小型网络
            device: 计算设备 ('auto', 'cpu', 'cuda')
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # 设置设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"使用设备: {self.device}")
        
        # 创建网络
        if use_small_network:
            self.network = PolicyValueNetworkSmall()
        else:
            self.network = PolicyValueNetwork()
        
        self.network.to(self.device)
        print(f"网络参数量: {self.network.count_parameters():,}")
        
        # 最佳模型
        self.best_network: Optional[PolicyValueNetwork] = None
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(max_size=50000)
        
        # 训练统计
        self.stats = {
            'iteration': 0,
            'total_games': 0,
            'win_rate_vs_random': 0.0,
            'losses': [],
            'best_model_iteration': 0
        }
    
    def train(
        self,
        iterations: int = 100,
        episodes_per_iteration: int = 100,
        simulations: int = 800,
        batch_size: int = 256,
        epochs_per_iteration: int = 5,
        lr: float = 0.001,
        lr_decay: float = 0.1,
        lr_decay_steps: int = 50,
        eval_games: int = 20,
        save_interval: int = 5,
        verbose: bool = True
    ):
        """
        主训练循环
        
        Args:
            iterations: 总迭代次数
            episodes_per_iteration: 每轮自我对弈局数
            simulations: MCTS模拟次数
            batch_size: 训练批次大小
            epochs_per_iteration: 每轮训练epoch数
            lr: 初始学习率
            lr_decay: 学习率衰减因子
            lr_decay_steps: 学习率衰减步数
            eval_games: 评估对局数
            save_interval: 保存间隔
            verbose: 是否打印详细信息
        """
        # 优化器
        optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=lr_decay)
        
        for iteration in range(1, iterations + 1):
            self.stats['iteration'] = iteration
            print(f"\n{'='*60}")
            print(f"迭代 {iteration}/{iterations}")
            print(f"{'='*60}")
            
            # Phase 1: 自我对弈
            print("\n[Phase 1] 自我对弈生成数据...")
            start_time = time.time()
            
            self_play_worker = SelfPlayWorker(
                self.network,
                simulations=simulations,
                temp_threshold=10
            )
            
            new_data = self_play_worker.self_play_games(
                num_games=episodes_per_iteration,
                augment=True,
                verbose=verbose
            )
            
            self.replay_buffer.add(new_data)
            self.stats['total_games'] += episodes_per_iteration
            
            elapsed = time.time() - start_time
            print(f"生成 {len(new_data)} 样本, 用时 {elapsed:.1f}s")
            print(f"缓冲区大小: {len(self.replay_buffer)}")
            
            # Phase 2: 网络训练
            print("\n[Phase 2] 训练神经网络...")
            start_time = time.time()
            
            losses = self._train_network(
                optimizer,
                batch_size=batch_size,
                epochs=epochs_per_iteration
            )
            
            self.stats['losses'].extend(losses)
            scheduler.step()
            
            elapsed = time.time() - start_time
            avg_loss = np.mean(losses)
            print(f"平均损失: {avg_loss:.4f}, 用时 {elapsed:.1f}s")
            print(f"当前学习率: {scheduler.get_last_lr()[0]:.6f}")
            
            # Phase 3: 模型评估
            if iteration % save_interval == 0 or iteration == 1:
                print("\n[Phase 3] 评估模型...")
                win_rate = self._evaluate_vs_random(num_games=eval_games)
                self.stats['win_rate_vs_random'] = win_rate
                print(f"对随机玩家胜率: {win_rate*100:.1f}%")
                
                # 保存检查点
                self._save_checkpoint(iteration)
                
                # 更新最佳模型
                if self.best_network is None or win_rate > 0.55:
                    self._update_best_model(iteration)
            
            # 保存统计信息
            self._save_stats()
        
        print("\n训练完成!")
        print(f"总游戏数: {self.stats['total_games']}")
        print(f"最终对随机玩家胜率: {self.stats['win_rate_vs_random']*100:.1f}%")
    
    def _train_network(
        self,
        optimizer: optim.Optimizer,
        batch_size: int,
        epochs: int
    ) -> List[float]:
        """
        训练网络
        
        Returns:
            损失列表
        """
        self.network.train()
        losses = []
        
        for epoch in range(epochs):
            if len(self.replay_buffer) < batch_size:
                continue
            
            # 采样
            states, probs, values = self.replay_buffer.sample(batch_size)
            
            # 转换为张量
            states_tensor = torch.FloatTensor(states).to(self.device)
            probs_tensor = torch.FloatTensor(probs).to(self.device)
            values_tensor = torch.FloatTensor(values).unsqueeze(1).to(self.device)
            
            # 前向传播
            log_probs, pred_values = self.network(states_tensor)
            
            # 计算损失
            # 策略损失: 交叉熵
            policy_loss = -torch.mean(torch.sum(probs_tensor * log_probs, dim=1))
            
            # 价值损失: MSE
            value_loss = nn.functional.mse_loss(pred_values, values_tensor)
            
            # 总损失
            total_loss = policy_loss + value_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
        
        return losses
    
    def _evaluate_vs_random(self, num_games: int = 20) -> float:
        """
        与随机玩家对战评估
        
        Returns:
            胜率
        """
        self.network.eval()
        wins = 0
        draws = 0
        
        mcts_player = MCTSPlayer(self.network, simulations=200, temperature=0)
        random_player = RandomPlayer()
        
        for game_idx in range(num_games):
            board = Board()
            
            # 交替先后手
            if game_idx % 2 == 0:
                players = [mcts_player, random_player]  # AI先手
                ai_color = 1
            else:
                players = [random_player, mcts_player]  # AI后手
                ai_color = 2
            
            current = 0
            while not board.is_game_over():
                if isinstance(players[current], MCTSPlayer):
                    action = players[current].get_action(board)
                else:
                    action = players[current].get_action(board)
                
                x, y = action // 15, action % 15
                board.move(x, y)
                current = 1 - current
            
            winner = board.get_winner()
            if winner == ai_color:
                wins += 1
            elif winner == 0:
                draws += 1
        
        return wins / num_games
    
    def _save_checkpoint(self, iteration: int) -> None:
        """保存检查点"""
        path = os.path.join(self.model_dir, f'checkpoint_{iteration}.pth')
        self.network.save(path)
        print(f"保存检查点: {path}")
    
    def _update_best_model(self, iteration: int) -> None:
        """更新最佳模型"""
        path = os.path.join(self.model_dir, 'best_model.pth')
        self.network.save(path)
        self.stats['best_model_iteration'] = iteration
        print(f"更新最佳模型 (迭代 {iteration})")
    
    def _save_stats(self) -> None:
        """保存训练统计信息"""
        path = os.path.join(self.model_dir, 'training_stats.json')
        
        # 转换为可序列化格式
        stats_to_save = {
            'iteration': self.stats['iteration'],
            'total_games': self.stats['total_games'],
            'win_rate_vs_random': self.stats['win_rate_vs_random'],
            'best_model_iteration': self.stats['best_model_iteration'],
            'avg_recent_loss': float(np.mean(self.stats['losses'][-100:])) if self.stats['losses'] else 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        self.network = PolicyValueNetwork.load(path, device=self.device)
        print(f"加载检查点: {path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='五子棋AI训练')
    parser.add_argument('--iterations', type=int, default=100, help='训练迭代次数')
    parser.add_argument('--episodes', type=int, default=10, help='每轮自我对弈局数')
    parser.add_argument('--simulations', type=int, default=400, help='MCTS模拟次数')
    parser.add_argument('--batch-size', type=int, default=256, help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=5, help='每轮训练epoch数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--model-dir', type=str, default='models', help='模型保存目录')
    parser.add_argument('--small-network', action='store_true', help='使用小型网络')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = Trainer(
        model_dir=args.model_dir,
        use_small_network=args.small_network,
        device=args.device
    )
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train(
        iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        simulations=args.simulations,
        batch_size=args.batch_size,
        epochs_per_iteration=args.epochs,
        lr=args.lr,
        verbose=True
    )


if __name__ == '__main__':
    main()
