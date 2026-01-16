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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board
from ai.network import PolicyValueNetwork, PolicyValueNetworkSmall
from ai.mcts import MCTS, MCTSPlayer, RandomPlayer
from ai.self_play import SelfPlayWorker, ReplayBuffer, evaluate_games_parallel


class Trainer:
    """
    训练器
    管理整个训练流程
    """
    
    def __init__(
        self,
        model_dir: str = 'models',
        data_dir: str = 'data',
        use_small_network: bool = False,
        device: str = 'auto',
        buffer_size: int = 50000
    ):
        """
        Args:
            model_dir: 模型保存目录
            data_dir: 数据保存目录（经验池等）
            use_small_network: 是否使用小型网络
            device: 计算设备 ('auto', 'cpu', 'cuda', 'hybrid')
                   'hybrid' = GPU训练 + CPU自我对弈/评估（推荐有GPU时使用）
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        self.use_small_network = use_small_network
        
        # 设置设备
        self.hybrid_mode = (device == 'hybrid')
        
        if device == 'hybrid':
            # 混合模式：GPU训练，CPU推理
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.inference_device = 'cpu'
                device_name = torch.cuda.get_device_name(0)
                print(f"🚀 混合模式: GPU训练 ({device_name}) + CPU自我对弈/评估")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
                self.inference_device = 'cpu'
                print(f"🚀 混合模式: MPS训练 + CPU自我对弈/评估")
            else:
                self.device = 'cpu'
                self.inference_device = 'cpu'
                self.hybrid_mode = False
                print(f"💻 使用设备: CPU (未检测到GPU，无法使用混合模式)")
        elif device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                device_name = torch.cuda.get_device_name(0)
                print(f"🚀 使用设备: GPU ({device_name})")
            else:
                self.device = 'cpu'
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    print(f"💻 使用设备: CPU (提示: 检测到MPS可用但MCTS场景下CPU更快)")
                else:
                    print(f"💻 使用设备: CPU")
            self.inference_device = self.device
        else:
            self.device = device
            self.inference_device = device
            print(f"使用设备: {self.device}")
        
        # 创建网络
        if use_small_network:
            self.network = PolicyValueNetworkSmall()
        else:
            self.network = PolicyValueNetwork()
        
        self.network.to(self.device)
        print(f"网络参数量: {self.network.count_parameters():,}")
        
        # 最佳模型 - 尝试加载已有的best_model
        self.best_network: Optional[PolicyValueNetwork] = None
        self._load_best_model_if_exists(use_small_network)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        
        # 训练统计
        self.stats = {
            'iteration': 0,
            'total_games': 0,
            'win_rate_vs_random': 0.0,
            'losses': [],
            'best_model_iteration': 0
        }
    
    def _get_cpu_network(self) -> PolicyValueNetwork:
        """
        获取CPU版本的网络（用于自我对弈和评估）
        将当前训练网络的权重复制到CPU网络
        """
        if self.use_small_network:
            cpu_network = PolicyValueNetworkSmall()
        else:
            cpu_network = PolicyValueNetwork()
        
        cpu_network.load_state_dict(self.network.state_dict())
        cpu_network.to('cpu')
        cpu_network.eval()
        return cpu_network
    
    def _get_cpu_best_network(self) -> Optional[PolicyValueNetwork]:
        """获取CPU版本的最佳网络"""
        if self.best_network is None:
            return None
        
        if self.use_small_network:
            cpu_network = PolicyValueNetworkSmall()
        else:
            cpu_network = PolicyValueNetwork()
        
        cpu_network.load_state_dict(self.best_network.state_dict())
        cpu_network.to('cpu')
        cpu_network.eval()
        return cpu_network
    
    def _load_best_model_if_exists(self, use_small_network: bool) -> None:
        """尝试加载已有的最佳模型"""
        best_path = os.path.join(self.model_dir, 'best_model.pth')
        if os.path.exists(best_path):
            try:
                checkpoint = torch.load(best_path, map_location=self.device)
                
                # 根据保存的配置创建网络
                num_blocks = checkpoint.get('num_res_blocks', 10)
                num_channels = checkpoint.get('num_channels', 64)
                
                if num_blocks <= 5 or use_small_network:
                    self.best_network = PolicyValueNetworkSmall()
                else:
                    self.best_network = PolicyValueNetwork(
                        num_channels=num_channels,
                        num_res_blocks=num_blocks
                    )
                
                self.best_network.load_state_dict(checkpoint['model_state_dict'])
                self.best_network.to(self.device)
                self.best_network.eval()
                print(f"✓ 加载已有最佳模型: {best_path}")
            except Exception as e:
                print(f"⚠ 无法加载已有模型: {e}，将从头开始训练")
                self.best_network = None
        else:
            print("未找到已有最佳模型，将从头开始训练")
    
    def train(
        self,
        iterations: int = 100,
        episodes_per_iteration: int = 100,
        simulations: int = 800,
        batch_size: int = 256,
        epochs_per_iteration: int = 5,
        lr: float = 0.001,
        lr_min: float = 1e-5,
        eval_games: int = 20,
        save_interval: int = 5,
        num_workers: int = 1,
        verbose: bool = True,
        start_iteration: int = 0,
        optimizer_state: dict = None,
        scheduler_state: dict = None
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
            lr_min: 最低学习率（余弦退火终点）
            eval_games: 评估对局数
            save_interval: 保存间隔
            num_workers: 并行自我对弈的进程数 (1=不并行)
            verbose: 是否打印详细信息
            start_iteration: 起始迭代数（用于断点续训）
            optimizer_state: 优化器状态（用于断点续训）
            scheduler_state: 调度器状态（用于断点续训）
        """
        # 优化器 + ReduceLROnPlateau 调度器
        optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,          # 每次减半
            patience=5           # 5轮不降再降lr
        )
        
        # 恢复优化器和调度器状态
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            print(f"✓ 恢复优化器状态")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            print(f"✓ 恢复学习率调度器状态")
        
        # 计算实际迭代范围
        actual_start = start_iteration + 1
        actual_end = start_iteration + iterations + 1
        
        if start_iteration > 0:
            print(f"📌 从迭代 {actual_start} 继续训练，目标迭代 {actual_end - 1}")
        
        # 精确计算总局数（考虑 eval_interval）
        # 评估轮：第1轮 + 每隔 save_interval 的轮
        eval_random_games = 10
        num_eval_iterations = 1  # 第1轮
        for i in range(save_interval, iterations + 1, save_interval):
            num_eval_iterations += 1
        
        # 总局数 = 所有轮的自我对弈 + 评估轮的评估对局
        total_self_play = iterations * episodes_per_iteration
        total_eval = num_eval_iterations * (eval_random_games + eval_games)
        total_games_all = total_self_play + total_eval
        total_iters = iterations  # 总轮数
        
        # 总体进度条（在进度条和时间之间显示轮数）
        pbar = tqdm(total=total_games_all, desc="🎮 训练进度", unit="局",
                    bar_format='{l_bar}{bar}| {postfix}')
        
        for iteration in range(actual_start, actual_end):
            self.stats['iteration'] = iteration
            # 更新：轮数 + 时间 + 当前阶段
            pbar.set_postfix_str(f"{iteration}/{total_iters}轮 自我对弈 0/{episodes_per_iteration}局")
            
            # Phase 1: 自我对弈
            start_time = time.time()
            
            # 混合模式：自我对弈使用CPU版本的网络
            if self.hybrid_mode:
                inference_network = self._get_cpu_network()
            else:
                inference_network = self.network
            
            self_play_worker = SelfPlayWorker(
                inference_network,
                simulations=simulations,
                temp_threshold=10
            )
            
            # 进度回调：每完成一局更新进度条
            last_completed = [0]
            def self_play_progress(completed, total):
                delta = completed - last_completed[0]
                last_completed[0] = completed
                pbar.update(delta)
                pbar.set_postfix_str(f"自我对弈 {completed}/{total}局")
            
            # 根据进程数选择并行或串行
            if num_workers > 1:
                new_data = self_play_worker.self_play_games_parallel(
                    num_games=episodes_per_iteration,
                    num_workers=num_workers,
                    augment=True,
                    progress_callback=self_play_progress
                )
            else:
                # 单进程模式也需要更新进度
                new_data = []
                for game_idx in range(episodes_per_iteration):
                    game_data = self_play_worker.self_play_games(num_games=1, augment=True, verbose=False, show_progress=False)
                    new_data.extend(game_data)
                    pbar.update(1)
                    pbar.set_postfix_str(f"自我对弈 {game_idx+1}/{episodes_per_iteration}局")
            
            self.replay_buffer.add(new_data)
            self.stats['total_games'] += episodes_per_iteration
            
            elapsed = time.time() - start_time
            
            # Phase 2: 网络训练
            pbar.set_postfix_str("训练网络中...")
            start_time = time.time()
            
            losses = self._train_network(
                optimizer,
                batch_size=batch_size,
                epochs=epochs_per_iteration
            )
            
            self.stats['losses'].extend(losses)
            elapsed = time.time() - start_time
            avg_loss = np.mean(losses)
            scheduler.step(avg_loss)
            
            if verbose:
                tqdm.write(f"[迭代 {iteration}] 自我对弈: {len(new_data)} 样本, {elapsed:.1f}s, loss={avg_loss:.4f}")
            
            # Phase 3: 模型评估（只在特定轮次执行）
            is_eval_iteration = (iteration % save_interval == 0 or iteration == 1)
            
            # 评估时使用的进程数（与自我对弈共享设置）
            eval_workers = num_workers if num_workers > 1 else 1
            
            if is_eval_iteration:
                # 实际执行评估
                # vs 随机
                last_completed[0] = 0
                def eval_random_progress(completed, total, wins, losses_count, draws):
                    delta = completed - last_completed[0]
                    last_completed[0] = completed
                    pbar.update(delta)
                    pbar.set_postfix_str(f"vs随机 {completed}/{total}局 胜{wins}")
                
                win_rate_random = self._evaluate_vs_random(
                    num_games=eval_random_games, 
                    num_workers=eval_workers, 
                    pbar=None,  # 不传pbar，用回调
                    progress_callback=eval_random_progress
                )
                self.stats['win_rate_vs_random'] = win_rate_random
                
                # 保存检查点
                self._save_checkpoint(iteration, optimizer, scheduler)
                
                # vs 最佳
                if self.best_network is None:
                    # 首次训练，直接保存，跳过vs最佳的进度
                    pbar.update(eval_games)
                    self._update_best_model(iteration)
                    tqdm.write(f"[迭代 {iteration}] ✓ 初始最佳模型已保存")
                else:
                    last_completed[0] = 0
                    def eval_best_progress(completed, total, wins, losses_count, draws):
                        delta = completed - last_completed[0]
                        last_completed[0] = completed
                        pbar.update(delta)
                        pbar.set_postfix_str(f"vs最佳 {completed}/{total}局 胜{wins}")
                    
                    win_rate_vs_best = self._evaluate_vs_best(
                        num_games=eval_games, 
                        num_workers=eval_workers, 
                        pbar=None,
                        progress_callback=eval_best_progress
                    )
                    
                    if win_rate_vs_best > 0.55:
                        self._update_best_model(iteration)
                        tqdm.write(f"[迭代 {iteration}] ✓ 更新最佳模型 (vs随机:{win_rate_random*100:.0f}%, vs最佳:{win_rate_vs_best*100:.0f}%)")
                    else:
                        tqdm.write(f"[迭代 {iteration}] 保留旧模型 (vs随机:{win_rate_random*100:.0f}%, vs最佳:{win_rate_vs_best*100:.0f}%)")
                
                pbar.set_postfix_str(f"✓ 胜率{win_rate_random*100:.0f}%")
            else:
                # 非评估轮
                pbar.set_postfix_str("✓")
            
            # 保存统计信息
            self._save_stats()
        
        pbar.close()
        print(f"\n{'='*60}")
        print("🎉 训练完成!")
        print(f"{'='*60}")
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
            pred_logits, pred_values = self.network(states_tensor)
            
            # 计算损失
            # 策略损失: 交叉熵
            # policy_loss = -torch.mean(torch.sum(probs_tensor * log_probs, dim=1))
            policy_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(pred_logits, dim=1),
                probs_tensor,
                reduction='batchmean'
            )
            
            # 价值损失: MSE
            value_loss = nn.functional.mse_loss(pred_values, values_tensor)
            
            # 总损失
            total_loss = 3.0 * value_loss + policy_loss  # 3:1权重
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(total_loss.item())
        
        return losses

    
    def _evaluate_vs_random(self, num_games: int = 20, num_workers: int = 1, pbar = None, progress_callback = None) -> float:
        """
        与随机玩家对战评估（使用较少的模拟次数加速）
        
        Args:
            num_games: 评估局数
            num_workers: 并行进程数，>1时启用多进程
            pbar: 外部进度条对象（可选，用于更新postfix）
            progress_callback: 进度回调函数（可选，用于细粒度进度更新）
        
        Returns:
            胜率
        """
        # 多进程模式
        if num_workers > 1:
            network_state = self.network.cpu().state_dict()
            self.network.to(self.device)  # 恢复到原设备
            network_class = type(self.network).__name__
            
            # 进度回调：优先使用外部传入的回调
            def internal_callback(completed, total, wins, losses, draws):
                if progress_callback:
                    progress_callback(completed, total, wins, losses, draws)
                elif pbar:
                    pbar.set_postfix_str(f"vs随机 {completed}/{total}局 胜{wins}")
            
            wins, losses, draws = evaluate_games_parallel(
                network1_state_dict=network_state,
                network2_state_dict=None,  # None表示随机玩家
                network1_class=network_class,
                num_games=num_games,
                num_workers=num_workers,
                simulations=100,
                desc="  vs随机",
                progress_callback=internal_callback
            )
            return wins / num_games
        
        # 单进程模式（无嵌套进度条，直接用 progress_callback）
        if self.hybrid_mode:
            eval_network = self._get_cpu_network()
        else:
            self.network.eval()
            eval_network = self.network

        wins = 0
        draws = 0

        mcts_player = MCTSPlayer(eval_network, simulations=100, temperature=0)
        random_player = RandomPlayer()

        for game_idx in range(num_games):
            board = Board()
            if game_idx % 2 == 0:
                players = [mcts_player, random_player]
                ai_color = 1
            else:
                players = [random_player, mcts_player]
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

            if progress_callback:
                progress_callback(game_idx + 1, num_games, wins, num_games - wins - draws, draws)

        return wins / num_games
    
    def _evaluate_vs_best(self, num_games: int = 20, num_workers: int = 1, pbar = None, progress_callback = None) -> float:
        """
        与当前最佳模型对战评估（AlphaZero风格）
        
        Args:
            num_games: 评估局数（固定20局）
            num_workers: 并行进程数，>1时启用多进程
            pbar: 外部进度条对象（可选，用于更新postfix）
            progress_callback: 进度回调函数（可选，用于细粒度进度更新）
        
        Returns:
            新模型的胜率
        """
        if self.best_network is None:
            return 1.0
        
        # 多进程模式
        if num_workers > 1:
            network_state = self.network.cpu().state_dict()
            best_network_state = self.best_network.cpu().state_dict()
            self.network.to(self.device)  # 恢复到原设备
            self.best_network.to(self.device)
            network1_class = type(self.network).__name__
            network2_class = type(self.best_network).__name__
            
            # 进度回调：优先使用外部传入的回调
            def internal_callback(completed, total, wins, losses, draws):
                if progress_callback:
                    progress_callback(completed, total, wins, losses, draws)
                elif pbar:
                    pbar.set_postfix_str(f"vs最佳 {completed}/{total}局 胜{wins}")
            
            wins, losses, draws = evaluate_games_parallel(
                network1_state_dict=network_state,
                network2_state_dict=best_network_state,
                network1_class=network1_class,
                network2_class=network2_class,
                num_games=num_games,
                num_workers=num_workers,
                simulations=100,
                desc="  vs最佳",
                progress_callback=internal_callback
            )
            
            return (wins + 0.5 * draws) / num_games
        
        # 单进程模式（无嵌套进度条，直接用 progress_callback）
        if self.hybrid_mode:
            eval_network = self._get_cpu_network()
            best_eval_network = self._get_cpu_best_network()
        else:
            self.network.eval()
            self.best_network.eval()
            eval_network = self.network
            best_eval_network = self.best_network

        new_player = MCTSPlayer(eval_network, simulations=100, temperature=0)
        best_player = MCTSPlayer(best_eval_network, simulations=100, temperature=0)

        wins = 0
        losses = 0
        draws = 0
        for game_idx in range(num_games):
            board = Board()
            if game_idx % 2 == 0:
                players = [new_player, best_player]
                p1_color = 1
            else:
                players = [best_player, new_player]
                p1_color = 2

            current = 0
            while not board.is_game_over():
                action = players[current].get_action(board)
                x, y = action // 15, action % 15
                board.move(x, y)
                current = 1 - current

            winner = board.get_winner()
            if winner == p1_color:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1

            if progress_callback:
                progress_callback(game_idx + 1, num_games, wins, losses, draws)

        win_rate = (wins + 0.5 * draws) / num_games
        return win_rate
    
    def _play_evaluation_games(
        self, 
        player1: MCTSPlayer, 
        player2: MCTSPlayer, 
        num_games: int,
        desc: str
    ) -> Tuple[int, int, int]:
        """
        执行评估对局
        
        Returns:
            (player1胜, player1负, 平局)
        """
        wins = 0
        losses = 0
        draws = 0
        
        pbar = tqdm(range(num_games), desc=desc, leave=False, unit="局")
        for game_idx in pbar:
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
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
            
            pbar.set_postfix_str(f"{wins}胜{losses}负")
        pbar.close()
        
        return wins, losses, draws
    
    def _save_checkpoint(self, iteration: int, optimizer: optim.Optimizer = None, scheduler = None) -> None:
        """
        保存检查点（包含完整训练状态）
        
        Args:
            iteration: 当前迭代数
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
        """
        # 保存网络参数
        path = os.path.join(self.model_dir, f'checkpoint_{iteration}.pth')
        
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'network_class': type(self.network).__name__,
            'num_res_blocks': len(self.network.res_blocks),
            'num_channels': self.network.res_blocks[0].conv1.out_channels if hasattr(self.network, 'res_blocks') and len(self.network.res_blocks) > 0 else 64,
            'stats': {
                'iteration': self.stats['iteration'],
                'total_games': self.stats['total_games'],
                'win_rate_vs_random': self.stats['win_rate_vs_random'],
                'best_model_iteration': self.stats['best_model_iteration'],
            }
        }
        
        # 保存优化器状态
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # 保存调度器状态
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, path)
        tqdm.write(f"保存检查点: {path}")
        
        # 保存经验池到data目录
        buffer_path = os.path.join(self.data_dir, 'replay_buffer.pkl')
        self.replay_buffer.save(buffer_path)
    
    def _update_best_model(self, iteration: int) -> None:
        """更新最佳模型"""
        path = os.path.join(self.model_dir, 'best_model.pth')
        self.network.save(path)
        
        # 从文件加载创建新的best_network（比deepcopy更可靠）
        # 根据当前网络的残差块数量判断类型
        num_blocks = len(self.network.res_blocks)
        if num_blocks <= 5:
            self.best_network = PolicyValueNetworkSmall()
        else:
            self.best_network = PolicyValueNetwork()
        
        self.best_network.load_state_dict(self.network.state_dict())
        self.best_network.to(self.device)
        self.best_network.eval()
        
        self.stats['best_model_iteration'] = iteration
        tqdm.write(f"✓ 更新最佳模型 (迭代 {iteration})")
    
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
    
    def load_checkpoint(self, path: str) -> Tuple[int, Optional[dict], Optional[dict]]:
        """
        加载检查点（包含完整训练状态）
        
        Args:
            path: 检查点路径
        
        Returns:
            (起始迭代数, 优化器状态, 调度器状态)
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载网络参数
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.to(self.device)
        print(f"✓ 加载检查点: {path}")
        
        # 恢复训练统计
        if 'stats' in checkpoint:
            saved_stats = checkpoint['stats']
            self.stats['iteration'] = saved_stats.get('iteration', 0)
            self.stats['total_games'] = saved_stats.get('total_games', 0)
            self.stats['win_rate_vs_random'] = saved_stats.get('win_rate_vs_random', 0.0)
            self.stats['best_model_iteration'] = saved_stats.get('best_model_iteration', 0)
            print(f"  恢复训练状态: 迭代 {self.stats['iteration']}, 总对局 {self.stats['total_games']}")
        
        # 加载经验池
        buffer_path = os.path.join(self.data_dir, 'replay_buffer.pkl')
        if self.replay_buffer.load(buffer_path):
            print(f"  经验池: {len(self.replay_buffer)} 条数据")
        
        # 返回用于恢复优化器的状态
        start_iteration = checkpoint.get('iteration', 0)
        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        scheduler_state = checkpoint.get('scheduler_state_dict', None)
        
        return start_iteration, optimizer_state, scheduler_state


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
    parser.add_argument('--data-dir', type=str, default='data', help='数据保存目录')
    parser.add_argument('--small-network', action='store_true', help='使用小型网络')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    parser.add_argument('--resume', type=str, default=None, help='从指定检查点恢复')
    parser.add_argument('--auto-resume', action='store_true', help='自动从最新检查点恢复')
    parser.add_argument('--eval-interval', type=int, default=5, help='评估间隔(每N轮评估一次)')
    parser.add_argument('--workers', type=int, default=1, help='并行自我对弈进程数 (1=不并行)')
    parser.add_argument('--buffer-size', type=int, default=50000, help='经验池最大容量')
    
    args = parser.parse_args()
    
    # 自动查找最新检查点
    resume_path = args.resume
    if args.auto_resume and resume_path is None:
        resume_path = find_latest_checkpoint(args.model_dir)
        if resume_path:
            print(f"🔍 自动发现最新检查点: {resume_path}")
    
    # 创建训练器
    trainer = Trainer(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        use_small_network=args.small_network,
        device=args.device,
        buffer_size=args.buffer_size
    )
    
    # 断点续训参数
    start_iteration = 0
    optimizer_state = None
    scheduler_state = None
    
    # 恢复训练
    if resume_path:
        start_iteration, optimizer_state, scheduler_state = trainer.load_checkpoint(resume_path)
    
    # 开始训练
    trainer.train(
        iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        simulations=args.simulations,
        batch_size=args.batch_size,
        epochs_per_iteration=args.epochs,
        lr=args.lr,
        save_interval=args.eval_interval,
        num_workers=args.workers,
        verbose=True,
        start_iteration=start_iteration,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state
    )


def find_latest_checkpoint(model_dir: str) -> Optional[str]:
    """
    查找最新的检查点文件
    
    Args:
        model_dir: 模型目录
    
    Returns:
        最新检查点的路径，如果没有则返回None
    """
    import glob
    import re
    
    if not os.path.exists(model_dir):
        return None
    
    # 查找所有检查点文件
    pattern = os.path.join(model_dir, 'checkpoint_*.pth')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # 提取迭代数并排序
    def get_iteration(path):
        match = re.search(r'checkpoint_(\d+)\.pth', path)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_iteration, reverse=True)
    return checkpoints[0]


def check_resume_compatibility(checkpoint_path: str, use_small_network: bool) -> Tuple[bool, str]:
    """
    检查检查点与当前配置的兼容性
    
    Args:
        checkpoint_path: 检查点路径
        use_small_network: 是否使用小型网络
    
    Returns:
        (是否兼容, 原因说明)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 检查网络类型
        saved_class = checkpoint.get('network_class', 'PolicyValueNetwork')
        saved_blocks = checkpoint.get('num_res_blocks', 10)
        
        is_saved_small = saved_class == 'PolicyValueNetworkSmall' or saved_blocks <= 5
        
        if is_saved_small != use_small_network:
            if is_saved_small:
                return False, "检查点是小型网络，但当前未指定 --small-network"
            else:
                return False, "检查点是标准网络，但当前指定了 --small-network"
        
        return True, "兼容"
    except Exception as e:
        return False, f"无法读取检查点: {e}"


if __name__ == '__main__':
    main()
