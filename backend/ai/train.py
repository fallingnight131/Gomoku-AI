"""
五子棋 AI 模型训练脚本
基于自对弈 + MCTS 的强化学习训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import copy
import os
import json
import multiprocessing
from functools import partial
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from network import PolicyValueNetwork, board_to_tensor, BOARD_SIZE
from mcts import MCTS, Node, check_winner


# ==================== 训练日志 ====================

class TrainLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = 'logs'):
        # 创建带时间戳的训练专属目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(log_dir, f'run_{timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 日志文件路径
        self.log_file = os.path.join(self.run_dir, 'train.log')
        self.json_file = os.path.join(self.run_dir, 'history.json')
        self.curve_file = os.path.join(self.run_dir, 'curves.png')
        
        # 训练历史数据
        self.history = {
            'iterations': [],
            'train_value_loss': [],
            'train_policy_loss': [],
            'val_value_loss': [],
            'val_policy_loss': [],
            'eval_win_rates': [],
            'eval_iterations': [],
            'best_model_updates': []
        }
        
        self._write_log(f"训练日志开始 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log(f"日志目录: {self.run_dir}")
        self._write_log("=" * 60)
    
    def _write_log(self, message: str):
        """写入日志文件"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        print(message)
    
    def log_config(self, config):
        """记录训练配置"""
        self._write_log("\n训练配置:")
        self._write_log(f"  batch_size: {config.batch_size}")
        self._write_log(f"  num_epochs: {config.num_epochs}")
        self._write_log(f"  learning_rate: {config.learning_rate}")
        self._write_log(f"  num_samples: {config.num_samples}")
        self._write_log(f"  train_simulation: {config.train_simulation}")
        self._write_log(f"  eval_interval: {config.eval_interval}")
        self._write_log(f"  eval_games: {config.eval_games}")
        self._write_log(f"  win_threshold: {config.win_threshold}")
        self._write_log("")
    
    def log_iteration_start(self, iteration: int):
        """记录迭代开始"""
        self._write_log(f"\n{'='*50}")
        self._write_log(f"训练迭代 {iteration}")
        self._write_log(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log(f"{'='*50}")
    
    def log_data_generation(self, raw_count: int, augmented_count: int):
        """记录数据生成"""
        self._write_log(f"数据生成: 原始 {raw_count} -> 增强后 {augmented_count}")
    
    def log_epoch(self, iteration: int, epoch: int, num_epochs: int,
                  train_value: float, train_policy: float,
                  val_value: float, val_policy: float):
        """记录单个 epoch 的损失"""
        self._write_log(f"  Epoch {epoch}/{num_epochs}:")
        self._write_log(f"    Train - Value: {train_value:.4f}, Policy: {train_policy:.4f}")
        self._write_log(f"    Val   - Value: {val_value:.4f}, Policy: {val_policy:.4f}")
    
    def log_iteration_end(self, iteration: int,
                          train_value: float, train_policy: float,
                          val_value: float, val_policy: float):
        """记录迭代结束时的损失（最后一个 epoch 的值）"""
        self.history['iterations'].append(iteration)
        self.history['train_value_loss'].append(train_value)
        self.history['train_policy_loss'].append(train_policy)
        self.history['val_value_loss'].append(val_value)
        self.history['val_policy_loss'].append(val_policy)
        
        self._write_log(f"检查点已保存: {iteration}.pth")
        self._save_json()
    
    def log_evaluation(self, iteration: int, win_rate: float, 
                       updated_best: bool, threshold: float):
        """记录评估结果"""
        self.history['eval_iterations'].append(iteration)
        self.history['eval_win_rates'].append(win_rate)
        
        self._write_log(f"\n--- 评估对弈 ---")
        self._write_log(f"当前模型胜率: {win_rate*100:.1f}%")
        
        if updated_best:
            self.history['best_model_updates'].append(iteration)
            self._write_log(f"✓ 胜率超过 {threshold*100:.0f}%，更新最佳模型!")
        else:
            self._write_log(f"✗ 胜率未超过 {threshold*100:.0f}%，保持原最佳模型")
        
        self._save_json()
    
    def log_training_complete(self, best_model_path: str):
        """记录训练完成"""
        self._write_log(f"\n{'='*60}")
        self._write_log(f"训练完成! 最佳模型: {best_model_path}")
        self._write_log(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_log(f"{'='*60}")
        self._save_json()
    
    def _save_json(self):
        """保存 JSON 格式的历史数据"""
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_curves(self, save_path: str = None):
        """绘制训练曲线"""
        if not self.history['iterations']:
            print("没有数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('训练曲线', fontsize=14)
        
        iterations = self.history['iterations']
        
        # 1. Value Loss
        ax1 = axes[0, 0]
        ax1.plot(iterations, self.history['train_value_loss'], 'b-', label='Train', linewidth=1.5)
        ax1.plot(iterations, self.history['val_value_loss'], 'r-', label='Val', linewidth=1.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Value Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Policy Loss
        ax2 = axes[0, 1]
        ax2.plot(iterations, self.history['train_policy_loss'], 'b-', label='Train', linewidth=1.5)
        ax2.plot(iterations, self.history['val_policy_loss'], 'r-', label='Val', linewidth=1.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Policy Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Total Loss
        ax3 = axes[1, 0]
        train_total = [2*v + p for v, p in zip(self.history['train_value_loss'], 
                                                 self.history['train_policy_loss'])]
        val_total = [2*v + p for v, p in zip(self.history['val_value_loss'], 
                                               self.history['val_policy_loss'])]
        ax3.plot(iterations, train_total, 'b-', label='Train', linewidth=1.5)
        ax3.plot(iterations, val_total, 'r-', label='Val', linewidth=1.5)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Loss')
        ax3.set_title('Total Loss (2*Value + Policy)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Win Rate
        ax4 = axes[1, 1]
        if self.history['eval_iterations']:
            ax4.plot(self.history['eval_iterations'], 
                    [r * 100 for r in self.history['eval_win_rates']], 
                    'g-o', linewidth=1.5, markersize=4)
            ax4.axhline(y=55, color='r', linestyle='--', label='Threshold (55%)', alpha=0.7)
            # 标记更新最佳模型的点
            for update_iter in self.history['best_model_updates']:
                if update_iter in self.history['eval_iterations']:
                    idx = self.history['eval_iterations'].index(update_iter)
                    ax4.scatter([update_iter], 
                               [self.history['eval_win_rates'][idx] * 100],
                               color='red', s=100, zorder=5, marker='*')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_title('Win Rate vs Best Model')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.curve_file
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存: {save_path}")
    
    @staticmethod
    def load_and_plot(json_path: str, save_path: str = None):
        """从 JSON 文件加载数据并绘图"""
        with open(json_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        logger = TrainLogger.__new__(TrainLogger)
        logger.history = history
        logger.run_dir = os.path.dirname(json_path)
        logger.curve_file = os.path.join(logger.run_dir, 'curves.png')
        logger.plot_curves(save_path)


# ==================== 训练配置 ====================

class TrainConfig:
    """训练配置 (与 nn_012.py 一致)"""
    batch_size = 256
    num_epochs = 3
    learning_rate = 1e-4
    train_ratio = 0.9
    num_samples = 100           # 每轮自对弈局数
    num_workers = 10            # 并行进程数
    train_simulation = 30       # 训练时 MCTS 模拟次数
    base_path = None            # 基础模型路径
    model_path = 'models/checkpoints'  # 检查点保存目录
    best_model_path = 'models/best_model.pth'  # 最佳模型路径
    log_dir = 'logs'            # 日志目录
    eval_interval = 5           # 评估间隔（每N轮与最佳模型对弈）
    eval_games = 20             # 评估对弈局数
    eval_simulations = 100      # 评估时 MCTS 模拟次数
    win_threshold = 0.55        # 更新最佳模型的胜率阈值


# ==================== 数据集 ====================

class TrainDataset(Dataset):
    """训练数据集"""
    
    def __init__(self, boards, policies, values, weights):
        self.boards = boards
        self.policies = policies
        self.values = values
        self.weights = weights

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx], self.policies[idx], self.values[idx], self.weights[idx]


# ==================== 数据生成 ====================

def generate_random_board(model, device):
    """生成随机起始局面"""
    model = model.to(device)
    perm = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]
    random.shuffle(perm)
    
    best_val = 1e9
    best_board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    
    if random.randint(0, 4) == 0:
        return best_board
    
    num_tries = random.randint(50, random.randint(50, 1000))
    
    for _ in range(num_tries):
        num_moves = random.randint(0, 10)
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        current = -1
        
        for i in range(num_moves):
            x, y = perm[i]
            board[x][y] = current
            current = -current
        
        if check_winner(board) != 0:
            continue
        
        board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
        with torch.no_grad():
            value, _ = model.predict(board_tensor)
        
        val = abs(float(value))
        if val < best_val:
            best_val = val
            best_board = board
    
    return best_board


def calc_next_move(board, probs, temperature=0):
    """根据概率选择下一步"""
    valid_moves = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 0:
                valid_moves.append((i, j, probs[i][j]))
    
    if temperature == 0:
        valid_moves.sort(key=lambda x: x[2], reverse=True)
        return valid_moves[0][:2]
    else:
        moves = [(i, j) for i, j, _ in valid_moves]
        ps = np.array([p for _, _, p in valid_moves], dtype=np.float64)
        if temperature != 1.0:
            ps = ps ** (1.0 / temperature)
        ps = ps / (ps.sum() + 1e-10)
        idx = np.random.choice(len(moves), p=ps)
        return moves[idx]


def generate_single_game(_, model_state_dict, num_simulations):
    """生成单局自对弈数据"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PolicyValueNetwork()
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    board = generate_random_board(model, device)
    temperature = 0.1 * random.randint(0, 9)
    
    mcts = MCTS(model=model, c_puct=0.8, use_noise=0.01)
    
    root = None
    with torch.no_grad():
        for move_num in range(BOARD_SIZE * BOARD_SIZE):
            (value, action_probs), new_root = mcts.search(board, num_simulations, root)
            
            action = calc_next_move(board, action_probs, temperature)
            x, y = action
            board[x][y] = 1
            
            # 复用搜索树
            if (x, y) in new_root.children and new_root.children[(x, y)][0] is not None:
                root = new_root.children[(x, y)][0]
            else:
                root = None
            
            # 检查游戏结束
            if mcts._is_terminal(board):
                break
            
            # 翻转视角
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    board[i][j] *= -1
    
    return mcts.get_train_data()


def augment_data(boards, policies, values, weights):
    """数据增强：D4 对称变换"""
    aug_boards, aug_policies, aug_values, aug_weights = [], [], [], []
    
    for board, policy, value, weight in zip(boards, policies, values, weights):
        value_t = torch.tensor(value).float()
        weight_t = torch.tensor(weight).float()
        
        for k in range(4):
            for flip in [False, True]:
                new_board = torch.rot90(board, k, [1, 2])
                new_policy = torch.rot90(policy, k, [0, 1])
                
                if flip:
                    new_board = torch.flip(new_board, [2])
                    new_policy = torch.flip(new_policy, [1])
                
                aug_boards.append(new_board)
                aug_policies.append(new_policy)
                aug_values.append(value_t)
                aug_weights.append(weight_t)
    
    return (
        torch.stack(aug_boards),
        torch.stack(aug_policies),
        torch.stack(aug_values),
        torch.stack(aug_weights)
    )


def generate_selfplay_data(model, config: TrainConfig):
    """生成自对弈训练数据"""
    model_state_dict = model.state_dict()
    
    with multiprocessing.get_context('spawn').Pool(processes=config.num_workers) as pool:
        func = partial(
            generate_single_game,
            model_state_dict=model_state_dict,
            num_simulations=config.train_simulation
        )
        results = list(tqdm(
            pool.imap(func, range(config.num_samples)),
            total=config.num_samples,
            desc="生成自对弈数据"
        ))
    
    # 整合数据
    boards, policies, values, weights = [], [], [], []
    for game_boards, game_policies, game_values, game_weights in results:
        boards.extend(game_boards)
        policies.extend(game_policies)
        values.extend(game_values)
        weights.extend(game_weights)
    
    boards = torch.stack(boards)
    policies = torch.stack(policies)
    values = torch.FloatTensor(values)
    weights = torch.FloatTensor(weights)
    
    print(f"原始数据量: {len(boards)}")
    
    # 数据增强
    boards, policies, values, weights = augment_data(boards, policies, values, weights)
    
    print(f"增强后数据量: {len(boards)}")
    
    return boards, policies, values, weights


# ==================== 训练 ====================

def train_model(model, train_loader, val_loader, config: TrainConfig, 
                logger: TrainLogger = None, iteration: int = 0):
    """
    训练模型
    
    Returns:
        (avg_train_value, avg_train_policy, avg_val_value, avg_val_policy)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    value_criterion = nn.MSELoss(reduction='none')
    policy_criterion = nn.KLDivLoss(reduction='none')
    val_value_criterion = nn.MSELoss()
    val_policy_criterion = nn.KLDivLoss(reduction='batchmean')
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5
    )
    
    final_losses = (0, 0, 0, 0)
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_value_loss, train_policy_loss = 0, 0
        
        for batch_boards, batch_policies, batch_values, batch_weights in tqdm(
            train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}'
        ):
            batch_boards = batch_boards.to(device)
            batch_policies = batch_policies.to(device).view(batch_policies.size(0), -1)
            batch_values = batch_values.to(device).unsqueeze(1)
            batch_weights = batch_weights.to(device)
            
            optimizer.zero_grad()
            
            pred_values, pred_policies = model(batch_boards)
            
            # 计算加权损失
            value_loss = value_criterion(pred_values, batch_values).squeeze(1)
            policy_loss = policy_criterion(
                F.log_softmax(pred_policies, dim=1),
                batch_policies
            ).sum(dim=1)
            
            weighted_value_loss = (value_loss * batch_weights).mean()
            weighted_policy_loss = (policy_loss * batch_weights).mean()
            
            loss = 2 * weighted_value_loss + weighted_policy_loss
            loss.backward()
            optimizer.step()
            
            train_value_loss += weighted_value_loss.item()
            train_policy_loss += weighted_policy_loss.item()
        
        # 验证阶段
        model.eval()
        val_value_loss, val_policy_loss = 0, 0
        with torch.no_grad():
            for boards, policies, values, weights in val_loader:
                boards = boards.to(device)
                policies = policies.to(device).view(policies.size(0), -1)
                values = values.to(device).unsqueeze(1)
                
                pred_values, pred_policies = model(boards)
                val_value_loss += val_value_criterion(pred_values, values).item()
                val_policy_loss += val_policy_criterion(
                    F.log_softmax(pred_policies, dim=1),
                    policies
                ).item()
        
        # 计算平均损失
        avg_train_value = train_value_loss / len(train_loader)
        avg_train_policy = train_policy_loss / len(train_loader)
        avg_val_value = val_value_loss / len(val_loader)
        avg_val_policy = val_policy_loss / len(val_loader)
        
        final_losses = (avg_train_value, avg_train_policy, avg_val_value, avg_val_policy)
        
        # 记录日志
        if logger:
            logger.log_epoch(iteration, epoch + 1, config.num_epochs,
                           avg_train_value, avg_train_policy,
                           avg_val_value, avg_val_policy)
        else:
            print(f"  Train - Value: {avg_train_value:.4f}, Policy: {avg_train_policy:.4f}")
            print(f"  Val   - Value: {avg_val_value:.4f}, Policy: {avg_val_policy:.4f}")
        
        # 学习率调度
        val_total_loss = 2 * avg_val_value + avg_val_policy
        scheduler.step(val_total_loss)
    
    return final_losses


# ==================== 评估函数 ====================

def calc_next_move_eval(board, probs):
    """评估时选择最佳落子"""
    best_move = None
    best_prob = -1
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 0 and probs[i][j] > best_prob:
                best_prob = probs[i][j]
                best_move = (i, j)
    return best_move


def evaluate_vs_model(model1, model2, device, config: TrainConfig) -> float:
    """
    两个模型对弈评估
    
    Returns:
        model1 的胜率
    """
    model1.eval()
    model2.eval()
    
    mcts1 = MCTS(model=model1, c_puct=0.8, use_noise=0)
    mcts2 = MCTS(model=model2, c_puct=0.8, use_noise=0)
    
    model1_wins = 0
    draws = 0
    
    for game_idx in range(config.eval_games):
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        model1_plays_first = game_idx % 2 == 0
        current_is_model1 = model1_plays_first
        
        for move_num in range(BOARD_SIZE * BOARD_SIZE):
            mcts = mcts1 if current_is_model1 else mcts2
            (_, probs), _ = mcts.search(board, config.eval_simulations)
            action = calc_next_move_eval(board, probs)
            
            if action is None:
                draws += 1
                break
            
            x, y = action
            board[x][y] = 1
            
            result = check_winner(board)
            if result != 0:
                if current_is_model1:
                    model1_wins += 1
                break
            
            # 检查棋盘是否已满
            if all(board[i][j] != 0 for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)):
                draws += 1
                break
            
            # 翻转视角
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    board[i][j] *= -1
            current_is_model1 = not current_is_model1
    
    # 计算胜率（平局算 0.5 胜）
    win_rate = (model1_wins + draws * 0.5) / config.eval_games
    return win_rate


# ==================== 主训练函数 ====================

def train(start_iter=0, end_iter=100, config: TrainConfig = None, logger: TrainLogger = None):
    """主训练函数"""
    if config is None:
        config = TrainConfig()
    
    # 初始化日志记录器
    if logger is None:
        logger = TrainLogger(config.log_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger._write_log(f"使用设备: {device}")
    logger.log_config(config)
    
    # 创建目录
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(os.path.dirname(config.best_model_path), exist_ok=True)
    
    # 初始化模型
    model = PolicyValueNetwork()
    
    if config.base_path and os.path.exists(config.base_path):
        model.load_state_dict(torch.load(config.base_path, map_location=device, weights_only=True))
        logger._write_log(f"加载基础模型: {config.base_path}")
    
    model.to(device)
    
    # 加载或初始化最佳模型
    best_model = PolicyValueNetwork()
    if os.path.exists(config.best_model_path):
        best_model.load_state_dict(torch.load(config.best_model_path, map_location=device, weights_only=True))
        logger._write_log(f"加载最佳模型: {config.best_model_path}")
    else:
        # 首次训练，使用当前模型作为最佳模型
        best_model.load_state_dict(model.state_dict())
        logger._write_log("未找到最佳模型，使用初始权重")
    best_model.to(device)
    
    try:
        for iteration in range(start_iter, end_iter):
            logger.log_iteration_start(iteration + 1)
            
            model.eval()
            
            # 生成自对弈数据
            boards, policies, values, weights = generate_selfplay_data(model, config)
            logger.log_data_generation(len(boards) // 8, len(boards))  # 增强前/后
            
            # 划分训练集和验证集
            num_train = int(len(boards) * config.train_ratio)
            
            train_dataset = TrainDataset(
                boards[:num_train], policies[:num_train],
                values[:num_train], weights[:num_train]
            )
            val_dataset = TrainDataset(
                boards[num_train:], policies[num_train:],
                values[num_train:], weights[num_train:]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            
            # 训练模型
            losses = train_model(model, train_loader, val_loader, config, logger, iteration + 1)
            
            # 保存检查点
            checkpoint_path = os.path.join(config.model_path, f"{iteration + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            
            # 记录迭代结束
            logger.log_iteration_end(iteration + 1, *losses)
            
            # 评估：每 eval_interval 轮与最佳模型对弈
            if (iteration + 1) % config.eval_interval == 0:
                win_rate = evaluate_vs_model(model, best_model, device, config)
                updated_best = win_rate > config.win_threshold
                
                if updated_best:
                    torch.save(model.state_dict(), config.best_model_path)
                    best_model.load_state_dict(model.state_dict())
                
                logger.log_evaluation(iteration + 1, win_rate, updated_best, config.win_threshold)
            
            # 每轮结束后更新曲线图
            logger.plot_curves()
        
        logger.log_training_complete(config.best_model_path)
        
    except KeyboardInterrupt:
        logger._write_log("\n训练被用户中断")
        logger.plot_curves()
    
    return model


def find_latest_checkpoint(checkpoint_dir: str) -> tuple[int, str | None]:
    """
    查找最新的检查点
    
    Returns:
        (最新迭代号, 检查点路径) 或 (0, None) 如果没有检查点
    """
    if not os.path.exists(checkpoint_dir):
        return 0, None
    
    latest_iter = 0
    latest_path = None
    
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth'):
            try:
                # 文件名格式: 123.pth
                iter_num = int(filename[:-4])
                if iter_num > latest_iter:
                    latest_iter = iter_num
                    latest_path = os.path.join(checkpoint_dir, filename)
            except ValueError:
                continue
    
    return latest_iter, latest_path


# ==================== 命令行接口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="五子棋 AI 训练")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # ===== 训练命令 =====
    train_parser = subparsers.add_parser('train', help='开始训练')
    
    # 迭代控制
    train_parser.add_argument("--iterations", "-n", type=int, default=100, help="训练迭代次数")
    
    # 训练参数
    train_parser.add_argument("--samples", type=int, default=100, help="每轮自对弈局数")
    train_parser.add_argument("--simulations", type=int, default=30, help="MCTS 模拟次数")
    train_parser.add_argument("--batch-size", type=int, default=256, help="训练批次大小")
    train_parser.add_argument("--epochs", type=int, default=3, help="每轮训练 epoch 数")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    train_parser.add_argument("--train-ratio", type=float, default=0.9, help="训练集比例")
    
    # 评估参数
    train_parser.add_argument("--eval-interval", type=int, default=10, help="评估间隔（每N轮与最佳模型对弈）")
    train_parser.add_argument("--eval-games", type=int, default=20, help="评估对弈局数")
    train_parser.add_argument("--eval-simulations", type=int, default=100, help="评估时 MCTS 模拟次数")
    train_parser.add_argument("--win-threshold", type=float, default=0.55, help="更新最佳模型的胜率阈值")
    
    # 并行与路径
    train_parser.add_argument("--workers", type=int, default=10, help="并行进程数")
    train_parser.add_argument("--base", type=str, default=None, help="基础模型路径")
    train_parser.add_argument("--save-dir", type=str, default="models/checkpoints", help="检查点保存目录")
    train_parser.add_argument("--best-model", type=str, default="models/best_model.pth", help="最佳模型路径")
    train_parser.add_argument("--log-dir", type=str, default="logs", help="日志保存目录")
    
    # ===== 绘图命令 =====
    plot_parser = subparsers.add_parser('plot', help='从日志绘制训练曲线')
    plot_parser.add_argument("json_file", type=str, help="训练日志的 JSON 文件路径")
    plot_parser.add_argument("--output", "-o", type=str, default=None, help="输出图片路径")
    
    args = parser.parse_args()
    
    # 处理绘图命令
    if args.command == 'plot':
        TrainLogger.load_and_plot(args.json_file, args.output)
    
    # 处理训练命令（或无命令时默认训练）
    elif args.command == 'train' or args.command is None:
        # 如果没有子命令，重新解析为训练参数
        if args.command is None:
            train_parser = argparse.ArgumentParser(description="五子棋 AI 训练")
            train_parser.add_argument("--iterations", "-n", type=int, default=100, help="训练迭代次数")
            train_parser.add_argument("--samples", type=int, default=100, help="每轮自对弈局数")
            train_parser.add_argument("--simulations", type=int, default=30, help="MCTS 模拟次数")
            train_parser.add_argument("--batch-size", type=int, default=256, help="训练批次大小")
            train_parser.add_argument("--epochs", type=int, default=3, help="每轮训练 epoch 数")
            train_parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
            train_parser.add_argument("--train-ratio", type=float, default=0.9, help="训练集比例")
            train_parser.add_argument("--eval-interval", type=int, default=10, help="评估间隔")
            train_parser.add_argument("--eval-games", type=int, default=20, help="评估对弈局数")
            train_parser.add_argument("--eval-simulations", type=int, default=100, help="评估时 MCTS 模拟次数")
            train_parser.add_argument("--win-threshold", type=float, default=0.55, help="更新最佳模型的胜率阈值")
            train_parser.add_argument("--workers", type=int, default=10, help="并行进程数")
            train_parser.add_argument("--base", type=str, default=None, help="基础模型路径")
            train_parser.add_argument("--save-dir", type=str, default="models/checkpoints", help="检查点保存目录")
            train_parser.add_argument("--best-model", type=str, default="models/best_model.pth", help="最佳模型路径")
            train_parser.add_argument("--log-dir", type=str, default="logs", help="日志保存目录")
            args = train_parser.parse_args()
        
        # 创建配置
        config = TrainConfig()
        config.num_samples = args.samples
        config.train_simulation = args.simulations
        config.batch_size = args.batch_size
        config.num_epochs = args.epochs
        config.learning_rate = args.lr
        config.train_ratio = args.train_ratio
        config.eval_interval = args.eval_interval
        config.eval_games = args.eval_games
        config.eval_simulations = args.eval_simulations
        config.win_threshold = args.win_threshold
        config.num_workers = args.workers
        config.base_path = args.base
        config.model_path = args.save_dir
        config.best_model_path = args.best_model
        config.log_dir = args.log_dir
        
        # 自动检测最新检查点
        latest_iter, latest_path = find_latest_checkpoint(config.model_path)
        
        if latest_iter > 0:
            print(f"检测到最新检查点: 迭代 {latest_iter} ({latest_path})")
            # 如果没有指定base，自动使用最新检查点续训
            if config.base_path is None:
                config.base_path = latest_path
                print(f"自动从检查点续训")
        else:
            print("未检测到检查点，从头开始训练")
        
        start_iter = latest_iter
        end_iter = latest_iter + args.iterations
        
        print(f"训练范围: 迭代 {start_iter + 1} 到 {end_iter}")
        
        # 开始训练
        train(start_iter=start_iter, end_iter=end_iter, config=config)
