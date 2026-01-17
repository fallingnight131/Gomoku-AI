"""
五子棋神经网络模块
实现策略-价值网络 (Policy-Value Network)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BOARD_SIZE = 15


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class PolicyValueNetwork(nn.Module):
    """
    策略-价值网络
    
    输入: (batch, 3, 15, 15) - 3通道棋盘状态
    输出: (value, policy_logits)
        - value: (batch, 1) 当前局面评估值 [-1, 1]
        - policy_logits: (batch, 225) 落子概率分布的logits
    """
    
    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, 
                 num_blocks: int = 5, value_dim: int = 128):
        super().__init__()
        
        # 初始卷积层
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(hidden_channels)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])
        
        # 策略头
        self.policy_conv1 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.policy_bn1 = nn.BatchNorm2d(hidden_channels // 2)
        self.policy_conv2 = nn.Conv2d(hidden_channels // 2, 1, kernel_size=3, padding=1)
        
        # 价值头
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, value_dim)
        self.value_fc2 = nn.Linear(value_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        # 初始卷积
        x = F.relu(self.bn_init(self.conv_init(x)))
        
        # 残差块
        for block in self.res_blocks:
            x = block(x)
        
        # 策略头
        policy = F.relu(self.policy_bn1(self.policy_conv1(x)))
        policy = self.policy_conv2(policy)
        policy = policy.squeeze(1)
        policy_logits = policy.view(x.size(0), -1)
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(x.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return value, policy_logits
    
    def predict(self, x: torch.Tensor) -> tuple:
        """
        推理模式预测
        
        Returns:
            (value, probs): 价值和概率分布
        """
        self.eval()
        with torch.no_grad():
            value, logits = self.forward(x)
            probs = F.softmax(logits, dim=1).view(-1, BOARD_SIZE, BOARD_SIZE)
            return value, probs
    
    def count_parameters(self) -> int:
        """统计模型参数数量"""
        return sum(p.numel() for p in self.parameters())


def board_to_tensor(board: list) -> torch.Tensor:
    """
    将棋盘转换为神经网络输入张量
    
    Args:
        board: 15x15 棋盘，1=当前方，-1=对方，0=空
        
    Returns:
        (3, 15, 15) tensor
    """
    board_arr = np.array(board)
    
    current_player = (board_arr == 1).astype(np.float32)
    opponent = (board_arr == -1).astype(np.float32)
    empty = (board_arr == 0).astype(np.float32)
    
    tensor = np.stack([current_player, opponent, empty], axis=0)
    return torch.FloatTensor(tensor)


def get_network_prediction(model: PolicyValueNetwork, board: list, device: torch.device) -> tuple:
    """
    获取神经网络对棋盘的预测
    
    Args:
        model: 神经网络模型
        board: 棋盘状态
        device: 计算设备
        
    Returns:
        (value, policy): 价值评估和策略概率
    """
    board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        value, policy = model.predict(board_tensor)
    return float(value), policy.squeeze(0).cpu().numpy().tolist()
