"""
五子棋神经网络模块
实现策略-价值网络，用于MCTS引导
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    残差块
    使用两个3x3卷积和跳跃连接
    """
    
    def __init__(self, channels: int):
        """
        Args:
            channels: 输入输出通道数
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # 残差连接
        out = F.relu(out)
        
        return out


class PolicyValueNetwork(nn.Module):
    """
    策略-价值网络
    
    输入: (batch, 3, 15, 15)
        - 通道0: 当前玩家棋子
        - 通道1: 对手棋子
        - 通道2: 当前玩家标识
    
    输出:
        - 策略: (batch, 225) 动作概率分布
        - 价值: (batch, 1) 局面评估值 [-1, 1]
    """
    
    def __init__(self, board_size: int = 15, num_channels: int = 64, num_res_blocks: int = 10):
        """
        Args:
            board_size: 棋盘大小，默认15
            num_channels: 卷积通道数，默认64
            num_res_blocks: 残差块数量，默认10
        """
        super().__init__()
        
        self.board_size = board_size
        self.num_channels = num_channels
        self.action_size = board_size * board_size
        
        # 初始卷积层
        self.conv_input = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # 策略头
        # self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        # self.policy_bn = nn.BatchNorm2d(2)
        # self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)
        self.policy_conv1 = nn.Conv2d(num_channels, num_channels // 4, kernel_size=3, padding=1)
        self.policy_bn1 = nn.BatchNorm2d(num_channels // 4)
        self.policy_conv2 = nn.Conv2d(num_channels // 4, 1, kernel_size=3, padding=1)
        
        # 价值头
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
        # 特殊处理：价值头最后一层用小初始化
        # 避免tanh饱和
        if hasattr(self, 'value_fc2'):
            nn.init.normal_(self.value_fc2.weight, mean=0.0, std=0.01)
            if self.value_fc2.bias is not None:
                nn.init.constant_(self.value_fc2.bias, 0.0)
        
        # 策略头最后一层也用小初始化
        # 避免输出过于极端
        if hasattr(self, 'policy_fc'):
            nn.init.normal_(self.policy_fc.weight, mean=0.0, std=0.01)
            if self.policy_fc.bias is not None:
                nn.init.constant_(self.policy_fc.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, 3, 15, 15)
        
        Returns:
            policy: 策略概率 (batch, 225)
            value: 局面价值 (batch, 1)
        """
        # 初始卷积
        out = self.conv_input(x)
        out = self.bn_input(out)
        out = F.relu(out)
        
        # 残差块
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # 策略头
        # policy = self.policy_conv(out)
        # policy = self.policy_bn(policy)
        # policy = F.relu(policy)
        # policy = policy.view(policy.size(0), -1)
        # policy = self.policy_fc(policy)
        # policy = F.log_softmax(policy, dim=1)  # 输出log概率
        policy = self.policy_conv1(out)     # (batch, 16, 15, 15)
        policy = self.policy_bn1(policy)
        policy = F.relu(policy)
        policy = self.policy_conv2(policy)  # (batch, 1, 15, 15)
        policy = policy.squeeze(1)          # (batch, 15, 15)
        policy = policy.view(policy.size(0), -1)  # (batch, 225)
        
        # 价值头
        value = self.value_conv(out)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)
        
        return policy, value
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        预测单个状态的策略和价值
        
        Args:
            state: 棋盘状态 (3, 15, 15)
        
        Returns:
            probs: 动作概率分布 (225,)
            value: 局面价值标量
        """
        self.eval()
        with torch.no_grad():
            # 添加batch维度，并移动到模型所在设备
            x = torch.FloatTensor(state).unsqueeze(0)
            device = next(self.parameters()).device
            x = x.to(device)
            
            logits, value = self.forward(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            value = value.squeeze().cpu().numpy().item()
        
        return probs, value
    
    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量预测
        
        Args:
            states: 批量棋盘状态 (batch, 3, 15, 15)
        
        Returns:
            probs: 动作概率分布 (batch, 225)
            values: 局面价值 (batch,)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(states)
            device = next(self.parameters()).device
            x = x.to(device)
            
            log_probs, values = self.forward(x)
            probs = torch.exp(log_probs).cpu().numpy()
            values = values.squeeze(-1).cpu().numpy()
        
        return probs, values
    
    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'board_size': self.board_size,
            'num_channels': self.num_channels,
            'num_res_blocks': len(self.res_blocks)
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'PolicyValueNetwork':
        """
        加载模型
        
        Args:
            path: 模型文件路径
            device: 设备 ('cpu' 或 'cuda')
        
        Returns:
            加载的模型实例
        """
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            board_size=checkpoint.get('board_size', 15),
            num_channels=checkpoint.get('num_channels', 64),
            num_res_blocks=checkpoint.get('num_res_blocks', 10)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model
    
    def count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 轻量版网络，用于快速测试
class PolicyValueNetworkSmall(PolicyValueNetwork):
    """
    小型策略-价值网络
    使用5个残差块，适合快速验证
    """
    
    def __init__(self, board_size: int = 15):
        super().__init__(
            board_size=board_size,
            num_channels=32,
            num_res_blocks=5
        )


if __name__ == '__main__':
    # 测试网络
    model = PolicyValueNetwork()
    print(f"参数量: {model.count_parameters():,}")
    
    # 测试前向传播
    batch_size = 8
    x = torch.randn(batch_size, 3, 15, 15)
    policy, value = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"策略输出形状: {policy.shape}")
    print(f"价值输出形状: {value.shape}")
    
    # 测试单个预测
    state = np.random.randn(3, 15, 15).astype(np.float32)
    probs, v = model.predict(state)
    print(f"预测概率和: {probs.sum():.4f}")
    print(f"预测价值: {v:.4f}")
