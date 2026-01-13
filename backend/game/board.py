"""
五子棋棋盘类
实现棋盘状态管理、落子、悔棋等功能
"""

import numpy as np
from typing import Tuple, Optional, List
from copy import deepcopy
from .rules import check_winner, get_winner_line, is_valid_move, get_legal_moves, is_board_full


class Board:
    """
    五子棋棋盘类
    
    Attributes:
        size: 棋盘大小，默认15x15
        board: 棋盘状态数组，0=空，1=黑，2=白
        current_player: 当前玩家，1=黑，2=白
        history: 落子历史记录
        last_move: 最后一步落子位置
        winner: 获胜者，0=无，1=黑，2=白
        game_over: 游戏是否结束
    """
    
    def __init__(self, size: int = 15):
        """
        初始化棋盘
        
        Args:
            size: 棋盘大小，默认15
        """
        self.size = size
        self.reset()
    
    def reset(self) -> None:
        """重置棋盘到初始状态"""
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1  # 黑先
        self.history: List[Tuple[int, int]] = []
        self.last_move: Optional[Tuple[int, int]] = None
        self.winner = 0
        self.game_over = False
    
    def copy(self) -> 'Board':
        """创建棋盘副本"""
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.history = self.history.copy()
        new_board.last_move = self.last_move
        new_board.winner = self.winner
        new_board.game_over = self.game_over
        return new_board
    
    def move(self, x: int, y: int) -> bool:
        """
        在指定位置落子
        
        Args:
            x: 落子x坐标 (0-14)
            y: 落子y坐标 (0-14)
        
        Returns:
            落子是否成功
        """
        if self.game_over:
            return False
        
        if not is_valid_move(self.board, x, y):
            return False
        
        # 落子
        self.board[x, y] = self.current_player
        self.history.append((x, y))
        self.last_move = (x, y)
        
        # 检查胜负
        self.winner = check_winner(self.board, self.last_move)
        if self.winner != 0:
            self.game_over = True
        elif is_board_full(self.board):
            self.game_over = True  # 平局
        else:
            # 交换玩家
            self.current_player = 3 - self.current_player
        
        return True
    
    def move_by_action(self, action: int) -> bool:
        """
        通过动作编号落子
        
        Args:
            action: 动作编号 (0-224)，action = x * 15 + y
        
        Returns:
            落子是否成功
        """
        x = action // self.size
        y = action % self.size
        return self.move(x, y)
    
    def undo(self) -> bool:
        """
        悔棋（撤销最后一步）
        
        Returns:
            悔棋是否成功
        """
        if not self.history:
            return False
        
        # 撤销最后一步
        x, y = self.history.pop()
        self.board[x, y] = 0
        
        # 重置游戏状态
        self.game_over = False
        self.winner = 0
        self.current_player = 3 - self.current_player
        
        # 更新最后落子位置
        if self.history:
            self.last_move = self.history[-1]
        else:
            self.last_move = None
        
        return True
    
    def get_legal_actions(self) -> List[int]:
        """
        获取所有合法动作编号
        
        Returns:
            合法动作列表
        """
        actions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    actions.append(i * self.size + j)
        return actions
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        获取所有合法落子位置
        
        Returns:
            合法位置列表
        """
        return get_legal_moves(self.board)
    
    def is_game_over(self) -> bool:
        """检查游戏是否结束"""
        return self.game_over
    
    def get_winner(self) -> int:
        """获取获胜者，0=无/平局，1=黑，2=白"""
        return self.winner
    
    def get_winner_line(self) -> Optional[List[Tuple[int, int]]]:
        """获取获胜连线"""
        if self.winner == 0:
            return None
        return get_winner_line(self.board, self.last_move)
    
    def encode_state(self) -> np.ndarray:
        """
        将棋盘状态编码为神经网络输入格式
        
        Returns:
            形状为(3, 15, 15)的numpy数组
            - 通道0: 当前玩家棋子位置
            - 通道1: 对手棋子位置
            - 通道2: 当前玩家标识（全1或全0）
        """
        state = np.zeros((3, self.size, self.size), dtype=np.float32)
        
        # 通道0: 当前玩家棋子
        state[0] = (self.board == self.current_player).astype(np.float32)
        
        # 通道1: 对手棋子
        opponent = 3 - self.current_player
        state[1] = (self.board == opponent).astype(np.float32)
        
        # 通道2: 当前玩家标识
        if self.current_player == 1:
            state[2] = np.ones((self.size, self.size), dtype=np.float32)
        else:
            state[2] = np.zeros((self.size, self.size), dtype=np.float32)
        
        return state
    
    def get_symmetries(self, probs: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        获取棋盘状态的对称变换（用于数据增强）
        
        Args:
            probs: 动作概率分布 (225,)
        
        Returns:
            [(state, probs), ...] 包含8种对称变换
        """
        state = self.encode_state()
        probs_2d = probs.reshape(self.size, self.size)
        
        symmetries = []
        
        for i in range(4):
            # 旋转
            rotated_state = np.array([np.rot90(s, i) for s in state])
            rotated_probs = np.rot90(probs_2d, i)
            symmetries.append((rotated_state, rotated_probs.flatten()))
            
            # 水平翻转后旋转
            flipped_state = np.array([np.fliplr(np.rot90(s, i)) for s in state])
            flipped_probs = np.fliplr(np.rot90(probs_2d, i))
            symmetries.append((flipped_state, flipped_probs.flatten()))
        
        return symmetries
    
    def action_to_coord(self, action: int) -> Tuple[int, int]:
        """将动作编号转换为坐标"""
        return action // self.size, action % self.size
    
    def coord_to_action(self, x: int, y: int) -> int:
        """将坐标转换为动作编号"""
        return x * self.size + y
    
    def __str__(self) -> str:
        """棋盘的字符串表示"""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = []
        
        # 列标
        header = '   ' + ' '.join(f'{i:2d}' for i in range(self.size))
        lines.append(header)
        
        for i in range(self.size):
            row = f'{i:2d} '
            for j in range(self.size):
                symbol = symbols[self.board[i, j]]
                if self.last_move == (i, j):
                    row += f'[{symbol}]'
                else:
                    row += f' {symbol} '
            lines.append(row)
        
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        return f'Board(size={self.size}, current_player={self.current_player}, moves={len(self.history)})'
