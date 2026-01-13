"""
五子棋规则判断模块
实现胜负判定和合法性检查
"""

import numpy as np
from typing import Tuple, Optional, List

# 四个方向：水平、垂直、主对角线、副对角线
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def check_winner(board: np.ndarray, last_move: Tuple[int, int]) -> int:
    """
    检查是否有玩家获胜
    
    Args:
        board: 15x15的棋盘数组，0=空，1=黑，2=白
        last_move: 最后一步落子位置 (x, y)
    
    Returns:
        获胜玩家编号（1或2），无获胜者返回0
    """
    if last_move is None:
        return 0
    
    x, y = last_move
    player = board[x, y]
    
    if player == 0:
        return 0
    
    for dx, dy in DIRECTIONS:
        count = 1  # 包含当前棋子
        
        # 正方向计数
        nx, ny = x + dx, y + dy
        while 0 <= nx < 15 and 0 <= ny < 15 and board[nx, ny] == player:
            count += 1
            nx += dx
            ny += dy
        
        # 反方向计数
        nx, ny = x - dx, y - dy
        while 0 <= nx < 15 and 0 <= ny < 15 and board[nx, ny] == player:
            count += 1
            nx -= dx
            ny -= dy
        
        if count >= 5:
            return player
    
    return 0


def get_winner_line(board: np.ndarray, last_move: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    获取获胜的连线坐标
    
    Args:
        board: 15x15的棋盘数组
        last_move: 最后一步落子位置
    
    Returns:
        获胜连线的坐标列表，无获胜者返回None
    """
    if last_move is None:
        return None
    
    x, y = last_move
    player = board[x, y]
    
    if player == 0:
        return None
    
    for dx, dy in DIRECTIONS:
        line = [(x, y)]
        
        # 正方向
        nx, ny = x + dx, y + dy
        while 0 <= nx < 15 and 0 <= ny < 15 and board[nx, ny] == player:
            line.append((nx, ny))
            nx += dx
            ny += dy
        
        # 反方向
        nx, ny = x - dx, y - dy
        while 0 <= nx < 15 and 0 <= ny < 15 and board[nx, ny] == player:
            line.insert(0, (nx, ny))
            nx -= dx
            ny -= dy
        
        if len(line) >= 5:
            return line
    
    return None


def is_valid_move(board: np.ndarray, x: int, y: int) -> bool:
    """
    检查落子是否合法
    
    Args:
        board: 15x15的棋盘数组
        x: 落子x坐标
        y: 落子y坐标
    
    Returns:
        是否合法
    """
    if not (0 <= x < 15 and 0 <= y < 15):
        return False
    return board[x, y] == 0


def get_legal_moves(board: np.ndarray) -> List[Tuple[int, int]]:
    """
    获取所有合法落子位置
    
    Args:
        board: 15x15的棋盘数组
    
    Returns:
        合法位置列表
    """
    moves = []
    for i in range(15):
        for j in range(15):
            if board[i, j] == 0:
                moves.append((i, j))
    return moves


def is_board_full(board: np.ndarray) -> bool:
    """
    检查棋盘是否已满
    
    Args:
        board: 15x15的棋盘数组
    
    Returns:
        棋盘是否已满
    """
    return np.sum(board == 0) == 0
