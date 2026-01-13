"""
五子棋游戏模块
包含棋盘逻辑和规则判断
"""

from .board import Board
from .rules import check_winner, get_winner_line

__all__ = ['Board', 'check_winner', 'get_winner_line']
