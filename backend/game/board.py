"""
五子棋棋盘模块
管理游戏状态和规则
"""

from typing import List, Tuple, Optional

BOARD_SIZE = 15


class Board:
    """
    五子棋棋盘类
    
    棋盘表示:
        - 0: 空位
        - 1: 黑棋
        - 2: 白棋
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置棋盘"""
        self.board: List[List[int]] = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.current_player = 1  # 1=黑先
        self.history: List[Tuple[int, int]] = []
        self.last_move: Optional[Tuple[int, int]] = None
        self.game_over = False
        self.winner = 0
    
    def copy(self) -> 'Board':
        """创建副本"""
        new_board = Board()
        new_board.board = [row[:] for row in self.board]
        new_board.current_player = self.current_player
        new_board.history = self.history[:]
        new_board.last_move = self.last_move
        new_board.game_over = self.game_over
        new_board.winner = self.winner
        return new_board
    
    def move(self, x: int, y: int) -> bool:
        """
        落子
        
        Args:
            x, y: 落子坐标
            
        Returns:
            是否成功
        """
        if self.game_over:
            return False
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return False
        if self.board[x][y] != 0:
            return False
        
        self.board[x][y] = self.current_player
        self.history.append((x, y))
        self.last_move = (x, y)
        
        # 检查胜负
        if self._check_win(x, y):
            self.game_over = True
            self.winner = self.current_player
        elif self._is_full():
            self.game_over = True
            self.winner = 0
        else:
            self.current_player = 3 - self.current_player
        
        return True
    
    def undo(self) -> bool:
        """悔棋"""
        if not self.history:
            return False
        
        x, y = self.history.pop()
        self.board[x][y] = 0
        self.game_over = False
        self.winner = 0
        self.current_player = 3 - self.current_player
        
        self.last_move = self.history[-1] if self.history else None
        return True
    
    def _check_win(self, x: int, y: int) -> bool:
        """检查是否获胜"""
        player = self.board[x][y]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            count = 1
            for sign in [1, -1]:
                for i in range(1, 5):
                    nx, ny = x + sign * dx * i, y + sign * dy * i
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx][ny] == player:
                        count += 1
                    else:
                        break
            if count >= 5:
                return True
        return False
    
    def _is_full(self) -> bool:
        """检查棋盘是否已满"""
        return all(self.board[i][j] != 0 for i in range(BOARD_SIZE) for j in range(BOARD_SIZE))
    
    def get_winner_line(self) -> Optional[List[Tuple[int, int]]]:
        """获取获胜连线"""
        if self.winner == 0 or not self.last_move:
            return None
        
        x, y = self.last_move
        player = self.winner
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            line = [(x, y)]
            for sign in [1, -1]:
                for i in range(1, 5):
                    nx, ny = x + sign * dx * i, y + sign * dy * i
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[nx][ny] == player:
                        if sign == 1:
                            line.append((nx, ny))
                        else:
                            line.insert(0, (nx, ny))
                    else:
                        break
            if len(line) >= 5:
                return line[:5]
        return None
    
    def to_mcts_format(self, perspective: int) -> List[List[int]]:
        """
        转换为 MCTS 使用的棋盘格式
        
        MCTS 格式: 1=当前方, -1=对方, 0=空
        
        Args:
            perspective: 视角 (1=黑, 2=白)
        """
        mcts_board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == 0:
                    mcts_board[i][j] = 0
                elif self.board[i][j] == perspective:
                    mcts_board[i][j] = 1
                else:
                    mcts_board[i][j] = -1
        return mcts_board
