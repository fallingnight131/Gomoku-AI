"""
五子棋AI后端服务
FastAPI实现的Web API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
import uuid
import os
import json
import torch

from game.board import Board
from ai.network import PolicyValueNetwork, PolicyValueNetworkSmall
from ai.mcts import MCTS, MCTSPlayer


# ==================== Pydantic模型 ====================

class NewGameRequest(BaseModel):
    """创建新游戏请求"""
    player_first: bool = True  # 玩家是否先手


class NewGameResponse(BaseModel):
    """创建新游戏响应"""
    game_id: str
    board: List[List[int]]
    current_player: int
    player_color: int  # 玩家颜色 1=黑 2=白
    ai_move: Optional[Tuple[int, int]] = None  # AI先手时的落子


class MoveRequest(BaseModel):
    """落子请求"""
    x: int
    y: int


class MoveResponse(BaseModel):
    """落子响应"""
    success: bool
    board: List[List[int]]
    ai_move: Optional[Tuple[int, int]] = None
    win_rate: float = 0.5
    game_over: bool = False
    winner: int = 0  # 0=无 1=黑 2=白
    winner_line: Optional[List[Tuple[int, int]]] = None
    message: str = ""


class GameStateResponse(BaseModel):
    """游戏状态响应"""
    board: List[List[int]]
    current_player: int
    player_color: int
    game_over: bool
    winner: int
    last_move: Optional[Tuple[int, int]] = None
    history: List[Tuple[int, int]]


class UndoResponse(BaseModel):
    """悔棋响应"""
    success: bool
    board: List[List[int]]
    current_player: int
    message: str = ""


class ModelInfoResponse(BaseModel):
    """模型信息响应"""
    model_loaded: bool
    model_path: str
    parameters: int
    training_iteration: int
    win_rate_vs_random: float


# ==================== 游戏管理 ====================

class GameManager:
    """游戏管理器，管理所有进行中的游戏"""
    
    def __init__(self):
        self.games: Dict[str, Dict] = {}
        self.network: Optional[PolicyValueNetwork] = None
        self.model_path: str = ""
        self.training_stats: Dict = {}
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载神经网络模型"""
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # 尝试加载最佳模型
        best_model_path = os.path.join(model_dir, 'best_model.pth')
        
        if os.path.exists(best_model_path):
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.network = PolicyValueNetwork.load(best_model_path, device=device)
                self.model_path = best_model_path
                print(f"加载模型: {best_model_path}")
            except Exception as e:
                print(f"加载模型失败: {e}")
                self._create_new_model()
        else:
            print("未找到训练好的模型，创建新模型")
            self._create_new_model()
        
        # 加载训练统计
        stats_path = os.path.join(model_dir, 'training_stats.json')
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    self.training_stats = json.load(f)
            except:
                self.training_stats = {}
    
    def _create_new_model(self):
        """创建新的未训练模型"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.network = PolicyValueNetworkSmall()
        self.network.to(device)
        self.model_path = "未训练模型"
    
    def create_game(self, player_first: bool = True) -> str:
        """创建新游戏"""
        game_id = str(uuid.uuid4())[:8]
        
        board = Board()
        player_color = 1 if player_first else 2  # 先手为黑
        
        self.games[game_id] = {
            'board': board,
            'player_color': player_color,
            'ai_color': 3 - player_color
        }
        
        return game_id
    
    def get_game(self, game_id: str) -> Optional[Dict]:
        """获取游戏"""
        return self.games.get(game_id)
    
    def delete_game(self, game_id: str) -> bool:
        """删除游戏"""
        if game_id in self.games:
            del self.games[game_id]
            return True
        return False
    
    def get_ai_move(self, board: Board) -> Tuple[int, float]:
        """
        获取AI落子
        
        Returns:
            (action, win_rate)
        """
        if self.network is None:
            # 无模型时随机落子
            import numpy as np
            legal = board.get_legal_actions()
            return np.random.choice(legal), 0.5
        
        mcts = MCTS(self.network, simulations=800, c_puct=2.0)
        action = mcts.get_best_action(board, add_noise=False)
        win_rate = (mcts.get_root_q_value() + 1) / 2  # 转换为0-1范围
        
        return action, win_rate


# ==================== FastAPI应用 ====================

app = FastAPI(
    title="五子棋AI API",
    description="基于MCTS和深度学习的五子棋AI服务",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 游戏管理器实例
game_manager = GameManager()


# ==================== API端点 ====================

@app.get("/")
async def root():
    """根路径"""
    return {"message": "五子棋AI服务运行中", "version": "1.0.0"}


@app.post("/api/game/new", response_model=NewGameResponse)
async def new_game(request: NewGameRequest):
    """
    创建新游戏
    
    - player_first: 玩家是否先手（黑方）
    """
    game_id = game_manager.create_game(player_first=request.player_first)
    game = game_manager.get_game(game_id)
    board = game['board']
    
    response = {
        'game_id': game_id,
        'board': board.board.tolist(),
        'current_player': board.current_player,
        'player_color': game['player_color'],
        'ai_move': None
    }
    
    # 如果AI先手，让AI落子
    if not request.player_first:
        action, _ = game_manager.get_ai_move(board)
        x, y = action // 15, action % 15
        board.move(x, y)
        response['board'] = board.board.tolist()
        response['current_player'] = board.current_player
        response['ai_move'] = (x, y)
    
    return response


@app.post("/api/game/{game_id}/move", response_model=MoveResponse)
async def player_move(game_id: str, request: MoveRequest):
    """
    玩家落子
    
    - x: 落子x坐标 (0-14)
    - y: 落子y坐标 (0-14)
    """
    game = game_manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    
    board = game['board']
    
    # 检查是否轮到玩家
    if board.current_player != game['player_color']:
        return MoveResponse(
            success=False,
            board=board.board.tolist(),
            message="还未轮到你落子"
        )
    
    # 玩家落子
    if not board.move(request.x, request.y):
        return MoveResponse(
            success=False,
            board=board.board.tolist(),
            message="非法落子位置"
        )
    
    # 检查游戏是否结束
    if board.is_game_over():
        return MoveResponse(
            success=True,
            board=board.board.tolist(),
            game_over=True,
            winner=board.winner,
            winner_line=board.get_winner_line(),
            message="你赢了！" if board.winner == game['player_color'] else "平局"
        )
    
    # AI落子
    action, win_rate = game_manager.get_ai_move(board)
    ai_x, ai_y = action // 15, action % 15
    board.move(ai_x, ai_y)
    
    # 再次检查游戏是否结束
    if board.is_game_over():
        return MoveResponse(
            success=True,
            board=board.board.tolist(),
            ai_move=(ai_x, ai_y),
            win_rate=win_rate,
            game_over=True,
            winner=board.winner,
            winner_line=board.get_winner_line(),
            message="AI赢了！" if board.winner == game['ai_color'] else "平局"
        )
    
    return MoveResponse(
        success=True,
        board=board.board.tolist(),
        ai_move=(ai_x, ai_y),
        win_rate=win_rate,
        message=""
    )


@app.get("/api/game/{game_id}/state", response_model=GameStateResponse)
async def get_game_state(game_id: str):
    """获取游戏状态"""
    game = game_manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    
    board = game['board']
    
    return GameStateResponse(
        board=board.board.tolist(),
        current_player=board.current_player,
        player_color=game['player_color'],
        game_over=board.game_over,
        winner=board.winner,
        last_move=board.last_move,
        history=board.history
    )


@app.post("/api/game/{game_id}/undo", response_model=UndoResponse)
async def undo_move(game_id: str):
    """
    悔棋
    
    撤销玩家和AI的最后一步
    """
    game = game_manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    
    board = game['board']
    
    if len(board.history) < 2:
        return UndoResponse(
            success=False,
            board=board.board.tolist(),
            current_player=board.current_player,
            message="无法悔棋"
        )
    
    # 撤销两步（AI和玩家各一步）
    board.undo()
    board.undo()
    
    return UndoResponse(
        success=True,
        board=board.board.tolist(),
        current_player=board.current_player,
        message="悔棋成功"
    )


@app.delete("/api/game/{game_id}")
async def delete_game(game_id: str):
    """删除游戏"""
    if game_manager.delete_game(game_id):
        return {"message": "游戏已删除"}
    raise HTTPException(status_code=404, detail="游戏不存在")


@app.get("/api/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """获取模型信息"""
    network = game_manager.network
    stats = game_manager.training_stats
    
    return ModelInfoResponse(
        model_loaded=network is not None,
        model_path=game_manager.model_path,
        parameters=network.count_parameters() if network else 0,
        training_iteration=stats.get('iteration', 0),
        win_rate_vs_random=stats.get('win_rate_vs_random', 0.0)
    )


@app.post("/api/model/reload")
async def reload_model():
    """重新加载模型"""
    game_manager._load_model()
    return {"message": "模型已重新加载", "model_path": game_manager.model_path}


# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
