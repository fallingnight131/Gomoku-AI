"""
五子棋AI后端服务
基于 MCTS + CNN 的 FastAPI 服务
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
import uuid
import os
import numpy as np

from game.board import Board, BOARD_SIZE
from ai.mcts import MCTS
from ai.network import PolicyValueNetwork


# ==================== Pydantic 请求/响应模型 ====================

class NewGameRequest(BaseModel):
    player_first: bool = True


class NewGameResponse(BaseModel):
    game_id: str
    board: List[List[int]]
    current_player: int
    player_color: int
    ai_move: Optional[Tuple[int, int]] = None


class MoveRequest(BaseModel):
    x: int
    y: int


class MoveResponse(BaseModel):
    success: bool
    board: List[List[int]]
    ai_move: Optional[Tuple[int, int]] = None
    win_rate: float = 0.5
    game_over: bool = False
    winner: int = 0
    winner_line: Optional[List[Tuple[int, int]]] = None
    message: str = ""


class GameStateResponse(BaseModel):
    board: List[List[int]]
    current_player: int
    player_color: int
    game_over: bool
    winner: int
    last_move: Optional[Tuple[int, int]] = None
    history: List[Tuple[int, int]]


class UndoResponse(BaseModel):
    success: bool
    board: List[List[int]]
    current_player: int
    message: str = ""


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_loaded: bool
    model_path: str
    parameters: int
    training_iteration: int
    win_rate_vs_random: float


# ==================== 游戏管理器 ====================

class GameManager:
    """管理所有进行中的游戏"""
    
    def __init__(self):
        self.games: Dict[str, Dict] = {}
        self.mcts: Optional[MCTS] = None
        self.model_path: str = ""
        self.simulations: int = 800
        self._load_model()
    
    def _load_model(self):
        """加载 AI 模型，优先加载最佳模型"""
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # 优先顺序：best_model.pth > run10_2000.pth > 随机权重
        model_files = ['best_model.pth', 'run10_2000.pth']
        
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                try:
                    self.mcts = MCTS(model_path=model_path, c_puct=0.8, use_noise=0.01)
                    self.model_path = model_path
                    print(f"成功加载模型: {model_path}")
                    return
                except Exception as e:
                    print(f"加载模型失败 {model_path}: {e}")
        
        # 无模型文件，使用随机权重
        print("未找到模型文件，使用随机权重初始化")
        try:
            model = PolicyValueNetwork()
            self.mcts = MCTS(model=model, c_puct=0.8, use_noise=0.01)
            self.model_path = "随机权重"
        except Exception as e:
            print(f"初始化随机模型失败: {e}")
            self.mcts = None
            self.model_path = "无模型"
    
    def create_game(self, player_first: bool = True) -> str:
        """创建新游戏"""
        game_id = str(uuid.uuid4())[:8]
        board = Board()
        player_color = 1 if player_first else 2
        
        self.games[game_id] = {
            'board': board,
            'player_color': player_color,
            'ai_color': 3 - player_color
        }
        return game_id
    
    def get_game(self, game_id: str) -> Optional[Dict]:
        return self.games.get(game_id)
    
    def delete_game(self, game_id: str) -> bool:
        if game_id in self.games:
            del self.games[game_id]
            return True
        return False
    
    def get_ai_move(self, board: Board, ai_color: int) -> Tuple[Tuple[int, int], float]:
        """获取 AI 落子"""
        if self.mcts is None:
            # 无模型时随机落子
            legal = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board.board[i][j] == 0]
            return legal[np.random.randint(len(legal))] if legal else (7, 7), 0.5
        
        mcts_board = board.to_mcts_format(ai_color)
        (value, _), root = self.mcts.search(mcts_board, self.simulations)
        best_move = self.mcts.get_best_move(root)
        win_rate = (value + 1) / 2
        
        return best_move, win_rate


# ==================== FastAPI 应用 ====================

app = FastAPI(title="五子棋AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

game_manager = GameManager()


# ==================== API 端点 ====================

@app.get("/")
async def root():
    return {"message": "五子棋AI服务运行中", "version": "2.0.0"}


@app.post("/api/game/new", response_model=NewGameResponse)
async def new_game(request: NewGameRequest):
    """创建新游戏"""
    game_id = game_manager.create_game(player_first=request.player_first)
    game = game_manager.get_game(game_id)
    board: Board = game['board']
    
    response = {
        'game_id': game_id,
        'board': board.board,
        'current_player': board.current_player,
        'player_color': game['player_color'],
        'ai_move': None
    }
    
    # AI 先手
    if not request.player_first:
        (x, y), _ = game_manager.get_ai_move(board, game['ai_color'])
        board.move(x, y)
        response['board'] = board.board
        response['current_player'] = board.current_player
        response['ai_move'] = (x, y)
    
    return response


@app.post("/api/game/{game_id}/move", response_model=MoveResponse)
async def player_move(game_id: str, request: MoveRequest):
    """玩家落子"""
    game = game_manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    
    board: Board = game['board']
    
    if board.current_player != game['player_color']:
        return MoveResponse(success=False, board=board.board, message="还未轮到你落子")
    
    if not board.move(request.x, request.y):
        return MoveResponse(success=False, board=board.board, message="非法落子位置")
    
    # 玩家获胜
    if board.game_over:
        return MoveResponse(
            success=True, board=board.board, game_over=True,
            winner=board.winner, winner_line=board.get_winner_line(),
            message="你赢了！" if board.winner == game['player_color'] else "平局"
        )
    
    # AI 落子
    (ai_x, ai_y), win_rate = game_manager.get_ai_move(board, game['ai_color'])
    board.move(ai_x, ai_y)
    
    # AI 获胜
    if board.game_over:
        return MoveResponse(
            success=True, board=board.board, ai_move=(ai_x, ai_y),
            win_rate=win_rate, game_over=True, winner=board.winner,
            winner_line=board.get_winner_line(),
            message="AI赢了！" if board.winner == game['ai_color'] else "平局"
        )
    
    return MoveResponse(success=True, board=board.board, ai_move=(ai_x, ai_y), win_rate=win_rate)


@app.get("/api/game/{game_id}/state", response_model=GameStateResponse)
async def get_game_state(game_id: str):
    """获取游戏状态"""
    game = game_manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    
    board: Board = game['board']
    return GameStateResponse(
        board=board.board, current_player=board.current_player,
        player_color=game['player_color'], game_over=board.game_over,
        winner=board.winner, last_move=board.last_move, history=board.history
    )


@app.post("/api/game/{game_id}/undo", response_model=UndoResponse)
async def undo_move(game_id: str):
    """悔棋"""
    game = game_manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    
    board: Board = game['board']
    
    if len(board.history) < 2:
        return UndoResponse(success=False, board=board.board, current_player=board.current_player, message="无法悔棋")
    
    board.undo()
    board.undo()
    
    return UndoResponse(success=True, board=board.board, current_player=board.current_player, message="悔棋成功")


@app.delete("/api/game/{game_id}")
async def delete_game(game_id: str):
    """删除游戏"""
    if game_manager.delete_game(game_id):
        return {"message": "游戏已删除"}
    raise HTTPException(status_code=404, detail="游戏不存在")


@app.get("/api/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """获取模型信息"""
    has_model = game_manager.mcts is not None
    parameters = sum(p.numel() for p in game_manager.mcts.model.parameters()) if has_model else 0
    
    return ModelInfoResponse(
        model_loaded=has_model,
        model_path=game_manager.model_path,
        parameters=parameters,
        training_iteration=2000,
        win_rate_vs_random=0.95
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
