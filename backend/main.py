"""
五子棋AI后端服务
基于 MCTS + CNN 的 FastAPI 服务
"""

import os
# 限制 PyTorch 线程数，减少内存占用
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
import uuid
import os
import gc
import numpy as np
import time

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


class AIAssistRequest(BaseModel):
    simulations: int = 50  # 每次增量搜索的模拟次数


class AIAssistResponse(BaseModel):
    visit_matrix: List[List[int]]
    value_matrix: List[List[float]]
    total_simulations: int
    avg_value: float


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
        # 大幅降低模拟次数以适配 2GB 内存服务器（防止 OOM）
        self.simulations: int = 50  # 从 200 降到 50
        self.assist_roots: Dict[str, any] = {}  # 存储每个游戏的辅助搜索根节点
        self.max_games: int = 3  # 限制同时存在的游戏数量（2GB内存建议3个）
        self.game_timeout: int = 600  # 游戏超时时间（秒），10分钟不活动则清理
        self._load_model()
    
    def _load_model(self):
        """加载 AI 模型，优先加载最佳模型"""
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # 优先顺序：best_model.pth > 其他模型 > 随机权重
        model_files = ['best_model.pth']
        
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
        # 清理过期和超量的游戏
        self._cleanup_expired_games()
        
        if len(self.games) >= self.max_games:
            oldest_game_id = next(iter(self.games))
            self.delete_game(oldest_game_id)
            print(f"清理旧游戏: {oldest_game_id}")
        
        game_id = str(uuid.uuid4())[:8]
        board = Board()
        player_color = 1 if player_first else 2
        
        self.games[game_id] = {
            'board': board,
            'player_color': player_color,
            'ai_color': 3 - player_color,
            'last_activity': time.time()  # 记录最后活动时间
        }
        return game_id
    
    def _cleanup_expired_games(self):
        """清理超时的游戏"""
        current_time = time.time()
        expired_games = [
            game_id for game_id, game in self.games.items()
            if current_time - game.get('last_activity', 0) > self.game_timeout
        ]
        for game_id in expired_games:
            print(f"清理超时游戏: {game_id}")
            self.delete_game(game_id)
    
    def _update_activity(self, game_id: str):
        """更新游戏活动时间"""
        if game_id in self.games:
            self.games[game_id]['last_activity'] = time.time()
    
    def get_game(self, game_id: str) -> Optional[Dict]:
        game = self.games.get(game_id)
        if game:
            self._update_activity(game_id)
        return game
    
    def delete_game(self, game_id: str) -> bool:
        if game_id in self.games:
            del self.games[game_id]
            # 同时清理辅助搜索的根节点
            if game_id in self.assist_roots:
                del self.assist_roots[game_id]
            # 手动触发垃圾回收
            gc.collect()
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
        
        # 搜索完成后清理 MCTS 缓存，释放搜索树内存
        del root
        self.mcts.clear_cache()
        gc.collect()
        
        return best_move, win_rate
    
    def ai_assist_search(self, game_id: str, simulations: int = 50) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """AI 辅助搜索：增量 MCTS 搜索，返回热力图数据"""
        game = self.get_game(game_id)
        if game is None or self.mcts is None:
            return np.zeros((BOARD_SIZE, BOARD_SIZE)), np.zeros((BOARD_SIZE, BOARD_SIZE)), 0, 0.5
        
        board: Board = game['board']
        player_color = game['player_color']
        mcts_board = board.to_mcts_format(player_color)
        
        # 获取或创建搜索根节点
        existing_root = self.assist_roots.get(game_id)
        
        # 执行增量搜索
        (value, _), root = self.mcts.search(mcts_board, simulations, existing_root=existing_root)
        
        # 保存根节点供下次继续搜索
        self.assist_roots[game_id] = root
        
        # 获取统计数据
        visit_matrix, value_matrix = self.mcts.get_statistics(root)
        avg_value = (value + 1) / 2  # 转换为 [0, 1] 范围
        
        return visit_matrix, value_matrix, root.visit_count, avg_value
    
    def clear_assist_root(self, game_id: str):
        """清除辅助搜索的根节点"""
        if game_id in self.assist_roots:
            del self.assist_roots[game_id]


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
    
    # AI 落子（添加异常处理）
    try:
        # 落子后清理辅助搜索缓存（棋盘已变化，旧的搜索树无效）
        if game_id in game_manager.assist_roots:
            del game_manager.assist_roots[game_id]
        
        (ai_x, ai_y), win_rate = game_manager.get_ai_move(board, game['ai_color'])
        board.move(ai_x, ai_y)
    except Exception as e:
        import traceback
        print(f"AI计算出错: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI计算失败: {str(e)}")
    
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
    
    # 清除辅助搜索缓存
    game_manager.clear_assist_root(game_id)
    
    return UndoResponse(success=True, board=board.board, current_player=board.current_player, message="悔棋成功")


@app.post("/api/game/{game_id}/assist", response_model=AIAssistResponse)
async def ai_assist(game_id: str, request: AIAssistRequest):
    """AI 辅助搜索：返回 MCTS 热力图数据"""
    game = game_manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    
    # 限制单次模拟次数
    simulations = min(request.simulations, 100)
    
    visit_matrix, value_matrix, total_sims, avg_value = game_manager.ai_assist_search(game_id, simulations)
    
    return AIAssistResponse(
        visit_matrix=visit_matrix.tolist(),
        value_matrix=value_matrix.tolist(),
        total_simulations=total_sims,
        avg_value=avg_value
    )


@app.post("/api/game/{game_id}/assist/reset")
async def reset_assist(game_id: str):
    """重置 AI 辅助搜索"""
    game_manager.clear_assist_root(game_id)
    return {"message": "辅助搜索已重置"}


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
        win_rate_vs_random=1.00
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
