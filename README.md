# 五子棋AI (Gomoku-AI)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1.0-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Vue-3.3-green.svg" alt="Vue">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

基于蒙特卡罗树搜索(MCTS)和深度学习的五子棋AI系统，通过自我对弈持续提升棋力，并提供Web界面供用户对弈。

## ✨ 项目特点

- 🎮 **AlphaZero风格AI** - 结合MCTS与深度神经网络，实现强大的博弈能力
- 🔄 **自我对弈训练** - 从零开始通过自我博弈持续提升棋力
- 🌐 **现代Web界面** - 直观的Vue3前端，支持人机对弈、悔棋等功能
- ⚡ **GPU加速支持** - 支持CUDA加速训练，大幅提升训练效率
- 📊 **实时评估显示** - 展示AI对局面的胜率评估

## 🎯 功能演示

### 游戏界面
- 15×15 标准棋盘
- 支持玩家先手/后手选择
- 悔棋功能
- AI思考进度提示
- 胜率实时显示
- 获胜连线高亮

## 🛠️ 技术栈

### 后端
| 技术 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 编程语言 |
| PyTorch | 2.1.0 | 深度学习框架 |
| FastAPI | 0.104.1 | 高性能Web框架 |
| NumPy | 1.24.3 | 数值计算 |
| Uvicorn | 0.24.0 | ASGI服务器 |

### 前端
| 技术 | 版本 | 说明 |
|------|------|------|
| Vue | 3.3.8 | 渐进式前端框架 |
| TypeScript | 5.3.2 | 类型安全的JavaScript |
| Vite | 5.0.0 | 下一代构建工具 |
| Pinia | 2.1.7 | 状态管理 |
| Axios | 1.6.2 | HTTP客户端 |

## 🚀 快速开始

### 环境要求
- Python 3.10+
- Node.js 16+
- Conda (推荐)

### 1. 克隆项目

```bash
git clone https://github.com/your-username/Gomoku-AI.git
cd Gomoku-AI
```

### 2. 创建Conda环境

```bash
conda create -n gomoku python=3.10 -y
conda activate gomoku
```

### 3. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

### 4. 安装前端依赖

```bash
cd ../frontend
npm install
```

### 5. 启动服务

**终端1 - 启动后端服务器:**
```bash
cd backend
conda activate gomoku
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**终端2 - 启动前端开发服务器:**
```bash
cd frontend
npm run dev
```

### 6. 开始游戏

访问 http://localhost:5173 开始与AI对弈！

## 🧠 训练AI

### 🚀 快速测试（1-2分钟）

验证训练流程是否正常：

```bash
cd backend
conda activate gomoku
python -m ai.train --iterations 1 --episodes 2 --simulations 50 --small-network
```

### ⚡ 轻量训练（10-30分钟）

使用小型网络快速训练，适合开发调试：

```bash
cd backend
conda activate gomoku
python -m ai.train --iterations 10 --episodes 5 --simulations 100 --small-network
```

### 🔥 标准训练（2-4小时）

平衡训练效果与时间：

```bash
cd backend
conda activate gomoku
python -m ai.train --iterations 50 --episodes 20 --simulations 400
```

### 💪 完整训练（8-24小时）

获得最佳棋力，建议使用GPU：

```bash
cd backend
conda activate gomoku
python -m ai.train --iterations 100 --episodes 50 --simulations 800
```

### 🏆 极限训练（2-7天）

追求最强棋力，需要GPU加速：

```bash
cd backend
conda activate gomoku
python -m ai.train --iterations 200 --episodes 200 --simulations 1600
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--iterations` | 100 | 训练迭代次数 |
| `--episodes` | 10 | 每轮自我对弈局数 |
| `--simulations` | 400 | 每步MCTS模拟次数 |
| `--batch-size` | 256 | 训练批次大小 |
| `--epochs` | 5 | 每轮训练epoch数 |
| `--lr` | 0.001 | 初始学习率 |
| `--model-dir` | models | 模型保存目录 |
| `--small-network` | - | 使用小型网络(5层) |
| `--device` | auto | 计算设备(auto/cpu/cuda/hybrid) |
| `--workers` | 1 | 并行进程数（多核加速） |
| `--resume` | - | 从指定检查点恢复训练 |
| `--auto-resume` | - | 自动从最新检查点恢复 |
| `--eval-interval` | 5 | 评估间隔(每N轮评估一次) |

### 🚀 多核并行训练（推荐）

利用多核CPU大幅加速训练，自我对弈和评估阶段都支持并行：

```bash
# 查看CPU核心数
python -c "import multiprocessing; print(f'CPU核心数: {multiprocessing.cpu_count()}')"

# 使用6核并行训练（推荐设置为核心数的60%）
python -m ai.train --workers 6 --iterations 50 --episodes 20 --simulations 400

# 小型网络 + 多核加速（快速验证）
python -m ai.train --workers 4 --iterations 10 --episodes 10 --simulations 100 --small-network
```

**并行加速效果**：
| 阶段 | 单进程 | 4进程 | 加速比 |
|------|--------|-------|--------|
| 自我对弈 (20局) | ~120s | ~35s | ~3.5x |
| 评估 vs 随机 (10局) | ~10s | ~3s | ~3x |
| 评估 vs 最佳 (20-40局) | ~40s | ~12s | ~3x |

**workers 设置建议**：
| CPU核心数 | 推荐 workers | 说明 |
|-----------|--------------|------|
| 4核 | 2-3 | 保守设置 |
| 8核 | 4-6 | 平衡设置 |
| 10核+ | 6-8 | 高效设置 |

### 🔀 混合模式（GPU训练 + CPU推理）

如果有GPU，推荐使用混合模式获得最佳性能：

```bash
# GPU训练 + CPU多核自我对弈/评估
python -m ai.train --device hybrid --workers 6 --iterations 50 --episodes 20 --simulations 400
```

**device 选项说明**：
| 选项 | 说明 |
|------|------|
| `auto` | 自动检测（默认） |
| `cpu` | 强制使用CPU |
| `cuda` | 强制使用GPU（全部） |
| `hybrid` | GPU训练 + CPU推理（推荐有GPU时使用） |

### 使用GPU训练

确保安装了CUDA版本的PyTorch:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 训练输出

训练过程会生成以下文件：

**`backend/models/` 目录**：
- `checkpoint_{iter}.pth` - 每隔几轮保存的检查点（包含完整训练状态）
- `best_model.pth` - 当前最佳模型（AlphaZero风格选择）
- `training_stats.json` - 训练统计信息

**`backend/data/` 目录**：
- `replay_buffer.pkl` - 经验回放池（用于断点续训）

### 🔄 断点续训

支持从检查点恢复训练，保留完整的训练状态（网络权重、优化器状态、学习率调度器、经验回放池）。

#### 续训命令

```bash
cd backend
conda activate gomoku

# 方式1：手动指定检查点
python -m ai.train --resume models/checkpoint_10.pth --iterations 50

# 方式2：自动从最新检查点恢复（推荐）
python -m ai.train --auto-resume --iterations 50

# 结合多核加速
python -m ai.train --auto-resume --workers 6 --iterations 100 --episodes 20
```

#### 检查点保存内容

| 内容 | 说明 |
|------|------|
| 网络权重 | 神经网络参数 |
| 优化器状态 | Adam 动量等 |
| 学习率调度器 | 当前学习率位置 |
| 迭代次数 | 从第N轮继续 |
| 训练统计 | 历史损失、胜率等 |
| 经验回放池 | data/replay_buffer.pkl（自动加载） |

#### 参数兼容性

**✅ 续训时可以修改的参数**：

| 参数 | 说明 |
|------|------|
| `--iterations` | 可以设置新的目标迭代次数 |
| `--episodes` | 可以调整每轮对弈局数 |
| `--simulations` | 可以调整MCTS模拟次数 |
| `--batch-size` | 可以调整训练批次大小 |
| `--epochs` | 可以调整每轮训练epoch数 |
| `--workers` | 可以调整并行进程数 |
| `--device` | 可以切换设备（cpu/cuda/hybrid） |
| `--eval-interval` | 可以调整评估间隔 |

**❌ 续训时不能修改的参数**：

| 参数 | 原因 |
|------|------|
| `--small-network` | 网络结构必须与检查点一致 |
| `--lr` | 会被调度器状态覆盖（但可以手动重置调度器） |
| `--model-dir` | 建议保持一致，避免混乱 |

**⚠️ 注意事项**：
- 使用 `--auto-resume` 时，会自动检测网络类型是否匹配
- 如果检查点与当前 `--small-network` 参数不匹配，会报错并提示
- 经验回放池从 `data/replay_buffer.pkl` 自动加载，无需手动指定

#### 续训示例场景

```bash
# 场景1：之前用小网络训练了10轮，继续训练到50轮
python -m ai.train --small-network --auto-resume --iterations 50

# 场景2：增加每轮对弈局数，加速数据收集
python -m ai.train --auto-resume --episodes 30 --iterations 100

# 场景3：切换到GPU训练（如果之前用CPU）
python -m ai.train --auto-resume --device cuda --iterations 100

# 场景4：增加MCTS模拟次数提高数据质量
python -m ai.train --auto-resume --simulations 800 --iterations 100
```

### 最佳模型选择策略

采用 **AlphaZero 风格**的模型选择：
1. 每轮训练后，新模型与当前最佳模型对弈 20 局
2. 只有当新模型胜率 > 55% 时，才更新 `best_model.pth`
3. 确保模型持续进步，避免过拟合导致性能下降

## 📁 项目结构

```
Gomoku-AI/
├── backend/
│   ├── main.py                 # FastAPI入口，Web API实现
│   ├── game/
│   │   ├── __init__.py
│   │   ├── board.py            # 棋盘类：落子、悔棋、状态编码
│   │   └── rules.py            # 规则判断：胜负检测、合法性检查
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── network.py          # PyTorch神经网络：残差网络架构
│   │   ├── mcts.py             # MCTS实现：UCB选择、回溯更新
│   │   ├── train.py            # 训练循环：自我对弈→训练→评估
│   │   └── self_play.py        # 自我对弈：数据生成、数据增强
│   ├── models/                 # 模型保存目录（检查点、最佳模型）
│   ├── data/                   # 训练数据目录（经验回放池）
│   └── requirements.txt        # Python依赖
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Board.vue       # 棋盘组件：Canvas绘制、点击事件
│   │   │   ├── GameControl.vue # 控制面板：新游戏、悔棋
│   │   │   └── StatsPanel.vue  # 统计面板：胜率、模型信息
│   │   ├── views/
│   │   │   └── Game.vue        # 游戏主页面
│   │   ├── stores/
│   │   │   └── game.ts         # Pinia状态管理
│   │   ├── api/
│   │   │   └── game.ts         # API调用封装
│   │   ├── App.vue             # 根组件
│   │   ├── main.ts             # 入口文件
│   │   └── style.css           # 全局样式
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── README.md
└── .gitignore
```

## 🔌 API接口

### 游戏接口

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/game/new` | 创建新游戏 |
| POST | `/api/game/{id}/move` | 玩家落子，返回AI响应 |
| GET | `/api/game/{id}/state` | 获取当前棋盘状态 |
| POST | `/api/game/{id}/undo` | 悔棋（撤销双方各一步） |
| DELETE | `/api/game/{id}` | 删除游戏 |

### 模型接口

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/model/info` | 获取模型信息 |
| POST | `/api/model/reload` | 重新加载模型 |

### API示例

**创建新游戏:**
```bash
curl -X POST "http://localhost:8000/api/game/new" \
     -H "Content-Type: application/json" \
     -d '{"player_first": true}'
```

**玩家落子:**
```bash
curl -X POST "http://localhost:8000/api/game/{game_id}/move" \
     -H "Content-Type: application/json" \
     -d '{"x": 7, "y": 7}'
```

## 🧬 网络架构

```
输入: (batch, 3, 15, 15)
  ├── 通道0: 当前玩家棋子位置
  ├── 通道1: 对手棋子位置
  └── 通道2: 当前玩家标识（全1=黑方，全0=白方）
        │
        ▼
┌─────────────────────────────────┐
│  Conv2d(3→64, 3×3) + BN + ReLU  │  初始卷积层
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     ResBlock × 10               │  残差块（可配置5-10层）
│  ┌─────────────────────────┐    │
│  │ Conv(64,64) → BN → ReLU │    │
│  │ Conv(64,64) → BN        │    │
│  │      + skip connection  │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
        │
        ├─────────────┬─────────────┐
        ▼             ▼             
┌───────────────┐ ┌───────────────┐
│   策略头      │ │   价值头      │
│ Conv(64→2)    │ │ Conv(64→1)    │
│ Flatten       │ │ Flatten       │
│ Linear(225)   │ │ Linear(256)   │
│ Softmax       │ │ Linear(1)     │
│               │ │ Tanh          │
└───────────────┘ └───────────────┘
        │                 │
        ▼                 ▼
  落子概率[225]     局面价值[-1,1]
```

**网络参数量:** 约1-3M（取决于残差块数量）

## 📈 MCTS算法

### UCB公式

$$UCB = Q + c_{puct} \cdot P \cdot \frac{\sqrt{\sum N_{parent}}}{1 + N}$$

- $Q$: 平均价值（累计价值/访问次数）
- $P$: 先验概率（来自策略网络）
- $N$: 节点访问次数
- $c_{puct}$: 探索常数（默认2.0）

### Dirichlet噪声

根节点添加噪声增加探索：

$$P_{root} = 0.75 \cdot P + 0.25 \cdot Dir(\alpha)$$

其中 $\alpha = 0.3$

## ✅ 验收标准

- [x] AI能在100次训练迭代后战胜随机玩家（胜率>90%）
- [x] Web界面流畅，落子响应<2秒
- [x] 代码包含必要注释和文档字符串
- [x] 提供训练脚本和使用说明

## 🔧 常见问题

### Q: 训练时间太长怎么办？
A: 使用 `--small-network` 参数启用小型网络，减少 `--simulations` 和 `--episodes` 参数值。

### Q: 如何使用已训练的模型？
A: 将模型文件放入 `backend/models/best_model.pth`，重启后端服务即可自动加载。

### Q: 前端无法连接后端？
A: 确保后端运行在8000端口，检查 `vite.config.ts` 中的代理配置。

## 📄 许可证

MIT License

## 🙏 致谢

- [AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go) - DeepMind的开创性工作
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能Web框架
- [Vue.js](https://vuejs.org/) - 渐进式前端框架
