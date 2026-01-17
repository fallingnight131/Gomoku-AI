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
- ⚡ **多进程加速** - 支持多进程并行自对弈，大幅提升训练效率
- 📊 **训练日志与可视化** - 自动记录训练过程，生成损失曲线和胜率图表
- 🔁 **自动续训** - 自动检测检查点，支持断点续训

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
| Matplotlib | - | 训练曲线绘制 |

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

### 基本训练命令

```bash
cd backend
conda activate gomoku
python -m ai.train -n 100  # 训练100轮
```

### 🚀 快速测试（5-10分钟）

验证训练流程是否正常：

```bash
python -m ai.train -n 5 --samples 10 --simulations 30 --workers 4
```

### ⚡ 标准训练（2-4小时）

平衡训练效果与时间：

```bash
python -m ai.train -n 50 --samples 100 --simulations 30 --workers 10
```

### 💪 完整训练（8-24小时）

获得较强棋力：

```bash
python -m ai.train -n 200 --samples 100 --simulations 30 --workers 10
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-n, --iterations` | 100 | 训练迭代次数 |
| `--samples` | 100 | 每轮自对弈局数 |
| `--simulations` | 30 | 训练时MCTS模拟次数 |
| `--batch-size` | 256 | 训练批次大小 |
| `--epochs` | 3 | 每轮训练epoch数 |
| `--lr` | 1e-4 | 学习率 |
| `--train-ratio` | 0.9 | 训练集比例 |
| `--workers` | 10 | 并行进程数 |
| `--eval-interval` | 10 | 评估间隔（每N轮与最佳模型对弈） |
| `--eval-games` | 20 | 评估对弈局数 |
| `--eval-simulations` | 100 | 评估时MCTS模拟次数 |
| `--win-threshold` | 0.55 | 更新最佳模型的胜率阈值 |
| `--base` | None | 指定基础模型路径 |
| `--save-dir` | models/checkpoints | 检查点保存目录 |
| `--best-model` | models/best_model.pth | 最佳模型路径 |
| `--log-dir` | logs | 日志保存目录 |

### 🔄 自动续训

训练脚本会自动检测最新的检查点并继续训练：

```bash
# 首次训练50轮
python -m ai.train -n 50

# 继续训练100轮（自动从第50轮继续）
python -m ai.train -n 100

# 输出示例：
# 检测到最新检查点: 迭代 50 (models/checkpoints/50.pth)
# 自动从检查点续训
# 训练范围: 迭代 51 到 150
```

### 📊 训练日志与可视化

每次训练会在 `logs/` 目录下创建独立的运行目录：

```
backend/logs/
├── run_20260118_120000/    # 第1次训练
│   ├── train.log           # 文本日志
│   ├── history.json        # JSON格式历史数据
│   └── curves.png          # 训练曲线图
│
└── run_20260118_150000/    # 第2次训练
    ├── train.log
    ├── history.json
    └── curves.png
```

**日志内容包括：**
- 训练配置参数
- 每轮迭代的开始时间
- 数据生成量（原始/增强后）
- 每个epoch的loss（value/policy, train/val）
- 模型评估胜率
- 最佳模型更新记录

**从历史日志重新绘图：**

```bash
python -m ai.train plot logs/run_20260118_120000/history.json
```

### 训练输出

训练过程会生成以下文件：

```
backend/
├── models/
│   ├── checkpoints/        # 每轮训练的检查点
│   │   ├── 1.pth
│   │   ├── 2.pth
│   │   └── ...
│   └── best_model.pth      # 当前最佳模型
│
└── logs/
    └── run_YYYYMMDD_HHMMSS/
        ├── train.log       # 训练日志
        ├── history.json    # 历史数据
        └── curves.png      # 训练曲线
```

### 最佳模型选择策略

采用 **AlphaZero 风格**的模型选择：
1. 每隔 `eval_interval` 轮，新模型与当前最佳模型对弈
2. 只有当新模型胜率 > 55% 时，才更新 `best_model.pth`
3. 确保模型持续进步，避免过拟合导致性能下降

## 📁 项目结构

```
Gomoku-AI/
├── backend/
│   ├── main.py                 # FastAPI入口，Web API实现
│   ├── requirements.txt        # Python依赖
│   ├── game/
│   │   ├── __init__.py
│   │   └── board.py            # 棋盘类：落子、悔棋、状态管理
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── network.py          # 神经网络：策略价值网络(残差架构)
│   │   ├── mcts.py             # MCTS实现：UCB选择、训练数据收集
│   │   └── train.py            # 训练脚本：自对弈、训练、评估、日志
│   ├── models/                 # 模型保存目录
│   │   ├── checkpoints/        # 训练检查点
│   │   └── best_model.pth      # 最佳模型
│   └── logs/                   # 训练日志目录
│       └── run_YYYYMMDD_HHMMSS/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Board.vue       # 棋盘组件
│   │   │   ├── GameControl.vue # 控制面板
│   │   │   └── StatsPanel.vue  # 统计面板
│   │   ├── views/
│   │   │   └── Game.vue        # 游戏主页面
│   │   ├── stores/
│   │   │   └── game.ts         # Pinia状态管理
│   │   ├── api/
│   │   │   └── game.ts         # API调用封装
│   │   ├── App.vue
│   │   ├── main.ts
│   │   └── style.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── other/                      # 参考脚本
│   ├── nn_012.py               # 原始训练脚本
│   └── gmk_run_0629.py         # 原始运行脚本
│
└── README.md
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
输入: (batch, 2, 15, 15)
  ├── 通道0: 当前玩家棋子位置 (1)
  └── 通道1: 对手棋子位置 (-1 → 1)
        │
        ▼
┌─────────────────────────────────┐
│  Conv2d(2→32, 3×3) + BN + ReLU  │  初始卷积层
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     ResBlock × 5                │  残差块
│  ┌─────────────────────────┐    │
│  │ Conv(32,32) → BN → ReLU │    │
│  │ Conv(32,32) → BN        │    │
│  │      + skip connection  │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
        │
        ├─────────────────────────┐
        ▼                         ▼
┌───────────────┐         ┌───────────────┐
│   价值头      │         │   策略头      │
│ Conv(32→1)    │         │ Conv(32→1)    │
│ Flatten       │         │ Flatten       │
│ Linear(64)    │         │ Softmax       │
│ Linear(1)     │         │               │
│ Tanh          │         │               │
└───────────────┘         └───────────────┘
        │                         │
        ▼                         ▼
  局面价值[-1,1]            落子概率[225]
```

## 📈 MCTS算法

### UCB公式

$$UCB = Q + c_{puct} \cdot P \cdot \frac{\sqrt{\sum N_{parent}}}{1 + N}$$

- $Q$: 平均价值（累计价值/访问次数）
- $P$: 先验概率（来自策略网络）
- $N$: 节点访问次数
- $c_{puct}$: 探索常数（默认0.8）

### Dirichlet噪声

训练时根节点添加噪声增加探索：

$$P_{root} = (1 - \epsilon) \cdot P + \epsilon \cdot Dir(\alpha)$$

其中 $\epsilon = 0.01$, $\alpha = 0.3$

## 🔧 常见问题

### Q: 训练时间太长怎么办？
A: 减少 `--samples` 和 `--simulations` 参数值，或增加 `--workers` 使用更多进程并行。

### Q: 如何使用已训练的模型？
A: 将模型文件放入 `backend/models/best_model.pth`，重启后端服务即可自动加载。

### Q: 前端无法连接后端？
A: 确保后端运行在8000端口，检查 `vite.config.ts` 中的代理配置。

### Q: 如何查看训练进度？
A: 查看 `logs/run_xxx/` 目录下的 `train.log` 文件和 `curves.png` 图表。

### Q: 如何从指定检查点继续训练？
A: 使用 `--base` 参数指定模型路径：
```bash
python -m ai.train -n 100 --base models/checkpoints/50.pth
```

## 📄 许可证

MIT License

## 🙏 致谢

- [AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go) - DeepMind的开创性工作
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能Web框架
- [Vue.js](https://vuejs.org/) - 渐进式前端框架
