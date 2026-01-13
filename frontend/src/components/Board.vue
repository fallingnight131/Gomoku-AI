<script setup lang="ts">
import { ref, onMounted, watch, computed } from 'vue'
import { useGameStore } from '@/stores/game'

const gameStore = useGameStore()

// 棋盘配置
const BOARD_SIZE = 15
const CELL_SIZE = 32
const PADDING = 20
const CANVAS_SIZE = CELL_SIZE * (BOARD_SIZE - 1) + PADDING * 2

const canvasRef = ref<HTMLCanvasElement | null>(null)

// 计算棋盘样式
const boardStyle = computed(() => ({
  width: `${CANVAS_SIZE}px`,
  height: `${CANVAS_SIZE}px`
}))

// 绘制棋盘
function drawBoard() {
  const canvas = canvasRef.value
  if (!canvas) return
  
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  
  // 设置canvas大小
  canvas.width = CANVAS_SIZE
  canvas.height = CANVAS_SIZE
  
  // 棋盘背景
  ctx.fillStyle = '#dcb35c'
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
  
  // 绘制网格线
  ctx.strokeStyle = '#8b6914'
  ctx.lineWidth = 1
  
  for (let i = 0; i < BOARD_SIZE; i++) {
    const pos = PADDING + i * CELL_SIZE
    
    // 横线
    ctx.beginPath()
    ctx.moveTo(PADDING, pos)
    ctx.lineTo(CANVAS_SIZE - PADDING, pos)
    ctx.stroke()
    
    // 竖线
    ctx.beginPath()
    ctx.moveTo(pos, PADDING)
    ctx.lineTo(pos, CANVAS_SIZE - PADDING)
    ctx.stroke()
  }
  
  // 绘制星位
  const starPoints = [3, 7, 11]
  ctx.fillStyle = '#8b6914'
  for (const x of starPoints) {
    for (const y of starPoints) {
      const px = PADDING + x * CELL_SIZE
      const py = PADDING + y * CELL_SIZE
      ctx.beginPath()
      ctx.arc(px, py, 4, 0, Math.PI * 2)
      ctx.fill()
    }
  }
  
  // 绘制棋子
  const board = gameStore.board
  for (let i = 0; i < BOARD_SIZE; i++) {
    for (let j = 0; j < BOARD_SIZE; j++) {
      const stone = board[i][j]
      if (stone !== 0) {
        drawStone(ctx, i, j, stone)
      }
    }
  }
  
  // 绘制最后落子标记
  if (gameStore.lastMove) {
    const [x, y] = gameStore.lastMove
    drawLastMoveMarker(ctx, x, y)
  }
  
  // 绘制获胜连线
  if (gameStore.winnerLine) {
    drawWinnerLine(ctx, gameStore.winnerLine)
  }
}

// 绘制棋子
function drawStone(ctx: CanvasRenderingContext2D, x: number, y: number, player: number) {
  const px = PADDING + y * CELL_SIZE
  const py = PADDING + x * CELL_SIZE
  const radius = CELL_SIZE * 0.42
  
  // 渐变效果
  const gradient = ctx.createRadialGradient(
    px - radius * 0.3, py - radius * 0.3, radius * 0.1,
    px, py, radius
  )
  
  if (player === 1) {
    // 黑子
    gradient.addColorStop(0, '#666')
    gradient.addColorStop(1, '#1a1a1a')
  } else {
    // 白子
    gradient.addColorStop(0, '#fff')
    gradient.addColorStop(1, '#d0d0d0')
  }
  
  ctx.fillStyle = gradient
  ctx.beginPath()
  ctx.arc(px, py, radius, 0, Math.PI * 2)
  ctx.fill()
  
  // 边框
  ctx.strokeStyle = player === 1 ? '#000' : '#aaa'
  ctx.lineWidth = 1
  ctx.stroke()
}

// 绘制最后落子标记
function drawLastMoveMarker(ctx: CanvasRenderingContext2D, x: number, y: number) {
  const px = PADDING + y * CELL_SIZE
  const py = PADDING + x * CELL_SIZE
  const stone = gameStore.board[x][y]
  
  ctx.fillStyle = stone === 1 ? '#ff5555' : '#ff0000'
  ctx.beginPath()
  ctx.arc(px, py, 5, 0, Math.PI * 2)
  ctx.fill()
}

// 绘制获胜连线
function drawWinnerLine(ctx: CanvasRenderingContext2D, line: [number, number][]) {
  if (line.length < 2) return
  
  ctx.strokeStyle = '#ff0000'
  ctx.lineWidth = 3
  ctx.lineCap = 'round'
  
  ctx.beginPath()
  const [startX, startY] = line[0]
  ctx.moveTo(PADDING + startY * CELL_SIZE, PADDING + startX * CELL_SIZE)
  
  for (let i = 1; i < line.length; i++) {
    const [x, y] = line[i]
    ctx.lineTo(PADDING + y * CELL_SIZE, PADDING + x * CELL_SIZE)
  }
  ctx.stroke()
}

// 处理点击事件
function handleClick(event: MouseEvent) {
  if (!gameStore.isPlayerTurn) return
  
  const canvas = canvasRef.value
  if (!canvas) return
  
  const rect = canvas.getBoundingClientRect()
  const clickX = event.clientX - rect.left
  const clickY = event.clientY - rect.top
  
  // 计算棋盘坐标
  const j = Math.round((clickX - PADDING) / CELL_SIZE)
  const i = Math.round((clickY - PADDING) / CELL_SIZE)
  
  if (i >= 0 && i < BOARD_SIZE && j >= 0 && j < BOARD_SIZE) {
    gameStore.makeMove(i, j)
  }
}

// 监听棋盘变化重绘
watch(
  () => [gameStore.board, gameStore.lastMove, gameStore.winnerLine],
  () => {
    drawBoard()
  },
  { deep: true }
)

onMounted(() => {
  drawBoard()
})
</script>

<template>
  <div class="board-container card">
    <canvas
      ref="canvasRef"
      :style="boardStyle"
      @click="handleClick"
      :class="{ 'clickable': gameStore.isPlayerTurn }"
    />
    
    <!-- 思考中遮罩 -->
    <div v-if="gameStore.isThinking" class="thinking-overlay">
      <div class="thinking-content">
        <div class="loading"></div>
        <span>AI思考中...</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.board-container {
  position: relative;
  display: inline-block;
  padding: 15px;
}

canvas {
  display: block;
  border-radius: 4px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

canvas.clickable {
  cursor: pointer;
}

.thinking-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
}

.thinking-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  color: white;
  font-size: 1.1rem;
}
</style>
