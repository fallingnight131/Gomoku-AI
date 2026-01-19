<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, computed } from 'vue'
import { useGameStore } from '@/stores/game'

const gameStore = useGameStore()

// 棋盘配置
const BOARD_SIZE = 15
const CELL_SIZE = 32
const PADDING = 20
const CANVAS_SIZE = CELL_SIZE * (BOARD_SIZE - 1) + PADDING * 2

const canvasRef = ref<HTMLCanvasElement | null>(null)
const scale = ref(1)

// 计算缩放比例
function updateScale() {
  const maxWidth = window.innerWidth - 40 // 留出边距
  if (maxWidth < CANVAS_SIZE) {
    scale.value = maxWidth / CANVAS_SIZE
  } else {
    scale.value = 1
  }
}

// 计算棋盘样式
const boardStyle = computed(() => ({
  width: `${CANVAS_SIZE}px`,
  height: `${CANVAS_SIZE}px`
}))

// 计算包裹层样式（用于缩放）
const wrapperStyle = computed(() => {
  const scaledWidth = CANVAS_SIZE * scale.value
  const scaledHeight = CANVAS_SIZE * scale.value
  return {
    transform: `scale(${scale.value})`,
    transformOrigin: 'top left',
    width: `${scaledWidth}px`,
    height: `${scaledHeight}px`
  }
})

// 计算遮罩层样式（需要反向缩放以匹配canvas实际显示尺寸）
const overlayStyle = computed(() => {
  // 遮罩层在缩放的wrapper内，需要使用原始canvas尺寸
  return {
    width: `${CANVAS_SIZE}px`,
    height: `${CANVAS_SIZE}px`
  }
})

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
  
  // 绘制 AI 辅助热力图
  if (gameStore.aiAssistMode && gameStore.visitMatrix && gameStore.valueMatrix) {
    drawHeatmap(ctx, gameStore.visitMatrix, gameStore.valueMatrix)
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

// 绘制热力图
function drawHeatmap(ctx: CanvasRenderingContext2D, visitMatrix: number[][], valueMatrix: number[][]) {
  // 找到最大访问次数
  let maxVisit = 0
  for (let i = 0; i < BOARD_SIZE; i++) {
    for (let j = 0; j < BOARD_SIZE; j++) {
      if (visitMatrix[i][j] > maxVisit) {
        maxVisit = visitMatrix[i][j]
      }
    }
  }
  
  if (maxVisit === 0) return
  
  // 计算加权平均值
  let sumValue = 0
  let sumWeight = 0
  for (let i = 0; i < BOARD_SIZE; i++) {
    for (let j = 0; j < BOARD_SIZE; j++) {
      if (visitMatrix[i][j] > 0) {
        const weight = visitMatrix[i][j] * visitMatrix[i][j]
        sumValue += weight * valueMatrix[i][j]
        sumWeight += weight
      }
    }
  }
  const meanValue = sumWeight > 0 ? sumValue / sumWeight : 0
  
  // 绘制每个位置的热力圆
  for (let i = 0; i < BOARD_SIZE; i++) {
    for (let j = 0; j < BOARD_SIZE; j++) {
      if (visitMatrix[i][j] > 0 && gameStore.board[i][j] === 0) {
        const px = PADDING + j * CELL_SIZE
        const py = PADDING + i * CELL_SIZE
        const radius = CELL_SIZE * 0.4
        
        // 透明度基于访问次数（使用幂函数使差异更明显）
        const alpha = Math.pow(visitMatrix[i][j] / maxVisit, 0.75)
        
        // 颜色基于价值差异：绿色=好棋，红色=差棋
        const valueDiff = Math.max(-1, Math.min(1, (meanValue - valueMatrix[i][j]) * 3))
        const red = Math.round(Math.max(0, Math.min(255, valueDiff * 255)))
        const green = Math.round(Math.max(0, Math.min(255, (1 - valueDiff) * 255)))
        
        ctx.fillStyle = `rgba(${red}, ${green}, 0, ${alpha * 0.8})`
        ctx.beginPath()
        ctx.arc(px, py, radius, 0, Math.PI * 2)
        ctx.fill()
      }
    }
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
  // 考虑缩放比例计算实际点击位置
  const clickX = (event.clientX - rect.left) / scale.value
  const clickY = (event.clientY - rect.top) / scale.value
  
  // 计算棋盘坐标
  const j = Math.round((clickX - PADDING) / CELL_SIZE)
  const i = Math.round((clickY - PADDING) / CELL_SIZE)
  
  if (i >= 0 && i < BOARD_SIZE && j >= 0 && j < BOARD_SIZE) {
    gameStore.makeMove(i, j)
  }
}

// 监听棋盘变化重绘
watch(
  () => [gameStore.board, gameStore.lastMove, gameStore.winnerLine, gameStore.visitMatrix, gameStore.aiAssistMode],
  () => {
    drawBoard()
  },
  { deep: true }
)

onMounted(() => {
  updateScale()
  window.addEventListener('resize', updateScale)
  drawBoard()
})

onUnmounted(() => {
  window.removeEventListener('resize', updateScale)
})
</script>

<template>
  <div class="board-container card">
    <div class="board-wrapper" :style="wrapperStyle">
      <canvas
        ref="canvasRef"
        :style="boardStyle"
        @click="handleClick"
        :class="{ 'clickable': gameStore.isPlayerTurn }"
      />
      
      <!-- 思考中遮罩 -->
      <div v-if="gameStore.isThinking" class="thinking-overlay" :style="overlayStyle">
        <div class="thinking-content">
          <div class="loading"></div>
          <span>AI思考中...</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.board-container {
  position: relative;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 15px;
}

.board-wrapper {
  position: relative;
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
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
}

.thinking-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  color: white;
  font-size: 1.1rem;
}

/* 平板端适配 */
@media (max-width: 900px) {
  .board-container {
    padding: 10px;
  }
}

/* 手机端适配 */
@media (max-width: 540px) {
  .board-container {
    padding: 5px;
  }
  
  .thinking-content {
    font-size: 1rem;
  }
}
</style>
