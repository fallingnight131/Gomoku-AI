/**
 * 游戏状态管理
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import * as gameApi from '@/api/game'

export interface Move {
  x: number
  y: number
}

export const useGameStore = defineStore('game', () => {
  // 状态
  const gameId = ref<string | null>(null)
  const board = ref<number[][]>(createEmptyBoard())
  const currentPlayer = ref(1) // 1=黑 2=白
  const playerColor = ref(1) // 玩家颜色
  const gameOver = ref(false)
  const winner = ref(0)
  const winnerLine = ref<[number, number][] | null>(null)
  const lastMove = ref<[number, number] | null>(null)
  const history = ref<[number, number][]>([])
  const isThinking = ref(false)
  const winRate = ref(0.5)
  const message = ref('')
  const modelInfo = ref<gameApi.ModelInfoResponse | null>(null)
  
  // AI 辅助模式状态
  const aiAssistMode = ref(false)
  const visitMatrix = ref<number[][] | null>(null)
  const valueMatrix = ref<number[][] | null>(null)
  const totalSimulations = ref(0)
  const maxSimulations = 5000
  let assistInterval: ReturnType<typeof setInterval> | null = null

  // 计算属性
  const isPlayerTurn = computed(() => {
    return !gameOver.value && currentPlayer.value === playerColor.value && !isThinking.value
  })

  const playerName = computed(() => playerColor.value === 1 ? '黑方' : '白方')
  const aiName = computed(() => playerColor.value === 1 ? '白方' : '黑方')

  const winnerName = computed(() => {
    if (winner.value === 0) return '平局'
    if (winner.value === playerColor.value) return '你赢了！'
    return 'AI赢了！'
  })

  // 方法
  function createEmptyBoard(): number[][] {
    return Array(15).fill(null).map(() => Array(15).fill(0))
  }

  async function newGame(playerFirst: boolean = true) {
    try {
      // 停止 AI 辅助模式
      stopAiAssist()
      
      isThinking.value = true
      message.value = ''
      
      const res = await gameApi.createGame(playerFirst)
      
      gameId.value = res.game_id
      board.value = res.board
      currentPlayer.value = res.current_player
      playerColor.value = res.player_color
      gameOver.value = false
      winner.value = 0
      winnerLine.value = null
      history.value = []
      winRate.value = 0.5
      
      if (res.ai_move) {
        lastMove.value = res.ai_move
        history.value.push(res.ai_move)
      } else {
        lastMove.value = null
      }
      
      message.value = playerFirst ? '请落子' : 'AI已落子，请继续'
    } catch (error) {
      message.value = '创建游戏失败'
      console.error(error)
    } finally {
      isThinking.value = false
    }
  }

  async function makeMove(x: number, y: number) {
    if (!gameId.value || !isPlayerTurn.value) return
    if (board.value[x][y] !== 0) return

    // 停止 AI 辅助模式
    stopAiAssist()

    // 先在本地显示玩家落子
    board.value[x][y] = playerColor.value
    lastMove.value = [x, y]
    history.value.push([x, y])
    
    try {
      isThinking.value = true
      message.value = 'AI思考中...'
      
      const res = await gameApi.makeMove(gameId.value, x, y)
      
      if (!res.success) {
        // 落子失败，撤销本地更新
        board.value[x][y] = 0
        lastMove.value = history.value.length > 1 ? history.value[history.value.length - 2] : null
        history.value.pop()
        message.value = res.message
        return
      }
      
      // 从服务器同步棋盘状态
      board.value = res.board
      winRate.value = res.win_rate
      
      if (res.ai_move) {
        lastMove.value = res.ai_move
        history.value.push(res.ai_move)
      }
      
      if (res.game_over) {
        gameOver.value = true
        winner.value = res.winner
        winnerLine.value = res.winner_line
        message.value = res.message || winnerName.value
      } else {
        message.value = '请落子'
        currentPlayer.value = playerColor.value
      }
    } catch (error) {
      // 请求失败，撤销本地更新
      board.value[x][y] = 0
      lastMove.value = history.value.length > 1 ? history.value[history.value.length - 2] : null
      history.value.pop()
      message.value = '落子失败'
      console.error(error)
    } finally {
      isThinking.value = false
    }
  }

  async function undo() {
    if (!gameId.value || history.value.length < 2) return

    try {
      const res = await gameApi.undoMove(gameId.value)
      
      if (res.success) {
        board.value = res.board
        currentPlayer.value = res.current_player
        gameOver.value = false
        winner.value = 0
        winnerLine.value = null
        
        // 移除最后两步
        history.value.pop()
        history.value.pop()
        
        if (history.value.length > 0) {
          lastMove.value = history.value[history.value.length - 1]
        } else {
          lastMove.value = null
        }
        
        // 停止 AI 辅助模式
        stopAiAssist()
        
        message.value = '悔棋成功'
      } else {
        message.value = res.message
      }
    } catch (error) {
      message.value = '悔棋失败'
      console.error(error)
    }
  }

  // AI 辅助模式方法
  async function startAiAssist() {
    if (!gameId.value || !isPlayerTurn.value || gameOver.value) return
    
    // 重置辅助搜索
    await gameApi.resetAssist(gameId.value)
    
    aiAssistMode.value = true
    visitMatrix.value = null
    valueMatrix.value = null
    totalSimulations.value = 0
    message.value = '义眼激活中...'
    
    // 开始定时增量搜索
    assistInterval = setInterval(async () => {
      if (!gameId.value || !aiAssistMode.value) {
        stopAiAssist()
        return
      }
      
      // 达到最大模拟次数后停止
      if (totalSimulations.value >= maxSimulations) {
        message.value = `模拟完${maxSimulations}种可能，等待落子`
        return
      }
      
      try {
        const res = await gameApi.aiAssist(gameId.value, 50)
        visitMatrix.value = res.visit_matrix
        valueMatrix.value = res.value_matrix
        totalSimulations.value = res.total_simulations
        message.value = `义眼已激活`
      } catch (error) {
        console.error('义眼搜索失败', error)
      }
    }, 200) // 每 200ms 更新一次
  }
  
  function stopAiAssist() {
    if (assistInterval) {
      clearInterval(assistInterval)
      assistInterval = null
    }
    aiAssistMode.value = false
    visitMatrix.value = null
    valueMatrix.value = null
    totalSimulations.value = 0
  }
  
  function toggleAiAssist() {
    if (aiAssistMode.value) {
      stopAiAssist()
      message.value = '请落子'
    } else {
      startAiAssist()
    }
  }

  async function loadModelInfo() {
    try {
      modelInfo.value = await gameApi.getModelInfo()
    } catch (error) {
      console.error('获取模型信息失败', error)
    }
  }

  return {
    // 状态
    gameId,
    board,
    currentPlayer,
    playerColor,
    gameOver,
    winner,
    winnerLine,
    lastMove,
    history,
    isThinking,
    winRate,
    message,
    modelInfo,
    
    // AI 辅助模式状态
    aiAssistMode,
    visitMatrix,
    valueMatrix,
    totalSimulations,
    maxSimulations,
    
    // 计算属性
    isPlayerTurn,
    playerName,
    aiName,
    winnerName,
    
    // 方法
    newGame,
    makeMove,
    undo,
    loadModelInfo,
    toggleAiAssist,
    stopAiAssist
  }
})
