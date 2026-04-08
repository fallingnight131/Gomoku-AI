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
  const winRateHistory = ref<number[]>([])  // 记录每回合后的胜率历史
  const message = ref('')
  const modelInfo = ref<gameApi.ModelInfoResponse | null>(null)
  
  // AI 辅助模式状态
  const aiAssistMode = ref(false)
  const visitMatrix = ref<number[][] | null>(null)
  const valueMatrix = ref<number[][] | null>(null)
  const totalSimulations = ref(0)
  const maxSimulations = 3000
  let assistInterval: ReturnType<typeof setTimeout> | null = null
  let isAssistFetching = false // 防止并发请求

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
      
      // 传入旧的 game_id，让后端释放上一局游戏
      const previousGameId = gameId.value ?? undefined
      const res = await gameApi.createGame(playerFirst, previousGameId)
      
      gameId.value = res.game_id
      board.value = res.board
      currentPlayer.value = res.current_player
      playerColor.value = res.player_color
      gameOver.value = false
      winner.value = 0
      winnerLine.value = null
      history.value = []
      winRate.value = 0.5
      winRateHistory.value = []  // 清空胜率历史
      
      if (res.ai_move) {
        lastMove.value = res.ai_move
        history.value.push(res.ai_move)
      } else {
        lastMove.value = null
      }
      
      message.value = playerFirst ? '请落子' : 'AI已落子，请继续'
    } catch (error: any) {
      if (error.response?.status === 503) {
        message.value = error.response.data.detail || '服务器繁忙，请稍后再试'
      } else {
        message.value = '创建游戏失败'
      }
      console.error(error)
    } finally {
      isThinking.value = false
    }
  }

  async function makeMove(x: number, y: number) {
    if (!gameId.value || !isPlayerTurn.value) return
    if (board.value[x][y] !== 0) return

    // 记录义眼状态，落子后（含AI响应）需要从新位置重启
    const wasAssisting = aiAssistMode.value
    if (wasAssisting) stopAiAssist()

    // 先在本地显示玩家落子
    board.value[x][y] = playerColor.value
    lastMove.value = [x, y]
    history.value.push([x, y])

    let shouldRestart = false

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
        shouldRestart = wasAssisting
        return
      }

      // 从服务器同步棋盘状态
      board.value = res.board
      winRate.value = res.win_rate
      // 记录本回合胜率（一回合 = 玩家落子 + AI落子）
      winRateHistory.value.push(res.win_rate)

      if (res.ai_move) {
        lastMove.value = res.ai_move
        history.value.push(res.ai_move)
      }

      if (res.game_over) {
        gameOver.value = true
        winner.value = res.winner
        winnerLine.value = res.winner_line
        message.value = res.message || winnerName.value
        // 游戏结束，不重启义眼
      } else {
        message.value = '请落子'
        currentPlayer.value = playerColor.value
        shouldRestart = wasAssisting  // 棋局继续，从新棋盘位置重启义眼
      }
    } catch (error: any) {
      // 请求失败，撤销本地更新
      board.value[x][y] = 0
      lastMove.value = history.value.length > 1 ? history.value[history.value.length - 2] : null
      history.value.pop()

      // 检查是否超时
      if (error.response?.status === 408) {
        message.value = '游戏已超时，请重新开始'
        gameOver.value = true
        gameId.value = null
      } else {
        message.value = '落子失败'
        shouldRestart = wasAssisting
      }
      console.error(error)
    } finally {
      isThinking.value = false
      if (shouldRestart) {
        startAiAssist()
      }
    }
  }

  async function undo() {
    if (!gameId.value || history.value.length < 2) return

    const wasAssisting = aiAssistMode.value
    if (wasAssisting) stopAiAssist()

    let shouldRestart = false

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

        // 从历史恢复胜率
        winRateHistory.value.pop()
        if (winRateHistory.value.length > 0) {
          winRate.value = winRateHistory.value[winRateHistory.value.length - 1]
        } else {
          winRate.value = 0.5
        }

        if (history.value.length > 0) {
          lastMove.value = history.value[history.value.length - 1]
        } else {
          lastMove.value = null
        }

        message.value = '悔棋成功'
        shouldRestart = wasAssisting  // 悔棋后从新棋盘位置重启义眼
      } else {
        message.value = res.message
        shouldRestart = wasAssisting
      }
    } catch (error) {
      message.value = '悔棋失败'
      console.error(error)
      shouldRestart = wasAssisting
    } finally {
      if (shouldRestart) {
        startAiAssist()
      }
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
    isAssistFetching = false
    message.value = '义眼激活中...'
    
    // 使用 setTimeout 链式调用，确保上一次请求完成后再发下一次
    async function fetchAssist() {
      if (!gameId.value || !aiAssistMode.value) {
        return
      }
      
      // 达到最大模拟次数后停止
      if (totalSimulations.value >= maxSimulations) {
        message.value = `已模拟完${maxSimulations}种可能，别让时间白白燃烧`
        return
      }
      
      // 防止并发请求
      if (isAssistFetching) {
        assistInterval = setTimeout(fetchAssist, 100)
        return
      }
      
      try {
        isAssistFetching = true
        const res = await gameApi.aiAssist(gameId.value, 50)
        
        // 再次检查是否还在辅助模式
        if (!aiAssistMode.value) return
        
        visitMatrix.value = res.visit_matrix
        valueMatrix.value = res.value_matrix
        totalSimulations.value = res.total_simulations
        
        if (totalSimulations.value >= maxSimulations) {
          message.value = `已模拟完${maxSimulations}种可能，别让时间白白燃烧`
        } else {
          message.value = `义眼已激活`
        }
      } catch (error) {
        console.error('义眼搜索失败', error)
      } finally {
        isAssistFetching = false
      }
      
      // 继续下一次请求
      if (aiAssistMode.value && totalSimulations.value < maxSimulations) {
        assistInterval = setTimeout(fetchAssist, 200)
      }
    }
    
    // 开始第一次请求
    fetchAssist()
  }
  
  function stopAiAssist() {
    if (assistInterval) {
      clearTimeout(assistInterval)
      assistInterval = null
    }
    aiAssistMode.value = false
    isAssistFetching = false
    visitMatrix.value = null
    valueMatrix.value = null
    totalSimulations.value = 0
    if (gameId.value) gameApi.resetAssist(gameId.value)
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

  async function exitGame() {
    // 停止 AI 辅助模式
    stopAiAssist()
    
    // 调用后端删除游戏，释放内存
    if (gameId.value) {
      try {
        await gameApi.deleteGame(gameId.value)
      } catch (error) {
        console.error('退出游戏失败', error)
      }
    }
    
    // 重置本地状态
    gameId.value = null
    board.value = createEmptyBoard()
    currentPlayer.value = 1
    gameOver.value = false
    winner.value = 0
    winnerLine.value = null
    lastMove.value = null
    history.value = []
    winRate.value = 0.5
    message.value = '已退出游戏'
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
    stopAiAssist,
    exitGame
  }
})
