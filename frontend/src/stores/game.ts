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

    try {
      isThinking.value = true
      message.value = 'AI思考中...'
      
      const res = await gameApi.makeMove(gameId.value, x, y)
      
      if (!res.success) {
        message.value = res.message
        return
      }
      
      board.value = res.board
      winRate.value = res.win_rate
      
      // 更新历史
      history.value.push([x, y])
      
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
        
        message.value = '悔棋成功'
      } else {
        message.value = res.message
      }
    } catch (error) {
      message.value = '悔棋失败'
      console.error(error)
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
    
    // 计算属性
    isPlayerTurn,
    playerName,
    aiName,
    winnerName,
    
    // 方法
    newGame,
    makeMove,
    undo,
    loadModelInfo
  }
})
