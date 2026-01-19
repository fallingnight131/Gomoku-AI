/**
 * 游戏API调用模块
 */

import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000
})

// 类型定义
export interface NewGameResponse {
  game_id: string
  board: number[][]
  current_player: number
  player_color: number
  ai_move: [number, number] | null
}

export interface MoveResponse {
  success: boolean
  board: number[][]
  ai_move: [number, number] | null
  win_rate: number
  game_over: boolean
  winner: number
  winner_line: [number, number][] | null
  message: string
}

export interface GameStateResponse {
  board: number[][]
  current_player: number
  player_color: number
  game_over: boolean
  winner: number
  last_move: [number, number] | null
  history: [number, number][]
}

export interface UndoResponse {
  success: boolean
  board: number[][]
  current_player: number
  message: string
}

export interface AIAssistResponse {
  visit_matrix: number[][]
  value_matrix: number[][]
  total_simulations: number
  avg_value: number
}

export interface ModelInfoResponse {
  model_loaded: boolean
  model_path: string
  parameters: number
  training_iteration: number
  win_rate_vs_random: number
}

// API函数
export const createGame = async (playerFirst: boolean = true): Promise<NewGameResponse> => {
  const res = await api.post('/game/new', { player_first: playerFirst })
  return res.data
}

export const makeMove = async (gameId: string, x: number, y: number): Promise<MoveResponse> => {
  const res = await api.post(`/game/${gameId}/move`, { x, y })
  return res.data
}

export const getGameState = async (gameId: string): Promise<GameStateResponse> => {
  const res = await api.get(`/game/${gameId}/state`)
  return res.data
}

export const undoMove = async (gameId: string): Promise<UndoResponse> => {
  const res = await api.post(`/game/${gameId}/undo`)
  return res.data
}

export const aiAssist = async (gameId: string, simulations: number = 50): Promise<AIAssistResponse> => {
  const res = await api.post(`/game/${gameId}/assist`, { simulations })
  return res.data
}

export const resetAssist = async (gameId: string): Promise<void> => {
  await api.post(`/game/${gameId}/assist/reset`)
}

export const deleteGame = async (gameId: string): Promise<void> => {
  await api.delete(`/game/${gameId}`)
}

export const getModelInfo = async (): Promise<ModelInfoResponse> => {
  const res = await api.get('/model/info')
  return res.data
}

export const reloadModel = async (): Promise<void> => {
  await api.post('/model/reload')
}
