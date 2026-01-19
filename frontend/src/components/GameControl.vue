<script setup lang="ts">
import { ref, watch } from 'vue'
import { useGameStore } from '@/stores/game'

const gameStore = useGameStore()
const playerFirst = ref(true)

function startGame() {
  gameStore.newGame(playerFirst.value)
}

// 选择先手后自动重新开始游戏
watch(playerFirst, () => {
  if (gameStore.gameId) {
    startGame()
  }
})
</script>

<template>
  <div class="game-control card">
    <h3>游戏控制</h3>
    
    <!-- 开始新游戏 -->
    <div class="control-group">
      <div class="option-row">
        <label>
          <input type="radio" v-model="playerFirst" :value="true" />
          玩家先手（黑方）
        </label>
        <label>
          <input type="radio" v-model="playerFirst" :value="false" />
          AI先手（黑方）
        </label>
      </div>
      
      <button 
        class="btn btn-primary full-width"
        @click="startGame"
        :disabled="gameStore.isThinking"
      >
        {{ gameStore.gameId ? '重新开始' : '开始游戏' }}
      </button>
    </div>
    
    <!-- 游戏信息 -->
    <div v-if="gameStore.gameId" class="game-info">
      <div class="info-row">
        <span class="label">你的颜色:</span>
        <span class="value">
          <span :class="['stone-icon', gameStore.playerColor === 1 ? 'black' : 'white']"></span>
          {{ gameStore.playerName }}
        </span>
      </div>
      
      <div class="info-row">
        <span class="label">当前回合:</span>
        <span class="value">
          {{ gameStore.isPlayerTurn ? '轮到你' : 'AI思考中' }}
        </span>
      </div>
      
      <div class="info-row">
        <span class="label">步数:</span>
        <span class="value">{{ gameStore.history.length }}</span>
      </div>
    </div>
    
    <!-- 操作按钮 -->
    <div v-if="gameStore.gameId" class="actions">
      <button 
        class="btn btn-secondary"
        @click="gameStore.undo"
        :disabled="gameStore.isThinking || gameStore.history.length < 2"
      >
        悔棋
      </button>
      <button 
        class="btn"
        :class="gameStore.aiAssistMode ? 'btn-active' : 'btn-secondary'"
        @click="gameStore.toggleAiAssist"
        :disabled="gameStore.isThinking || !gameStore.isPlayerTurn || gameStore.gameOver"
      >
        {{ gameStore.aiAssistMode ? '关闭义眼' : '电子义眼' }}
      </button>
    </div>
    
    <!-- AI 辅助进度 -->
    <div v-if="gameStore.aiAssistMode" class="assist-progress">
      <div class="progress-bar">
        <div 
          class="progress-fill" 
          :style="{ width: `${(gameStore.totalSimulations / gameStore.maxSimulations) * 100}%` }"
        ></div>
      </div>
      <span class="progress-text">{{ gameStore.totalSimulations }} / {{ gameStore.maxSimulations }}</span>
    </div>
    
    <!-- 消息 -->
    <div v-if="gameStore.message" class="message" :class="{ 'winner': gameStore.gameOver }">
      {{ gameStore.message }}
    </div>
    
    <!-- 游戏结束 -->
    <div v-if="gameStore.gameOver" class="game-over">
      <div class="result" :class="{ 'win': gameStore.winner === gameStore.playerColor }">
        {{ gameStore.winnerName }}
      </div>
      <button class="btn btn-primary" @click="startGame">
        再来一局
      </button>
    </div>
  </div>
</template>

<style scoped>
.game-control {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

h3 {
  margin-bottom: 5px;
  color: var(--text);
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.option-row {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.option-row label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  color: var(--text-secondary);
}

.option-row input[type="radio"] {
  accent-color: var(--accent);
}

.full-width {
  width: 100%;
}

.game-info {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 15px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.info-row:last-child {
  border-bottom: none;
}

.label {
  color: var(--text-secondary);
}

.value {
  color: var(--text);
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
}

.stone-icon {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  display: inline-block;
}

.stone-icon.black {
  background: linear-gradient(135deg, #666, #1a1a1a);
  border: 1px solid #000;
}

.stone-icon.white {
  background: linear-gradient(135deg, #fff, #d0d0d0);
  border: 1px solid #aaa;
}

.actions {
  display: flex;
  gap: 10px;
}

.actions .btn {
  flex: 1;
}

.btn-active {
  background: var(--accent) !important;
  color: white !important;
}

.assist-progress {
  display: flex;
  align-items: center;
  gap: 10px;
}

.progress-bar {
  flex: 1;
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #4ade80, #22d3ee);
  border-radius: 4px;
  transition: width 0.2s ease;
}

.progress-text {
  font-size: 0.8rem;
  color: var(--text-secondary);
  white-space: nowrap;
}

.message {
  padding: 12px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  text-align: center;
  color: var(--text-secondary);
}

.message.winner {
  background: rgba(233, 69, 96, 0.2);
  color: var(--accent);
  font-weight: 600;
}

.game-over {
  text-align: center;
  padding: 20px;
  background: linear-gradient(135deg, var(--primary), var(--card-bg));
  border-radius: 8px;
}

.result {
  font-size: 1.5rem;
  font-weight: bold;
  margin-bottom: 15px;
  color: var(--text);
}

.result.win {
  color: #4ade80;
}
</style>
