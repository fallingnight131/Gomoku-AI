<script setup lang="ts">
import { computed } from 'vue'
import { useGameStore } from '@/stores/game'

const gameStore = useGameStore()

// 后端返回的 winRate 是 AI 的胜率评估
const aiWinRatePercent = computed(() => {
  return Math.round(gameStore.winRate * 100)
})

const winRatePercent = computed(() => {
  return 100 - aiWinRatePercent.value
})

const winRateBarStyle = computed(() => ({
  width: `${winRatePercent.value}%`
}))
</script>

<template>
  <div class="stats-panel card">
    <h3>AI评估</h3>
    
    <!-- 胜率条 -->
    <div class="win-rate-section">
      <div class="win-rate-labels">
        <span>玩家 {{ winRatePercent }}%</span>
        <span>AI {{ aiWinRatePercent }}%</span>
      </div>
      <div class="win-rate-bar">
        <div class="player-bar" :style="winRateBarStyle"></div>
      </div>
    </div>
    
    <!-- 模型信息 -->
    <div v-if="gameStore.modelInfo" class="model-info">
      <h4>模型信息</h4>
      <div class="info-grid">
        <div class="info-item">
          <span class="label">状态</span>
          <span class="value" :class="{ 'active': gameStore.modelInfo.model_loaded }">
            {{ gameStore.modelInfo.model_loaded ? '已加载' : '未加载' }}
          </span>
        </div>
        <div class="info-item">
          <span class="label">参数量</span>
          <span class="value">{{ formatNumber(gameStore.modelInfo.parameters) }}</span>
        </div>
        <div class="info-item">
          <span class="label">训练轮数</span>
          <span class="value">{{ gameStore.modelInfo.training_iteration }}</span>
        </div>
        <div class="info-item">
          <span class="label">对随机胜率</span>
          <span class="value">{{ (gameStore.modelInfo.win_rate_vs_random * 100).toFixed(1) }}%</span>
        </div>
      </div>
    </div>
    
    <!-- 使用说明 -->
    <div class="tips">
      <h4>提示</h4>
      <ul>
        <li>点击棋盘落子</li>
        <li>五子连珠获胜</li>
        <li>悔棋将撤销双方各一步</li>
        <li>可以接入电子义眼以获得友方AI视角</li>
      </ul>
    </div>
  </div>
</template>

<script lang="ts">
function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M'
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K'
  }
  return num.toString()
}
</script>

<style scoped>
.stats-panel {
  display: flex;
  flex-direction: column;
  gap: 20px;
  width: 100%;
  box-sizing: border-box;
  overflow: hidden;
}

h3 {
  margin-bottom: 5px;
}

h4 {
  font-size: 0.9rem;
  color: var(--text-secondary);
  margin-bottom: 10px;
}

.win-rate-section {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 15px;
}

.win-rate-labels {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 0.9rem;
}

.win-rate-bar {
  height: 12px;
  background: var(--accent);
  border-radius: 6px;
  overflow: hidden;
}

.player-bar {
  height: 100%;
  background: linear-gradient(90deg, #4ade80, #22c55e);
  border-radius: 6px 0 0 6px;
  transition: width 0.3s ease;
}

.model-info {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 15px;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.info-item .label {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.info-item .value {
  font-weight: 600;
  color: var(--text);
}

.info-item .value.active {
  color: #4ade80;
}

.tips {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 15px;
}

.tips ul {
  list-style: none;
  padding: 0;
}

.tips li {
  padding: 6px 0;
  padding-left: 20px;
  position: relative;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

.tips li::before {
  content: '•';
  position: absolute;
  left: 0;
  color: var(--accent);
}

/* 手机端适配 */
@media (max-width: 480px) {
  .stats-panel {
    padding: 15px;
    gap: 15px;
  }
  
  h3 {
    text-align: center;
  }
  
  .win-rate-labels {
    font-size: 0.85rem;
  }
  
  .model-info, .tips {
    padding: 12px;
  }
  
  h4 {
    font-size: 0.85rem;
    text-align: center;
  }
  
  .info-grid {
    gap: 10px;
  }
  
  .info-item .label {
    font-size: 0.75rem;
  }
  
  .info-item .value {
    font-size: 0.9rem;
  }
  
  .tips li {
    font-size: 0.85rem;
    padding: 4px 0;
    padding-left: 15px;
  }
}
</style>
