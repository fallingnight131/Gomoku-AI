<script setup lang="ts">
import { onMounted } from 'vue'
import { useGameStore } from '@/stores/game'
import Board from '@/components/Board.vue'
import GameControl from '@/components/GameControl.vue'
import StatsPanel from '@/components/StatsPanel.vue'

const gameStore = useGameStore()

onMounted(() => {
  gameStore.loadModelInfo()
})
</script>

<template>
  <div class="game-view container">
    <div class="game-layout">
      <div class="board-section">
        <Board />
      </div>
      
      <div class="control-section">
        <GameControl />
        <StatsPanel />
      </div>
    </div>
  </div>
</template>

<style scoped>
.game-view {
  display: flex;
  justify-content: center;
}

.game-layout {
  display: flex;
  gap: 30px;
  justify-content: center;
  align-items: flex-start;
  flex-wrap: wrap;
}

.board-section {
  flex-shrink: 0;
  display: flex;
  justify-content: center;
}

.control-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
  min-width: 280px;
  max-width: 320px;
}

/* 平板端适配 */
@media (max-width: 900px) {
  .game-layout {
    flex-direction: column;
    align-items: center;
    gap: 20px;
  }
  
  .control-section {
    width: 100%;
    max-width: 500px;
    min-width: auto;
  }
}

/* 手机端适配 */
@media (max-width: 540px) {
  .game-view.container {
    padding: 5px;
  }
  
  .game-layout {
    gap: 10px;
    width: 100%;
    align-items: center;
  }
  
  .board-section {
    width: 100%;
    display: flex;
    justify-content: center;
  }
  
  .control-section {
    max-width: 100%;
    min-width: 0;
    width: calc(100% - 10px);
    padding: 0;
    box-sizing: border-box;
  }
}
</style>
