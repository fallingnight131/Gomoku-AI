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
.game-layout {
  display: flex;
  gap: 30px;
  justify-content: center;
  flex-wrap: wrap;
}

.board-section {
  flex-shrink: 0;
}

.control-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
  min-width: 280px;
}

@media (max-width: 900px) {
  .game-layout {
    flex-direction: column;
    align-items: center;
  }
  
  .control-section {
    width: 100%;
    max-width: 500px;
  }
}
</style>
