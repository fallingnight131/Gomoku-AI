<script setup lang="ts">
import GameView from './views/Game.vue'
</script>

<template>
  <div class="app">
    <header class="header">
      <div class="container">
        <h1>🎮 五子棋AI</h1>
        <p class="subtitle">基于MCTS和深度学习的智能对弈系统</p>
      </div>
    </header>
    
    <main class="main">
      <GameView />
    </main>
    
    <footer class="footer">
      <div class="container">
        <p>Powered by AlphaZero-style AI</p>
      </div>
    </footer>
  </div>
</template>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 100vw;
  overflow-x: hidden;
}

.header {
  background: linear-gradient(135deg, var(--primary), var(--card-bg));
  padding: 30px 0;
  text-align: center;
  width: 100%;
  min-width: 100vw;
}

.header .container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
}

.header h1 {
  font-size: 2.5rem;
  margin-bottom: 10px;
}

.subtitle {
  color: var(--text-secondary);
  font-size: 1rem;
}

.main {
  flex: 1;
  padding: 30px 0;
}

.footer {
  background-color: var(--card-bg);
  padding: 20px 0;
  text-align: center;
  color: var(--text-secondary);
  font-size: 0.9rem;
}

/* 平板端适配 */
@media (max-width: 900px) {
  .header {
    padding: 20px 0;
  }
  
  .header h1 {
    font-size: 2rem;
  }
  
  .main {
    padding: 20px 0;
  }
}

/* 手机端适配 */
@media (max-width: 480px) {
  .header {
    padding: 15px 0;
  }
  
  .header h1 {
    font-size: 1.5rem;
  }
  
  .subtitle {
    font-size: 0.85rem;
  }
  
  .main {
    padding: 10px 0;
  }
  
  .footer {
    padding: 15px 0;
    font-size: 0.8rem;
  }
}
</style>
