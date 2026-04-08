<script setup lang="ts">
import GameView from './views/Game.vue'
</script>

<template>
  <div class="app">
    <header class="header">
      <div class="container">
        <h1>🎮 五子棋AI</h1>
        <p class="subtitle">基于MCTS和残差网络的人机对弈系统</p>
      </div>
    </header>
    
    <main class="main">
      <GameView />
    </main>
    
    <footer class="footer">
      <div class="container">
        <p>Powered by AlphaZero-style AI</p>
        <div class="footer-links">
          <a href="https://github.com/fallingnight131/Gomoku-AI" target="_blank" rel="noopener">
            📦 代码仓库
          </a>
          <span class="divider">|</span>
          <a href="https://fallingnight.com/blog/2026/01/20/%E4%BB%8E%E9%9B%B6%E5%AE%9E%E7%8E%B0-AlphaZero-%E9%A3%8E%E6%A0%BC%E7%9A%84%E4%BA%94%E5%AD%90%E6%A3%8B-AI/" target="_blank" rel="noopener">
            📖 说明文档
          </a>
        </div>
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
  background: linear-gradient(135deg, var(--shadow-transparent), var(--card-bg));
  padding: 30px 0;
  text-align: center;
  width: 100%;
  min-width: 100vw;
  box-shadow: 0 0 10px var(--shadow);
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
  box-shadow: 0 0 10px var(--shadow);
}

.footer-links {
  margin-top: 10px;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 12px;
}

.footer-links a {
  color: var(--text);
  text-decoration: none;
  transition: color 0.2s ease;
}

.footer-links a:hover {
  color: var(--accent);
}

.footer-links .divider {
  color: var(--text-secondary);
  opacity: 0.5;
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
