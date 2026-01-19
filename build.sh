#!/bin/bash
# 五子棋 AI 项目打包脚本

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🔨 开始构建项目..."

# 1. 构建前端
echo "[1/2] 构建前端..."
cd frontend
npm run build
cd ..

# 2. 打包项目
echo "[2/2] 打包项目文件..."
rm -f gomoku-deploy.zip
zip -r gomoku-deploy.zip backend frontend/dist -x "*.pyc" -x "*__pycache__*" -x "*.git*" -x "*node_modules*" -x "backend/logs/*"

echo "✅ 打包完成！"
echo "📦 文件位置: $(pwd)/gomoku-deploy.zip"
echo "📊 文件大小: $(du -h gomoku-deploy.zip | cut -f1)"
echo ""
echo "💡 上传到服务器命令："
echo "   scp gomoku-deploy.zip root@101.200.196.246:/www/wwwroot/"
