#!/bin/bash
# Gomoku AI 一键部署脚本
# 支持: Ubuntu/Debian, CentOS/RHEL/Alibaba Cloud Linux
# 在服务器上以 root 或 sudo 权限运行

set -e

echo "=========================================="
echo "    Gomoku AI 五子棋部署脚本"
echo "=========================================="

PROJECT_DIR="/var/www/gomoku"

# 检测系统类型
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        OS=$(uname -s)
    fi
    echo "检测到操作系统: $OS"
}

detect_os

# 1. 安装系统依赖
echo "[1/6] 安装系统依赖..."

if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
    # Ubuntu/Debian 系统
    apt update
    apt install -y git nginx python3 python3-venv python3-pip curl
    
    # 安装 Node.js 18
    if ! command -v node &> /dev/null || [[ $(node -v | cut -d'v' -f2 | cut -d'.' -f1) -lt 18 ]]; then
        curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
        apt install -y nodejs
    fi

elif [[ "$OS" == "centos" || "$OS" == "rhel" || "$OS" == "alinux" || "$OS" == "aliyun" || "$OS" == "anolis" ]]; then
    # CentOS/RHEL/Alibaba Cloud Linux 系统
    dnf install -y git nginx python3 python3-pip curl
    
    # 安装 Node.js 18
    if ! command -v node &> /dev/null || [[ $(node -v | cut -d'v' -f2 | cut -d'.' -f1) -lt 18 ]]; then
        curl -fsSL https://rpm.nodesource.com/setup_18.x | bash -
        dnf install -y nodejs
    fi
    
    # 启动并启用 nginx
    systemctl start nginx
    systemctl enable nginx
    
    # 配置防火墙
    if command -v firewall-cmd &> /dev/null; then
        firewall-cmd --permanent --add-service=http
        firewall-cmd --permanent --add-service=https
        firewall-cmd --reload
    fi
else
    echo "不支持的操作系统: $OS"
    exit 1
fi

# 2. 配置后端
echo "[2/6] 配置后端环境..."
cd $PROJECT_DIR/backend

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
# 安装 CPU 版本 PyTorch（轻量服务器通常没有 GPU）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

deactivate

# 3. 构建前端
echo "[3/6] 构建前端..."
cd $PROJECT_DIR/frontend
npm install
npm run build

# 4. 配置 Nginx
echo "[4/6] 配置 Nginx..."

if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
    # Ubuntu/Debian: 使用 sites-available/sites-enabled
    cp $PROJECT_DIR/deploy/nginx.conf /etc/nginx/sites-available/gomoku
    ln -sf /etc/nginx/sites-available/gomoku /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
elif [[ "$OS" == "centos" || "$OS" == "rhel" || "$OS" == "alinux" || "$OS" == "aliyun" || "$OS" == "anolis" ]]; then
    # CentOS/RHEL/Alinux: 使用 conf.d 目录
    cp $PROJECT_DIR/deploy/nginx.conf /etc/nginx/conf.d/gomoku.conf
    # 禁用默认配置
    if [ -f /etc/nginx/nginx.conf ]; then
        sed -i 's/^\([^#]*server {.*\)$/#\1/' /etc/nginx/nginx.conf 2>/dev/null || true
    fi
fi

nginx -t
systemctl restart nginx

# 5. 配置 Systemd 服务
echo "[5/6] 配置后端服务..."
mkdir -p /var/log/gomoku

# 设置权限 (不同系统用户名可能不同)
if id "www-data" &>/dev/null; then
    WEB_USER="www-data"
elif id "nginx" &>/dev/null; then
    WEB_USER="nginx"
else
    WEB_USER="nobody"
fi

# 更新 service 文件中的用户
sed -i "s/User=www-data/User=$WEB_USER/" $PROJECT_DIR/deploy/gomoku.service
sed -i "s/Group=www-data/Group=$WEB_USER/" $PROJECT_DIR/deploy/gomoku.service

chown -R $WEB_USER:$WEB_USER /var/www/gomoku
chown -R $WEB_USER:$WEB_USER /var/log/gomoku

cp $PROJECT_DIR/deploy/gomoku.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable gomoku
systemctl restart gomoku

# 6. 完成
echo "[6/6] 部署完成!"
echo ""
echo "=========================================="
echo "部署状态检查:"
echo "=========================================="
echo ""
echo "后端服务状态:"
systemctl status gomoku --no-pager -l | head -10
echo ""
echo "Nginx 状态:"
systemctl status nginx --no-pager | head -5
echo ""
echo "=========================================="
echo "访问地址: http://$(curl -s ifconfig.me)"
echo "=========================================="
