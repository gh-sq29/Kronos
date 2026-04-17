#!/bin/bash
# Kronos Service 部署脚本

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PORT=15321
API_BASE="http://localhost:${PORT}"
SERVICE="kronos-service"

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── 前置检查 ────────────────────────────────────────────────────────────────

check_deps() {
    if ! command -v docker &>/dev/null; then
        log_error "未找到 docker，请先安装 Docker"
        exit 1
    fi
    if ! [ -f "docker-compose.yml" ]; then
        log_error "请在项目根目录（包含 docker-compose.yml）中运行此脚本"
        exit 1
    fi
    if docker compose version &>/dev/null 2>&1; then
        COMPOSE="docker compose"
    elif command -v docker-compose &>/dev/null; then
        COMPOSE="docker-compose"
    else
        log_error "未找到 docker compose 或 docker-compose"
        exit 1
    fi
}

# ── 健康检查 ─────────────────────────────────────────────────────────────────

wait_healthy() {
    log_info "等待服务就绪（最多 60 秒）..."
    local elapsed=0
    until curl -sf "${API_BASE}/health" &>/dev/null; do
        if [ $elapsed -ge 60 ]; then
            log_error "服务未能在 60 秒内启动，查看最近日志："
            $COMPOSE logs --tail=40
            exit 1
        fi
        sleep 3
        elapsed=$((elapsed + 3))
        echo -n "."
    done
    echo ""
    log_info "服务已就绪"
}

print_info() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Kronos Service 部署成功${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "  前端界面:    ${API_BASE}"
    echo -e "  健康检查:    ${API_BASE}/health"
    echo ""
    echo -e "  更新代码:    git pull && ./deploy.sh restart"
    echo -e "  查看日志:    ./deploy.sh logs"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

# ── 子命令 ───────────────────────────────────────────────────────────────────

cmd_up() {
    log_info "首次部署（构建镜像 + 启动）..."
    check_deps
    $COMPOSE up -d --build
    wait_healthy
    print_info
}

cmd_update() {
    log_info "拉取最新代码并重启容器..."
    check_deps
    git pull
    $COMPOSE restart "$SERVICE"
    wait_healthy
    log_info "更新完成"
}

cmd_rebuild() {
    log_info "重新构建镜像（依赖变更时使用）..."
    check_deps
    $COMPOSE down
    $COMPOSE up -d --build
    wait_healthy
    print_info
}

cmd_down() {
    check_deps
    log_info "停止并移除容器..."
    $COMPOSE down
    log_info "容器已停止"
}

cmd_restart() {
    check_deps
    log_info "重启容器..."
    $COMPOSE restart
    wait_healthy
    log_info "重启完成"
}

cmd_clean_restart() {
    check_deps
    log_info "清理 Docker 日志并重启容器..."
    local log_file
    log_file=$(docker inspect --format='{{.LogPath}}' "$($COMPOSE ps -q "$SERVICE" 2>/dev/null | head -1)" 2>/dev/null || true)
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        truncate -s 0 "$log_file"
        log_info "日志已清理: $log_file"
    else
        log_warn "未找到日志文件，跳过清理"
    fi
    $COMPOSE restart "$SERVICE"
    wait_healthy
    log_info "重启完成"
}

cmd_logs() {
    check_deps
    $COMPOSE logs -f
}

cmd_status() {
    check_deps
    echo ""
    $COMPOSE ps
    echo ""
    if curl -sf "${API_BASE}/health" &>/dev/null; then
        log_info "健康检查: ${GREEN}OK${NC}"
    else
        log_warn "健康检查: 服务不可达 (${API_BASE}/health)"
    fi
    echo ""
}

# ── 主入口 ───────────────────────────────────────────────────────────────────

CMD="${1:-up}"

case "$CMD" in
    up)            cmd_up ;;
    update)        cmd_update ;;
    rebuild)       cmd_rebuild ;;
    down)          cmd_down ;;
    restart)       cmd_restart ;;
    clean-restart) cmd_clean_restart ;;
    logs)          cmd_logs ;;
    status)        cmd_status ;;
    *)
        echo "用法: $0 {up|update|rebuild|down|restart|clean-restart|logs|status}"
        echo ""
        echo "  up             首次部署（构建镜像 + 启动）"
        echo "  update         拉取最新代码 + 重启容器（无需重新构建）"
        echo "  rebuild        重新构建镜像（修改了 requirements.txt 时使用）"
        echo "  down           停止并移除容器"
        echo "  restart        重启容器"
        echo "  clean-restart  清理 Docker 日志后重启容器"
        echo "  logs           实时查看日志"
        echo "  status         查看容器状态和健康"
        exit 1
        ;;
esac
