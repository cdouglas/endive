#!/bin/bash
# Helper script for running the development container with Claude Code
# Usage:
#   ./scripts/dev-container.sh                          # Interactive shell
#   ./scripts/dev-container.sh claude                   # Run Claude Code
#   ./scripts/dev-container.sh claude -y                # With --dangerously-skip-permissions
#   ./scripts/dev-container.sh claude --resume ID      # Resume session
#   ./scripts/dev-container.sh claude --resume ID -y   # Resume with skip-permissions
#   ./scripts/dev-container.sh bd list                  # Run beads commands
#   ./scripts/dev-container.sh --build                  # Rebuild and run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Export UID/GID for docker-compose
export USER_UID=$(id -u)
export USER_GID=$(id -g)

COMPOSE_FILE="docker-compose.dev.yml"

# Check for --build flag
BUILD_FLAG=""
if [[ "$1" == "--build" ]]; then
    BUILD_FLAG="--build"
    shift
fi

# Build if image doesn't exist or --build was passed
if [[ -n "$BUILD_FLAG" ]] || ! docker images | grep -q "endive-dev"; then
    echo "Building development container..."
    docker compose -f "$COMPOSE_FILE" build
fi

# Install Python dependencies on first run or if requirements changed
install_deps() {
    docker compose -f "$COMPOSE_FILE" run --rm dev bash -c "
        if [ ! -d /app/lib/python*/site-packages/endive ]; then
            echo 'Installing Python dependencies...'
            pip install --user -r requirements.txt
            pip install --user -e .
        fi
    "
}

if [[ "$1" == "claude" ]]; then
    shift
    # Check for --resume shortcut
    CLAUDE_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --resume)
                if [[ -n "$2" ]]; then
                    CLAUDE_ARGS+=(--resume "$2")
                    shift 2
                else
                    echo "Error: --resume requires a session ID"
                    exit 1
                fi
                ;;
            --skip-permissions|-y)
                CLAUDE_ARGS+=(--dangerously-skip-permissions)
                shift
                ;;
            *)
                CLAUDE_ARGS+=("$1")
                shift
                ;;
        esac
    done
    # Run Claude Code with arguments
    exec docker compose -f "$COMPOSE_FILE" run --rm dev claude "${CLAUDE_ARGS[@]}"
elif [[ "$1" == "bd" || "$1" == "beads" ]]; then
    # Run beads commands
    exec docker compose -f "$COMPOSE_FILE" run --rm dev "$@"
elif [[ "$1" == "install" ]]; then
    # Force reinstall dependencies
    docker compose -f "$COMPOSE_FILE" run --rm dev bash -c "
        pip install --user -r requirements.txt
        pip install --user -e .
    "
elif [[ -n "$1" ]]; then
    # Run arbitrary command
    exec docker compose -f "$COMPOSE_FILE" run --rm dev "$@"
else
    # Interactive shell
    echo "Starting development container..."
    echo "  - venv auto-created on first run"
    echo "  - Claude Code: claude --dangerously-skip-permissions"
    echo "  - Run tests:   pytest tests/ -v"
    exec docker compose -f "$COMPOSE_FILE" run --rm dev
fi
