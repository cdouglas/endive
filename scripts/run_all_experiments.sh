#!/bin/bash
# run_all_experiments.sh
#
# Unified experiment runner for all blog post experiments.
# Supports running locally or in Docker with progress tracking and detach/reattach.
#
# Usage:
#   # Run all experiments locally (foreground)
#   ./scripts/run_all_experiments.sh --parallel 4 --seeds 3
#
#   # Run specific groups
#   ./scripts/run_all_experiments.sh --groups baseline,metadata --seeds 3
#
#   # Run in background with logging (detachable)
#   nohup ./scripts/run_all_experiments.sh --parallel 8 --seeds 3 2>&1 | \
#       tee experiment_logs/run_$(date +%Y%m%d_%H%M%S).log &
#
#   # Check progress
#   ./scripts/run_all_experiments.sh --status
#
#   # Quick test mode
#   ./scripts/run_all_experiments.sh --quick --parallel 4
#
#   # In Docker
#   docker run -d \
#       -e EXP_ARGS="--groups baseline,metadata --seeds 3 --parallel 8" \
#       -v $(pwd)/experiments:/app/experiments \
#       -v $(pwd)/experiment_logs:/app/experiment_logs \
#       cdouglas/endive-sim:latest \
#       bash -c "scripts/run_all_experiments.sh \${EXP_ARGS} 2>&1 | \
#           tee experiment_logs/run_\$(date +%Y%m%d_%H%M%S).log"
#
# Available groups:
#   trivial      - Single table trivial conflicts (Q1a)
#   mixed        - Single table mixed conflicts (Q1b)
#   multi_table  - Multi-table experiments (Q2a/2b)
#   baseline     - Baseline configs for S3/S3x/Azure
#   metadata     - Metadata not inlined experiments
#   ml_append    - Manifest list append experiments
#   combined     - Combined optimizations
#
# All groups if not specified.

set -e

# ============================================================================
# Default Configuration
# ============================================================================
PARALLEL=4
SEEDS=3
GROUPS=""
QUICK=false
DRY_RUN=false
STATUS=false
RESUME=false
FORCE=false

# ============================================================================
# Parse Arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel|-p)
            PARALLEL="$2"
            shift 2
            ;;
        --seeds|-s)
            SEEDS="$2"
            shift 2
            ;;
        --groups|-g)
            GROUPS="$2"
            shift 2
            ;;
        --quick|-q)
            QUICK=true
            shift
            ;;
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --status)
            STATUS=true
            shift
            ;;
        --resume|-r)
            RESUME=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --help|-h)
            head -n 50 "$0" | tail -n +2 | grep -E "^#" | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create log directory
mkdir -p experiment_logs

# Check if running in Docker
if [ -f "/.dockerenv" ] || [ -n "$DOCKER_CONTAINER" ]; then
    echo "Running in Docker container"
else
    # Activate virtual environment if exists
    if [ -f "bin/activate" ]; then
        source bin/activate
    elif [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
fi

# Verify Python and module available
if ! python -c "import endive.main" 2>/dev/null; then
    echo "Error: endive module not found. Install with 'pip install -e .'"
    exit 1
fi

# ============================================================================
# Build Python Command
# ============================================================================
CMD="python scripts/run_all_experiments.py"
CMD="$CMD --parallel $PARALLEL"
CMD="$CMD --seeds $SEEDS"

if [ -n "$GROUPS" ]; then
    CMD="$CMD --groups $GROUPS"
fi

if [ "$QUICK" = true ]; then
    CMD="$CMD --quick"
fi

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
fi

if [ "$STATUS" = true ]; then
    CMD="$CMD --status"
fi

if [ "$RESUME" = true ]; then
    CMD="$CMD --resume"
fi

if [ "$FORCE" = true ]; then
    CMD="$CMD --force"
fi

# ============================================================================
# Run
# ============================================================================
echo "========================================"
echo "  Endive Experiment Runner"
echo "========================================"
echo "Command: $CMD"
echo "Working directory: $PROJECT_DIR"
echo "========================================"
echo ""

exec $CMD
