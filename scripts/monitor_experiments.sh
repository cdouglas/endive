#!/bin/bash
# monitor_experiments.sh
#
# Monitor progress of running experiments
#
# Usage:
#   ./monitor_experiments.sh [OPTIONS]
#
# Options:
#   --watch N    Auto-refresh every N seconds (default: 10)
#   --summary    Show summary only (no detailed logs)
#
# Examples:
#   # Watch progress with auto-refresh
#   ./monitor_experiments.sh --watch 5
#
#   # One-time summary
#   ./monitor_experiments.sh --summary

# ============================================================================
# Configuration
# ============================================================================

WATCH_INTERVAL=10
WATCH_MODE=false
SUMMARY_ONLY=false
LOG_DIR="experiment_logs"

# ============================================================================
# Parse arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --watch)
            WATCH_MODE=true
            WATCH_INTERVAL="${2:-10}"
            shift 2
            ;;
        --summary)
            SUMMARY_ONLY=true
            shift
            ;;
        --help|-h)
            head -n 15 "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Helper Functions
# ============================================================================

show_status() {
    clear

    echo "========================================"
    echo "EXPERIMENT PROGRESS MONITOR"
    echo "========================================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Find latest log file
    LATEST_LOG=$(ls -t "$LOG_DIR"/baseline_experiments_*.log 2>/dev/null | head -n1)

    if [ -z "$LATEST_LOG" ]; then
        echo "No experiment logs found in $LOG_DIR/"
        echo ""
        echo "Start experiments with:"
        echo "  ./run_baseline_experiments.sh --seeds 3"
        return
    fi

    echo "Log file: $LATEST_LOG"
    echo ""

    # Extract progress from log
    TOTAL_RUNS=$(grep "Total simulations to run:" "$LATEST_LOG" | tail -n1 | awk '{print $NF}')
    CURRENT_RUN=$(grep -o '\[[0-9]*/[0-9]*\]' "$LATEST_LOG" | tail -n1 | tr -d '[]' | cut -d'/' -f1)

    if [ -n "$TOTAL_RUNS" ] && [ -n "$CURRENT_RUN" ]; then
        PERCENT=$((CURRENT_RUN * 100 / TOTAL_RUNS))
        REMAINING=$((TOTAL_RUNS - CURRENT_RUN))

        echo "Overall Progress: $CURRENT_RUN / $TOTAL_RUNS ($PERCENT%)"
        echo "Remaining: $REMAINING simulations"

        # Progress bar
        BAR_WIDTH=50
        FILLED=$((PERCENT * BAR_WIDTH / 100))
        printf "["
        printf "%${FILLED}s" | tr ' ' '='
        printf "%$((BAR_WIDTH - FILLED))s" | tr ' ' '-'
        printf "] $PERCENT%%\n"
        echo ""
    fi

    # Count successes and failures
    SUCCESS_COUNT=$(grep -c "✓ Success" "$LATEST_LOG" 2>/dev/null || echo 0)
    FAIL_COUNT=$(grep -c "✗ Failed" "$LATEST_LOG" 2>/dev/null || echo 0)

    echo "Results:"
    echo "  ✓ Successful: $SUCCESS_COUNT"
    echo "  ✗ Failed: $FAIL_COUNT"
    echo ""

    # Show current experiment
    CURRENT_EXP=$(grep -E "EXPERIMENT|Seed [0-9]" "$LATEST_LOG" | tail -n5)
    if [ -n "$CURRENT_EXP" ]; then
        echo "Current Activity:"
        echo "$CURRENT_EXP" | sed 's/^/  /'
        echo ""
    fi

    # Check for completion
    if grep -q "BASELINE EXPERIMENTS COMPLETE" "$LATEST_LOG"; then
        echo "=========================================="
        echo "EXPERIMENTS COMPLETE!"
        echo "=========================================="
        echo ""
        grep "Experiment 2\." "$LATEST_LOG" | tail -n2
        echo ""
        return 1  # Signal completion
    fi

    # Show recent activity (if not summary only)
    if [ "$SUMMARY_ONLY" = false ]; then
        echo "----------------------------------------"
        echo "Recent Activity (last 10 lines):"
        echo "----------------------------------------"
        tail -n10 "$LATEST_LOG" | sed 's/^/  /'
        echo ""
    fi

    # Experiment directory sizes
    echo "----------------------------------------"
    echo "Output Directory Status:"
    echo "----------------------------------------"
    if [ -d "experiments" ]; then
        EXP2_1_DIRS=$(ls -d experiments/exp2_1_* 2>/dev/null | wc -l)
        EXP2_2_DIRS=$(ls -d experiments/exp2_2_* 2>/dev/null | wc -l)
        TOTAL_SIZE=$(du -sh experiments 2>/dev/null | awk '{print $1}')

        echo "  Experiment 2.1 directories: $EXP2_1_DIRS"
        echo "  Experiment 2.2 directories: $EXP2_2_DIRS"
        echo "  Total size: $TOTAL_SIZE"
    else
        echo "  No experiments directory yet"
    fi

    echo ""

    if [ "$WATCH_MODE" = true ]; then
        echo "Refreshing every ${WATCH_INTERVAL}s... (Ctrl+C to exit)"
    fi

    return 0
}

# ============================================================================
# Main
# ============================================================================

if [ "$WATCH_MODE" = true ]; then
    # Watch mode - auto refresh
    while true; do
        show_status
        if [ $? -eq 1 ]; then
            # Experiments complete
            break
        fi
        sleep "$WATCH_INTERVAL"
    done
else
    # One-time display
    show_status
fi
