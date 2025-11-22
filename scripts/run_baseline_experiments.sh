#!/bin/bash
# run_baseline_experiments.sh
#
# Runs baseline experiments (Phase 2) from ANALYSIS_PLAN.md
#
# Usage:
#   ./run_baseline_experiments.sh [OPTIONS]
#
# Options:
#   --seeds N          Number of seeds per configuration (default: 3)
#   --exp2.1           Run only Experiment 2.1 (single table)
#   --exp2.2           Run only Experiment 2.2 (multi-table)
#   --quick            Quick test mode (fewer configs, shorter duration)
#   --dry-run          Show what would be run without executing
#
# Examples:
#   # Run all baseline experiments with 5 seeds each
#   ./run_baseline_experiments.sh --seeds 5
#
#   # Run in background with logging
#   nohup ./run_baseline_experiments.sh --seeds 3 > experiments.log 2>&1 &
#
#   # Quick test run
#   ./run_baseline_experiments.sh --quick --seeds 1

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Default values
NUM_SEEDS=3
RUN_EXP2_1=true
RUN_EXP2_2=true
QUICK_MODE=false
DRY_RUN=false

# Experiment parameters
EXP2_1_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP2_2_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP2_2_TABLES=(1 2 5 10 20 50)

# Quick mode parameters (for testing)
QUICK_LOADS=(100 500 2000)
QUICK_TABLES=(1 5 10)
QUICK_DURATION=10000  # 10 seconds

# Base config files
EXP2_1_CONFIG="experiment_configs/exp2_1_single_table_false_conflicts.toml"
EXP2_2_CONFIG="experiment_configs/exp2_2_multi_table_false_conflicts.toml"

# Logging
LOG_DIR="experiment_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/baseline_experiments_${TIMESTAMP}.log"

# ============================================================================
# Parse command line arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --exp2.1)
            RUN_EXP2_1=true
            RUN_EXP2_2=false
            shift
            ;;
        --exp2.2)
            RUN_EXP2_1=false
            RUN_EXP2_2=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            head -n 20 "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================

# Create log directory
mkdir -p "$LOG_DIR"

# Activate virtual environment
if [ ! -f "bin/activate" ]; then
    echo "Error: Virtual environment not found. Please run from project root."
    exit 1
fi

source bin/activate

# Verify icecap module is available
if ! python -c "import icecap.main" 2>/dev/null; then
    echo "Error: icecap module not found. Please install with 'pip install -e .'"
    exit 1
fi

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "$*" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
}

estimate_duration() {
    local num_runs=$1
    local avg_time_per_run=$2  # in seconds
    local total_seconds=$((num_runs * avg_time_per_run))

    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))

    if [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m"
    else
        echo "${minutes}m"
    fi
}

run_experiment() {
    local config_file=$1
    local output_file=$2

    if [ "$DRY_RUN" = true ]; then
        log "[DRY RUN] Would run: python -m icecap.main $output_file"
        return 0
    fi

    # Run simulation (auto-confirm with 'Y')
    echo "Y" | python -m icecap.main "$output_file" >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        log "  ✓ Success"
        return 0
    else
        log "  ✗ Failed (check log for details)"
        return 1
    fi
}

create_config_variant() {
    local base_config=$1
    local output_file=$2
    shift 2
    local modifications=("$@")

    # Copy base config
    cp "$base_config" "$output_file"

    # Apply modifications
    for mod in "${modifications[@]}"; do
        local param=$(echo "$mod" | cut -d'=' -f1)
        local value=$(echo "$mod" | cut -d'=' -f2)

        # Use sed to replace parameter value
        # Handle both "param = value" and "param=value" formats
        sed -i "s/^${param}[[:space:]]*=.*/${param} = ${value}/" "$output_file"
    done
}

# ============================================================================
# Calculate total runs
# ============================================================================

TOTAL_RUNS=0

if [ "$QUICK_MODE" = true ]; then
    log "Running in QUICK MODE (reduced parameters for testing)"
    EXP2_1_LOADS=("${QUICK_LOADS[@]}")
    EXP2_2_LOADS=("${QUICK_LOADS[@]}")
    EXP2_2_TABLES=("${QUICK_TABLES[@]}")
fi

if [ "$RUN_EXP2_1" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP2_1_LOADS[@]} * NUM_SEEDS))
fi

if [ "$RUN_EXP2_2" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP2_2_LOADS[@]} * ${#EXP2_2_TABLES[@]} * NUM_SEEDS))
fi

# ============================================================================
# Print Summary
# ============================================================================

log_section "BASELINE EXPERIMENTS - PHASE 2"
log "Configuration:"
log "  Number of seeds per config: $NUM_SEEDS"
log "  Run Experiment 2.1: $RUN_EXP2_1"
log "  Run Experiment 2.2: $RUN_EXP2_2"
log "  Quick mode: $QUICK_MODE"
log "  Dry run: $DRY_RUN"
log ""
log "Total simulations to run: $TOTAL_RUNS"

if [ "$QUICK_MODE" = true ]; then
    AVG_TIME=15  # 15 seconds per quick run
else
    AVG_TIME=180  # 3 minutes per full run (conservative estimate)
fi

ESTIMATED_TIME=$(estimate_duration $TOTAL_RUNS $AVG_TIME)
log "Estimated total time: $ESTIMATED_TIME"
log ""
log "Log file: $LOG_FILE"
log "Output directory: experiments/"

if [ "$DRY_RUN" = false ]; then
    log ""
    log "Starting experiments in 3 seconds... (Ctrl+C to cancel)"
    sleep 3
fi

# ============================================================================
# Experiment 2.1: Single Table Saturation (False Conflicts)
# ============================================================================

if [ "$RUN_EXP2_1" = true ]; then
    log_section "EXPERIMENT 2.1: Single Table Saturation (False Conflicts)"
    log "Research Question: What is the maximum throughput for a single table?"
    log ""
    log "Sweeping inter_arrival.scale: ${EXP2_1_LOADS[*]}"
    log "Seeds per configuration: $NUM_SEEDS"
    log ""

    CURRENT_RUN=0
    EXP2_1_SUCCESS=0
    EXP2_1_FAILED=0

    for load in "${EXP2_1_LOADS[@]}"; do
        log "Load: inter_arrival.scale = ${load}ms (~$((1000/load)) txn/sec offered)"

        for seed_num in $(seq 1 $NUM_SEEDS); do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

            log "$PROGRESS  Seed $seed_num/$NUM_SEEDS"

            # Create temporary config with modified parameters
            TEMP_CONFIG=$(mktemp)

            if [ "$QUICK_MODE" = true ]; then
                create_config_variant "$EXP2_1_CONFIG" "$TEMP_CONFIG" \
                    "inter_arrival.scale=${load}" \
                    "duration_ms=${QUICK_DURATION}"
            else
                create_config_variant "$EXP2_1_CONFIG" "$TEMP_CONFIG" \
                    "inter_arrival.scale=${load}"
            fi

            # Run experiment
            if run_experiment "$EXP2_1_CONFIG" "$TEMP_CONFIG"; then
                EXP2_1_SUCCESS=$((EXP2_1_SUCCESS + 1))
            else
                EXP2_1_FAILED=$((EXP2_1_FAILED + 1))
            fi

            rm "$TEMP_CONFIG"
        done

        log ""
    done

    log "Experiment 2.1 complete: $EXP2_1_SUCCESS succeeded, $EXP2_1_FAILED failed"
fi

# ============================================================================
# Experiment 2.2: Multi-Table Saturation (False Conflicts)
# ============================================================================

if [ "$RUN_EXP2_2" = true ]; then
    log_section "EXPERIMENT 2.2: Multi-Table Saturation (False Conflicts)"
    log "Research Question: How does table count affect throughput?"
    log ""
    log "Sweeping num_tables: ${EXP2_2_TABLES[*]}"
    log "Sweeping inter_arrival.scale: ${EXP2_2_LOADS[*]}"
    log "Seeds per configuration: $NUM_SEEDS"
    log ""

    EXP2_2_SUCCESS=0
    EXP2_2_FAILED=0

    for num_tables in "${EXP2_2_TABLES[@]}"; do
        log "Tables: num_tables = $num_tables (num_groups = $num_tables)"

        for load in "${EXP2_2_LOADS[@]}"; do
            log "  Load: inter_arrival.scale = ${load}ms"

            for seed_num in $(seq 1 $NUM_SEEDS); do
                CURRENT_RUN=$((CURRENT_RUN + 1))
                PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

                log "  $PROGRESS  Seed $seed_num/$NUM_SEEDS"

                # Create temporary config with modified parameters
                TEMP_CONFIG=$(mktemp)

                if [ "$QUICK_MODE" = true ]; then
                    create_config_variant "$EXP2_2_CONFIG" "$TEMP_CONFIG" \
                        "num_tables=${num_tables}" \
                        "num_groups=${num_tables}" \
                        "inter_arrival.scale=${load}" \
                        "duration_ms=${QUICK_DURATION}"
                else
                    create_config_variant "$EXP2_2_CONFIG" "$TEMP_CONFIG" \
                        "num_tables=${num_tables}" \
                        "num_groups=${num_tables}" \
                        "inter_arrival.scale=${load}"
                fi

                # Run experiment
                if run_experiment "$EXP2_2_CONFIG" "$TEMP_CONFIG"; then
                    EXP2_2_SUCCESS=$((EXP2_2_SUCCESS + 1))
                else
                    EXP2_2_FAILED=$((EXP2_2_FAILED + 1))
                fi

                rm "$TEMP_CONFIG"
            done
        done

        log ""
    done

    log "Experiment 2.2 complete: $EXP2_2_SUCCESS succeeded, $EXP2_2_FAILED failed"
fi

# ============================================================================
# Final Summary
# ============================================================================

log_section "BASELINE EXPERIMENTS COMPLETE"
log "Total runs: $TOTAL_RUNS"

if [ "$RUN_EXP2_1" = true ]; then
    log "  Experiment 2.1: $EXP2_1_SUCCESS succeeded, $EXP2_1_FAILED failed"
fi

if [ "$RUN_EXP2_2" = true ]; then
    log "  Experiment 2.2: $EXP2_2_SUCCESS succeeded, $EXP2_2_FAILED failed"
fi

log ""
log "Results stored in: experiments/"
log "  exp2_1_single_table_false-*/"
log "  exp2_2_multi_table_false-*/"
log ""
log "Next steps:"
log "  1. Verify results: ls -lh experiments/"
log "  2. Run analysis: python -m icecap.analysis all -i experiments/exp2_1_* -o plots/exp2_1"
log "  3. Generate visualizations for all experiments"

if [ "$DRY_RUN" = true ]; then
    log ""
    log "DRY RUN COMPLETE - No experiments were actually executed"
fi
