#!/bin/bash
# run_baseline_experiments.sh
#
# Unified driver for all experiment suites
#
# Usage:
#   ./run_baseline_experiments.sh [OPTIONS]
#
# Options:
#   --seeds N          Number of seeds per configuration (default: 3)
#   --parallel N       Number of concurrent experiments (default: # of CPU cores)
#   --exp2.1           Run Experiment 2.1 (single table, false conflicts)
#   --exp2.2           Run Experiment 2.2 (multi-table, false conflicts)
#   --exp3.1           Run Experiment 3.1 (single table, real conflicts)
#   --exp3.2           Run Experiment 3.2 (manifest count distribution)
#   --exp3.3           Run Experiment 3.3 (multi-table, real conflicts)
#   --exp3.4           Run Experiment 3.4 (multi-table, real conflicts with backoff)
#   --exp4.1           Run Experiment 4.1 (single table, false conflicts with backoff)
#   --exp5.1           Run Experiment 5.1 (single table, catalog latency)
#   --exp5.2           Run Experiment 5.2 (multi-table, catalog latency)
#   --exp5.3           Run Experiment 5.3 (transaction partitioning, catalog latency)
#   --quick            Quick test mode (fewer configs, shorter duration)
#   --dry-run          Show what would be run without executing
#   --help, -h         Show this help message
#
# If no --expM.N flags are specified, runs all experiments (Phase 2, 3, 4)
#
# Examples:
#   # Run all experiments (Phase 2, 3, & 4) with 5 seeds
#   ./run_baseline_experiments.sh --seeds 5
#
#   # Run only Phase 2 baseline experiments
#   ./run_baseline_experiments.sh --exp2.1 --exp2.2
#
#   # Run only Phase 3 real conflict experiments
#   ./run_baseline_experiments.sh --exp3.1 --exp3.2 --exp3.3
#
#   # Quick test of exp3.1
#   ./run_baseline_experiments.sh --exp3.1 --quick --seeds 1
#
#   # Run Phase 5 catalog latency experiments
#   ./run_baseline_experiments.sh --exp5.1 --exp5.2 --exp5.3
#
#   # Run with 8 parallel jobs in background
#   nohup ./run_baseline_experiments.sh --parallel 8 --seeds 3 > experiments.log 2>&1 &

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Default values
NUM_SEEDS=3
NUM_PARALLEL=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
QUICK_MODE=false
DRY_RUN=false

# Experiment selection (if none specified, run all)
RUN_EXP2_1=false
RUN_EXP2_2=false
RUN_EXP3_1=false
RUN_EXP3_2=false
RUN_EXP3_3=false
RUN_EXP3_4=false
RUN_EXP4_1=false
RUN_EXP5_1=false
RUN_EXP5_2=false
RUN_EXP5_3=false
ANY_EXP_SPECIFIED=false

# Experiment parameters
# Phase 2: False conflicts (baseline)
EXP2_1_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP2_2_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP2_2_TABLES=(1 2 5 10 20 50)

# Phase 3: Real conflicts
EXP3_1_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP3_1_REAL_PROBS=(0.0 0.1 0.2 0.3 0.5 0.7 1.0)

EXP3_2_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP3_2_DISTS=("fixed:1" "fixed:5" "fixed:10" "exponential:3")

EXP3_3_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP3_3_TABLES=(1 2 5 10 20)
EXP3_3_REAL_PROBS=(0.0 0.1 0.3 0.5)

EXP3_4_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP3_4_TABLES=(1 2 5 10 20)
EXP3_4_REAL_PROBS=(0.0 0.1 0.3 0.5)

# Phase 4: Exponential backoff
EXP4_1_LOADS=(10 20 50 100 200 500 1000 2000 5000)

# Phase 5: Catalog latency impact
EXP5_1_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP5_1_CAS_LATENCIES=(15 50 100 200 500 1000)

EXP5_2_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP5_2_CAS_LATENCIES=(15 50 100 200 500 1000)
EXP5_2_TABLES=(1 5 20 50)

EXP5_3_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP5_3_CAS_LATENCIES=(15 50 100 200 500 1000)
EXP5_3_GROUPS=(1 2 5 10 20)

# Quick mode parameters (for testing)
QUICK_LOADS=(100 500 2000)
QUICK_TABLES=(1 5 10)
QUICK_REAL_PROBS=(0.0 0.5 1.0)
QUICK_DISTS=("fixed:1" "exponential:3")
QUICK_DURATION=10000  # 10 seconds

# Base config files
EXP2_1_CONFIG="experiment_configs/exp2_1_single_table_false_conflicts.toml"
EXP2_2_CONFIG="experiment_configs/exp2_2_multi_table_false_conflicts.toml"
EXP3_1_CONFIG="experiment_configs/exp3_1_single_table_real_conflicts.toml"
EXP3_2_CONFIG="experiment_configs/exp3_2_manifest_count_distribution.toml"
EXP3_3_CONFIG="experiment_configs/exp3_3_multi_table_real_conflicts.toml"
EXP3_4_CONFIG="experiment_configs/exp3_4_multi_table_real_backoff.toml"
EXP4_1_CONFIG="experiment_configs/exp4_1_single_table_false_backoff.toml"
EXP5_1_CONFIG="experiment_configs/exp5_1_single_table_catalog_latency.toml"
EXP5_2_CONFIG="experiment_configs/exp5_2_multi_table_catalog_latency.toml"
EXP5_3_CONFIG="experiment_configs/exp5_3_transaction_partitioning_catalog_latency.toml"

# Logging
LOG_DIR="experiment_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/experiments_${TIMESTAMP}.log"

# Job tracking
declare -a JOB_PIDS=()
declare -a JOB_CONFIGS=()
JOB_SUCCESS=0
JOB_FAILED=0

# ============================================================================
# Parse command line arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --parallel)
            NUM_PARALLEL="$2"
            shift 2
            ;;
        --exp2.1)
            RUN_EXP2_1=true
            ANY_EXP_SPECIFIED=true
            shift
            ;;
        --exp2.2)
            RUN_EXP2_2=true
            ANY_EXP_SPECIFIED=true
            shift
            ;;
        --exp3.1)
            RUN_EXP3_1=true
            ANY_EXP_SPECIFIED=true
            shift
            ;;
        --exp3.2)
            RUN_EXP3_2=true
            ANY_EXP_SPECIFIED=true
            shift
            ;;
        --exp3.3)
            RUN_EXP3_3=true
            ANY_EXP_SPECIFIED=true
            shift
            ;;
        --exp3.4)
            RUN_EXP3_4=true
            ANY_EXP_SPECIFIED=true
            shift
            ;;
        --exp4.1)
            RUN_EXP4_1=true
            ANY_EXP_SPECIFIED=true
            shift
            ;;
        --exp5.1)
            RUN_EXP5_1=true
            ANY_EXP_SPECIFIED=true
            shift
            ;;
        --exp5.2)
            RUN_EXP5_2=true
            ANY_EXP_SPECIFIED=true
            shift
            ;;
        --exp5.3)
            RUN_EXP5_3=true
            ANY_EXP_SPECIFIED=true
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
            head -n 35 "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If no experiments specified, run all
if [ "$ANY_EXP_SPECIFIED" = false ]; then
    RUN_EXP2_1=true
    RUN_EXP2_2=true
    RUN_EXP3_1=true
    RUN_EXP3_2=true
    RUN_EXP3_3=true
    RUN_EXP3_4=true
    RUN_EXP4_1=true
fi

# ============================================================================
# Setup
# ============================================================================

# Create log directory
mkdir -p "$LOG_DIR"

# Check if running in Docker (no virtual environment)
if [ -f "/.dockerenv" ] || [ -n "$DOCKER_CONTAINER" ]; then
    echo "Running in Docker container"
else
    # Activate virtual environment
    if [ ! -f "bin/activate" ]; then
        echo "Error: Virtual environment not found. Please run from project root."
        exit 1
    fi
    source bin/activate
fi

# Verify endive module is available
if ! python -c "import endive.main" 2>/dev/null; then
    echo "Error: endive module not found. Please install with 'pip install -e .'"
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
    local num_parallel=$3

    # Calculate wall-clock time with parallelism
    local total_seconds=$(( (num_runs * avg_time_per_run + num_parallel - 1) / num_parallel ))

    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))

    if [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m"
    else
        echo "${minutes}m"
    fi
}

run_experiment_background() {
    local config_file=$1
    local output_file=$2
    local description=$3

    if [ "$DRY_RUN" = true ]; then
        log "[DRY RUN] Would run: python -m endive.main $output_file"
        return 0
    fi

    # Run simulation in background (auto-confirm with 'Y', disable progress bar for clean logs)
    {
        echo "Y" | python -m endive.main --no-progress "$output_file" >> "$LOG_FILE" 2>&1
        exit_code=$?

        # Write result to temp file
        echo "$exit_code|$description" > "$output_file.status"
        exit $exit_code
    } &

    local pid=$!
    JOB_PIDS+=($pid)
    JOB_CONFIGS+=("$description")
}

wait_for_job_slot() {
    # Wait until we have fewer than NUM_PARALLEL running jobs
    while [ ${#JOB_PIDS[@]} -ge $NUM_PARALLEL ]; do
        check_completed_jobs
        sleep 1
    done
}

check_completed_jobs() {
    local new_pids=()
    local new_configs=()

    for i in "${!JOB_PIDS[@]}"; do
        local pid=${JOB_PIDS[$i]}
        local config="${JOB_CONFIGS[$i]}"

        if ! kill -0 $pid 2>/dev/null; then
            # Job has finished, check status
            wait $pid
            local exit_code=$?

            if [ $exit_code -eq 0 ]; then
                log "  ✓ Success: $config"
                JOB_SUCCESS=$((JOB_SUCCESS + 1))
            else
                log "  ✗ Failed: $config"
                JOB_FAILED=$((JOB_FAILED + 1))
            fi
        else
            # Job still running, keep it in the list
            new_pids+=($pid)
            new_configs+=("$config")
        fi
    done

    JOB_PIDS=("${new_pids[@]}")
    JOB_CONFIGS=("${new_configs[@]}")
}

wait_for_all_jobs() {
    log ""
    log "Waiting for all jobs to complete..."

    while [ ${#JOB_PIDS[@]} -gt 0 ]; do
        check_completed_jobs
        if [ ${#JOB_PIDS[@]} -gt 0 ]; then
            log "  ${#JOB_PIDS[@]} jobs still running..."
            sleep 5
        fi
    done

    log "All jobs complete"
}

generate_or_load_session_nonce() {
    # Generate or load session nonce for deterministic seed generation
    # This makes experiments reentrant within a session but unique across sessions
    local nonce_file="experiments/.nonce"

    if [ -f "$nonce_file" ]; then
        # Reentrant: reuse existing nonce
        cat "$nonce_file"
        echo "[Session nonce] Reusing existing nonce from $nonce_file" >&2
    else
        # First run: generate new nonce
        local nonce=$(openssl rand -hex 16 2>/dev/null || cat /dev/urandom | head -c 16 | xxd -p)
        echo "$nonce" > "$nonce_file"
        echo "[Session nonce] Generated new nonce: $nonce" >&2
        echo "$nonce"
    fi
}

generate_deterministic_seed() {
    # Generate a deterministic seed from session nonce + experiment parameters
    # Uses: session nonce + experiment label + parameter string + seed_num
    local exp_label="$1"
    local param_string="$2"
    local seed_num="$3"

    # Create a hash of the input and convert to a positive integer seed
    # Uses SHA256 and takes first 8 hex digits (32 bits)
    local hash_input="${SESSION_NONCE}:${exp_label}:${param_string}:${seed_num}"
    local seed=$(echo -n "$hash_input" | sha256sum | cut -c1-8)
    # Convert hex to decimal
    echo $((0x${seed}))
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
        local value=$(echo "$mod" | cut -d'=' -f2-)

        # Try to replace existing uncommented line
        if grep -q "^${param}[[:space:]]*=" "$output_file" 2>/dev/null; then
            sed -i.bak "s/^${param}[[:space:]]*=.*/${param} = ${value}/" "$output_file"
        # Try to uncomment and replace commented line
        elif grep -q "^#[[:space:]]*${param}[[:space:]]*=" "$output_file" 2>/dev/null; then
            sed -i.bak "s/^#[[:space:]]*${param}[[:space:]]*=.*/${param} = ${value}/" "$output_file"
        # Otherwise append after [simulation] section
        else
            sed -i.bak "/^\[simulation\]/a ${param} = ${value}" "$output_file"
        fi
        rm -f "$output_file.bak"
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
    EXP3_1_LOADS=("${QUICK_LOADS[@]}")
    EXP3_1_REAL_PROBS=("${QUICK_REAL_PROBS[@]}")
    EXP3_2_LOADS=("${QUICK_LOADS[@]}")
    EXP3_2_DISTS=("${QUICK_DISTS[@]}")
    EXP3_3_LOADS=("${QUICK_LOADS[@]}")
    EXP3_3_TABLES=("${QUICK_TABLES[@]}")
    EXP3_3_REAL_PROBS=("${QUICK_REAL_PROBS[@]}")
fi

if [ "$RUN_EXP2_1" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP2_1_LOADS[@]} * NUM_SEEDS))
fi

if [ "$RUN_EXP2_2" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP2_2_LOADS[@]} * ${#EXP2_2_TABLES[@]} * NUM_SEEDS))
fi

if [ "$RUN_EXP3_1" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP3_1_LOADS[@]} * ${#EXP3_1_REAL_PROBS[@]} * NUM_SEEDS))
fi

if [ "$RUN_EXP3_2" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP3_2_LOADS[@]} * ${#EXP3_2_DISTS[@]} * NUM_SEEDS))
fi

if [ "$RUN_EXP3_3" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP3_3_LOADS[@]} * ${#EXP3_3_TABLES[@]} * ${#EXP3_3_REAL_PROBS[@]} * NUM_SEEDS))
fi

if [ "$RUN_EXP3_4" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP3_4_LOADS[@]} * ${#EXP3_4_TABLES[@]} * ${#EXP3_4_REAL_PROBS[@]} * NUM_SEEDS))
fi

if [ "$RUN_EXP4_1" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP4_1_LOADS[@]} * NUM_SEEDS))
fi

if [ "$RUN_EXP5_1" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP5_1_LOADS[@]} * ${#EXP5_1_CAS_LATENCIES[@]} * NUM_SEEDS))
fi

if [ "$RUN_EXP5_2" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP5_2_LOADS[@]} * ${#EXP5_2_CAS_LATENCIES[@]} * ${#EXP5_2_TABLES[@]} * NUM_SEEDS))
fi

if [ "$RUN_EXP5_3" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP5_3_LOADS[@]} * ${#EXP5_3_CAS_LATENCIES[@]} * ${#EXP5_3_GROUPS[@]} * NUM_SEEDS))
fi

# ============================================================================
# Print Summary
# ============================================================================

log_section "EXPERIMENT SUITE - PARALLEL EXECUTION"

# Get git SHA for reproducibility tracking
GIT_SHA="${GIT_SHA:-$(git rev-parse HEAD 2>/dev/null || echo 'unknown')}"
GIT_SHA_SHORT="${GIT_SHA:0:7}"

log "Configuration:"
log "  Git SHA: $GIT_SHA_SHORT ($GIT_SHA)"
log "  Number of seeds per config: $NUM_SEEDS"
log "  Parallel jobs: $NUM_PARALLEL"
log "  Quick mode: $QUICK_MODE"
log "  Dry run: $DRY_RUN"
log ""
log "Experiments selected:"
log "  Phase 2 (False Conflicts):"
log "    Exp 2.1 (Single table): $RUN_EXP2_1"
log "    Exp 2.2 (Multi-table): $RUN_EXP2_2"
log "  Phase 3 (Real Conflicts):"
log "    Exp 3.1 (Single table, varying prob): $RUN_EXP3_1"
log "    Exp 3.2 (Manifest count distribution): $RUN_EXP3_2"
log "    Exp 3.3 (Multi-table, real conflicts): $RUN_EXP3_3"
log "  Phase 5 (Catalog Latency):"
log "    Exp 5.1 (Single table, catalog latency): $RUN_EXP5_1"
log "    Exp 5.2 (Multi-table, catalog latency): $RUN_EXP5_2"
log "    Exp 5.3 (Transaction partitioning, catalog latency): $RUN_EXP5_3"
log ""
log "Total simulations to run: $TOTAL_RUNS"

if [ "$QUICK_MODE" = true ]; then
    AVG_TIME=15  # 15 seconds per quick run
else
    AVG_TIME=3600  # 1 hour per full run
fi

ESTIMATED_TIME=$(estimate_duration $TOTAL_RUNS $AVG_TIME $NUM_PARALLEL)
log "Estimated wall-clock time: $ESTIMATED_TIME (with $NUM_PARALLEL parallel jobs)"
log ""
log "Log file: $LOG_FILE"
log "Output directory: experiments/"

if [ "$DRY_RUN" = false ]; then
    log ""
    log "Starting experiments in 3 seconds... (Ctrl+C to cancel)"
    sleep 3
fi

# Generate or load session nonce for reentrant seed generation
mkdir -p experiments
SESSION_NONCE=$(generate_or_load_session_nonce)
export SESSION_NONCE
log ""

CURRENT_RUN=0

# ============================================================================
# Experiment 2.1: Single Table Saturation (False Conflicts)
# ============================================================================

if [ "$RUN_EXP2_1" = true ]; then
    log_section "EXPERIMENT 2.1: Single Table Saturation (False Conflicts)"
    log "Research Question 1a: Maximum throughput for single table?"
    log ""
    log "Sweeping inter_arrival.scale: ${EXP2_1_LOADS[*]}"
    log "Seeds per configuration: $NUM_SEEDS"
    log ""

    for load in "${EXP2_1_LOADS[@]}"; do
        for seed_num in $(seq 1 $NUM_SEEDS); do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

            # Generate deterministic seed
            SEED=$(generate_deterministic_seed "exp2_1" "load=${load}" "$seed_num")

            TEMP_CONFIG=$(mktemp)

            if [ "$QUICK_MODE" = true ]; then
                create_config_variant "$EXP2_1_CONFIG" "$TEMP_CONFIG" \
                    "seed=${SEED}" \
                    "inter_arrival.scale=${load}" \
                    "duration_ms=${QUICK_DURATION}"
            else
                create_config_variant "$EXP2_1_CONFIG" "$TEMP_CONFIG" \
                    "seed=${SEED}" \
                    "inter_arrival.scale=${load}"
            fi

            wait_for_job_slot
            DESC="$PROGRESS Exp2.1 load=$load seed=$SEED"
            log "  Starting: $DESC"
            run_experiment_background "$EXP2_1_CONFIG" "$TEMP_CONFIG" "$DESC"
        done
    done

    log "All Experiment 2.1 jobs submitted"
fi

# ============================================================================
# Experiment 2.2: Multi-Table Saturation (False Conflicts)
# ============================================================================

if [ "$RUN_EXP2_2" = true ]; then
    log_section "EXPERIMENT 2.2: Multi-Table Saturation (False Conflicts)"
    log "Research Question 2a: How does table count affect throughput?"
    log ""
    log "Sweeping num_tables: ${EXP2_2_TABLES[*]}"
    log "Sweeping inter_arrival.scale: ${EXP2_2_LOADS[*]}"
    log ""

    for num_tables in "${EXP2_2_TABLES[@]}"; do
        for load in "${EXP2_2_LOADS[@]}"; do
            for seed_num in $(seq 1 $NUM_SEEDS); do
                CURRENT_RUN=$((CURRENT_RUN + 1))
                PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

                # Generate deterministic seed
                SEED=$(generate_deterministic_seed "exp2_2" "tables=${num_tables}:load=${load}" "$seed_num")

                TEMP_CONFIG=$(mktemp)

                if [ "$QUICK_MODE" = true ]; then
                    create_config_variant "$EXP2_2_CONFIG" "$TEMP_CONFIG" \
                        "seed=${SEED}" \
                        "num_tables=${num_tables}" \
                        "num_groups=${num_tables}" \
                        "inter_arrival.scale=${load}" \
                        "duration_ms=${QUICK_DURATION}"
                else
                    create_config_variant "$EXP2_2_CONFIG" "$TEMP_CONFIG" \
                        "seed=${SEED}" \
                        "num_tables=${num_tables}" \
                        "num_groups=${num_tables}" \
                        "inter_arrival.scale=${load}"
                fi

                wait_for_job_slot
                DESC="$PROGRESS Exp2.2 tables=$num_tables load=$load seed=$SEED"
                log "  Starting: $DESC"
                run_experiment_background "$EXP2_2_CONFIG" "$TEMP_CONFIG" "$DESC"
            done
        done
    done

    log "All Experiment 2.2 jobs submitted"
fi

# ============================================================================
# Experiment 3.1: Single Table with Real Conflicts
# ============================================================================

if [ "$RUN_EXP3_1" = true ]; then
    log_section "EXPERIMENT 3.1: Single Table with Real Conflicts"
    log "Research Question 1b: How do real conflicts shift saturation?"
    log ""
    log "Sweeping real_conflict_probability: ${EXP3_1_REAL_PROBS[*]}"
    log "Sweeping inter_arrival.scale: ${EXP3_1_LOADS[*]}"
    log ""

    for prob in "${EXP3_1_REAL_PROBS[@]}"; do
        for load in "${EXP3_1_LOADS[@]}"; do
            for seed_num in $(seq 1 $NUM_SEEDS); do
                CURRENT_RUN=$((CURRENT_RUN + 1))
                PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

                # Generate deterministic seed
                SEED=$(generate_deterministic_seed "exp3_1" "prob=${prob}:load=${load}" "$seed_num")

                TEMP_CONFIG=$(mktemp)

                if [ "$QUICK_MODE" = true ]; then
                    create_config_variant "$EXP3_1_CONFIG" "$TEMP_CONFIG" \
                        "seed=${SEED}" \
                        "real_conflict_probability=${prob}" \
                        "inter_arrival.scale=${load}" \
                        "duration_ms=${QUICK_DURATION}"
                else
                    create_config_variant "$EXP3_1_CONFIG" "$TEMP_CONFIG" \
                        "seed=${SEED}" \
                        "real_conflict_probability=${prob}" \
                        "inter_arrival.scale=${load}"
                fi

                wait_for_job_slot
                DESC="$PROGRESS Exp3.1 prob=$prob load=$load seed=$SEED"
                log "  Starting: $DESC"
                run_experiment_background "$EXP3_1_CONFIG" "$TEMP_CONFIG" "$DESC"
            done
        done
    done

    log "All Experiment 3.1 jobs submitted"
fi

# ============================================================================
# Experiment 3.2: Manifest Count Distribution
# ============================================================================

if [ "$RUN_EXP3_2" = true ]; then
    log_section "EXPERIMENT 3.2: Manifest Count Distribution"
    log "Research Question: How does manifest count variance affect performance?"
    log ""
    log "Sweeping conflicting_manifests distribution: ${EXP3_2_DISTS[*]}"
    log "Sweeping inter_arrival.scale: ${EXP3_2_LOADS[*]}"
    log ""

    for dist_config in "${EXP3_2_DISTS[@]}"; do
        IFS=':' read -r dist value <<< "$dist_config"

        for load in "${EXP3_2_LOADS[@]}"; do
            for seed_num in $(seq 1 $NUM_SEEDS); do
                CURRENT_RUN=$((CURRENT_RUN + 1))
                PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

                # Generate deterministic seed
                SEED=$(generate_deterministic_seed "exp3_2" "dist=${dist_config}:load=${load}" "$seed_num")

                TEMP_CONFIG=$(mktemp)

                if [ "$dist" = "fixed" ]; then
                    if [ "$QUICK_MODE" = true ]; then
                        create_config_variant "$EXP3_2_CONFIG" "$TEMP_CONFIG" \
                            "seed=${SEED}" \
                            "conflicting_manifests.distribution=\"${dist}\"" \
                            "conflicting_manifests.value=${value}" \
                            "inter_arrival.scale=${load}" \
                            "duration_ms=${QUICK_DURATION}"
                    else
                        create_config_variant "$EXP3_2_CONFIG" "$TEMP_CONFIG" \
                            "seed=${SEED}" \
                            "conflicting_manifests.distribution=\"${dist}\"" \
                            "conflicting_manifests.value=${value}" \
                            "inter_arrival.scale=${load}"
                    fi
                else  # exponential
                    if [ "$QUICK_MODE" = true ]; then
                        create_config_variant "$EXP3_2_CONFIG" "$TEMP_CONFIG" \
                            "seed=${SEED}" \
                            "conflicting_manifests.distribution=\"${dist}\"" \
                            "conflicting_manifests.mean=${value}" \
                            "inter_arrival.scale=${load}" \
                            "duration_ms=${QUICK_DURATION}"
                    else
                        create_config_variant "$EXP3_2_CONFIG" "$TEMP_CONFIG" \
                            "seed=${SEED}" \
                            "conflicting_manifests.distribution=\"${dist}\"" \
                            "conflicting_manifests.mean=${value}" \
                            "inter_arrival.scale=${load}"
                    fi
                fi

                wait_for_job_slot
                DESC="$PROGRESS Exp3.2 dist=$dist_config load=$load seed=$SEED"
                log "  Starting: $DESC"
                run_experiment_background "$EXP3_2_CONFIG" "$TEMP_CONFIG" "$DESC"
            done
        done
    done

    log "All Experiment 3.2 jobs submitted"
fi

# ============================================================================
# Experiment 3.3: Multi-Table with Real Conflicts
# ============================================================================

if [ "$RUN_EXP3_3" = true ]; then
    log_section "EXPERIMENT 3.3: Multi-Table with Real Conflicts"
    log "Research Question 2b: How do real conflicts compound in multi-table txns?"
    log ""
    log "Sweeping num_tables: ${EXP3_3_TABLES[*]}"
    log "Sweeping real_conflict_probability: ${EXP3_3_REAL_PROBS[*]}"
    log "Sweeping inter_arrival.scale: ${EXP3_3_LOADS[*]}"
    log ""

    for num_tables in "${EXP3_3_TABLES[@]}"; do
        for prob in "${EXP3_3_REAL_PROBS[@]}"; do
            for load in "${EXP3_3_LOADS[@]}"; do
                for seed_num in $(seq 1 $NUM_SEEDS); do
                    CURRENT_RUN=$((CURRENT_RUN + 1))
                    PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

                    # Generate deterministic seed
                    SEED=$(generate_deterministic_seed "exp3_3" "tables=${num_tables}:prob=${prob}:load=${load}" "$seed_num")

                    TEMP_CONFIG=$(mktemp)

                    if [ "$QUICK_MODE" = true ]; then
                        create_config_variant "$EXP3_3_CONFIG" "$TEMP_CONFIG" \
                            "seed=${SEED}" \
                            "num_tables=${num_tables}" \
                            "num_groups=${num_tables}" \
                            "real_conflict_probability=${prob}" \
                            "inter_arrival.scale=${load}" \
                            "duration_ms=${QUICK_DURATION}"
                    else
                        create_config_variant "$EXP3_3_CONFIG" "$TEMP_CONFIG" \
                            "seed=${SEED}" \
                            "num_tables=${num_tables}" \
                            "num_groups=${num_tables}" \
                            "real_conflict_probability=${prob}" \
                            "inter_arrival.scale=${load}"
                    fi

                    wait_for_job_slot
                    DESC="$PROGRESS Exp3.3 tables=$num_tables prob=$prob load=$load seed=$SEED"
                    log "  Starting: $DESC"
                    run_experiment_background "$EXP3_3_CONFIG" "$TEMP_CONFIG" "$DESC"
                done
            done
        done
    done

    log "All Experiment 3.3 jobs submitted"
fi

# ============================================================================
# Experiment 3.4: Multi-table real conflicts with exponential backoff
# ============================================================================

if [ "$RUN_EXP3_4" = true ]; then
    log_section "EXPERIMENT 3.4: Multi-Table Real Conflicts with Backoff"
    log "Research Question 4b: Does backoff help with multi-table real conflicts?"
    log ""
    log "Sweeping num_tables: ${EXP3_4_TABLES[*]}"
    log "Sweeping real_conflict_probability: ${EXP3_4_REAL_PROBS[*]}"
    log "Sweeping inter_arrival.scale: ${EXP3_4_LOADS[*]}"
    log ""

    for num_tables in "${EXP3_4_TABLES[@]}"; do
        for prob in "${EXP3_4_REAL_PROBS[@]}"; do
            for load in "${EXP3_4_LOADS[@]}"; do
                for seed_num in $(seq 1 $NUM_SEEDS); do
                    CURRENT_RUN=$((CURRENT_RUN + 1))
                    PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

                    # Generate deterministic seed
                    SEED=$(generate_deterministic_seed "exp3_4" "tables=${num_tables}:prob=${prob}:load=${load}" "$seed_num")

                    TEMP_CONFIG=$(mktemp)

                    if [ "$QUICK_MODE" = true ]; then
                        create_config_variant "$EXP3_4_CONFIG" "$TEMP_CONFIG" \
                            "seed=${SEED}" \
                            "num_tables=${num_tables}" \
                            "num_groups=${num_tables}" \
                            "real_conflict_probability=${prob}" \
                            "inter_arrival.scale=${load}" \
                            "duration_ms=${QUICK_DURATION}"
                    else
                        create_config_variant "$EXP3_4_CONFIG" "$TEMP_CONFIG" \
                            "seed=${SEED}" \
                            "num_tables=${num_tables}" \
                            "num_groups=${num_tables}" \
                            "real_conflict_probability=${prob}" \
                            "inter_arrival.scale=${load}"
                    fi

                    wait_for_job_slot
                    DESC="$PROGRESS Exp3.4 tables=$num_tables prob=$prob load=$load seed=$SEED"
                    log "  Starting: $DESC"
                    run_experiment_background "$EXP3_4_CONFIG" "$TEMP_CONFIG" "$DESC"
                done
            done
        done
    done

    log "All Experiment 3.4 jobs submitted"
fi

# ============================================================================
# Experiment 4.1: Single-table false conflicts with exponential backoff
# ============================================================================

if [ "$RUN_EXP4_1" = true ]; then
    log_section "EXPERIMENT 4.1: Single-Table False Conflicts with Backoff"
    log "Research Question 4a: Does backoff improve performance under contention?"
    log ""
    log "Sweeping inter_arrival.scale: ${EXP4_1_LOADS[*]}"
    log ""

    for load in "${EXP4_1_LOADS[@]}"; do
        for seed_num in $(seq 1 $NUM_SEEDS); do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

            # Generate deterministic seed
            SEED=$(generate_deterministic_seed "exp4_1" "load=${load}" "$seed_num")

            TEMP_CONFIG=$(mktemp)

            if [ "$QUICK_MODE" = true ]; then
                create_config_variant "$EXP4_1_CONFIG" "$TEMP_CONFIG" \
                    "seed=${SEED}" \
                    "inter_arrival.scale=${load}" \
                    "duration_ms=${QUICK_DURATION}"
            else
                create_config_variant "$EXP4_1_CONFIG" "$TEMP_CONFIG" \
                    "seed=${SEED}" \
                    "inter_arrival.scale=${load}"
            fi

            wait_for_job_slot
            DESC="$PROGRESS Exp4.1 load=$load seed=$SEED"
            log "  Starting: $DESC"
            run_experiment_background "$EXP4_1_CONFIG" "$TEMP_CONFIG" "$DESC"
        done
    done

    log "All Experiment 4.1 jobs submitted"
fi

# ============================================================================
# Experiment 5.1: Single Table with Varying Catalog Latency - False Conflicts
# ============================================================================

if [ "$RUN_EXP5_1" = true ]; then
    log_section "EXPERIMENT 5.1: Single Table with Varying Catalog Latency"
    log "Research Question: How does catalog latency affect throughput ceiling?"
    log ""
    log "Sweeping T_CAS.mean: ${EXP5_1_CAS_LATENCIES[*]} ms"
    log "Sweeping inter_arrival.scale: ${EXP5_1_LOADS[*]}"
    log ""

    for cas_latency in "${EXP5_1_CAS_LATENCIES[@]}"; do
        cas_stddev=$((cas_latency / 10))  # 10% stddev
        for load in "${EXP5_1_LOADS[@]}"; do
            for seed_num in $(seq 1 $NUM_SEEDS); do
                CURRENT_RUN=$((CURRENT_RUN + 1))
                PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

                # Generate deterministic seed
                SEED=$(generate_deterministic_seed "exp5_1" "cas=${cas_latency}:load=${load}" "$seed_num")

                TEMP_CONFIG=$(mktemp)

                if [ "$QUICK_MODE" = true ]; then
                    create_config_variant "$EXP5_1_CONFIG" "$TEMP_CONFIG" \
                        "seed=${SEED}" \
                        "T_CAS.mean=${cas_latency}" \
                        "T_CAS.stddev=${cas_stddev}" \
                        "T_METADATA_ROOT.read.mean=${cas_latency}" \
                        "T_METADATA_ROOT.read.stddev=${cas_stddev}" \
                        "T_METADATA_ROOT.write.mean=${cas_latency}" \
                        "T_METADATA_ROOT.write.stddev=${cas_stddev}" \
                        "inter_arrival.scale=${load}" \
                        "duration_ms=${QUICK_DURATION}"
                else
                    create_config_variant "$EXP5_1_CONFIG" "$TEMP_CONFIG" \
                        "seed=${SEED}" \
                        "T_CAS.mean=${cas_latency}" \
                        "T_CAS.stddev=${cas_stddev}" \
                        "T_METADATA_ROOT.read.mean=${cas_latency}" \
                        "T_METADATA_ROOT.read.stddev=${cas_stddev}" \
                        "T_METADATA_ROOT.write.mean=${cas_latency}" \
                        "T_METADATA_ROOT.write.stddev=${cas_stddev}" \
                        "inter_arrival.scale=${load}"
                fi

                wait_for_job_slot
                DESC="$PROGRESS Exp5.1 cas=${cas_latency}ms load=$load seed=$SEED"
                log "  Starting: $DESC"
                run_experiment_background "$EXP5_1_CONFIG" "$TEMP_CONFIG" "$DESC"
            done
        done
    done

    log "All Experiment 5.1 jobs submitted"
fi

# ============================================================================
# Experiment 5.2: Multi-Table with Varying Catalog Latency - False Conflicts
# ============================================================================

if [ "$RUN_EXP5_2" = true ]; then
    log_section "EXPERIMENT 5.2: Multi-Table with Varying Catalog Latency"
    log "Research Question: Does catalog latency change multi-table scaling?"
    log ""
    log "Sweeping T_CAS.mean: ${EXP5_2_CAS_LATENCIES[*]} ms"
    log "Sweeping num_tables: ${EXP5_2_TABLES[*]}"
    log "Sweeping inter_arrival.scale: ${EXP5_2_LOADS[*]}"
    log ""

    for cas_latency in "${EXP5_2_CAS_LATENCIES[@]}"; do
        cas_stddev=$((cas_latency / 10))  # 10% stddev
        for num_tables in "${EXP5_2_TABLES[@]}"; do
            for load in "${EXP5_2_LOADS[@]}"; do
                for seed_num in $(seq 1 $NUM_SEEDS); do
                    CURRENT_RUN=$((CURRENT_RUN + 1))
                    PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

                    # Generate deterministic seed
                    SEED=$(generate_deterministic_seed "exp5_2" "cas=${cas_latency}:tables=${num_tables}:load=${load}" "$seed_num")

                    TEMP_CONFIG=$(mktemp)

                    if [ "$QUICK_MODE" = true ]; then
                        create_config_variant "$EXP5_2_CONFIG" "$TEMP_CONFIG" \
                            "seed=${SEED}" \
                            "num_tables=${num_tables}" \
                            "num_groups=${num_tables}" \
                            "T_CAS.mean=${cas_latency}" \
                            "T_CAS.stddev=${cas_stddev}" \
                            "T_METADATA_ROOT.read.mean=${cas_latency}" \
                            "T_METADATA_ROOT.read.stddev=${cas_stddev}" \
                            "T_METADATA_ROOT.write.mean=${cas_latency}" \
                            "T_METADATA_ROOT.write.stddev=${cas_stddev}" \
                            "inter_arrival.scale=${load}" \
                            "duration_ms=${QUICK_DURATION}"
                    else
                        create_config_variant "$EXP5_2_CONFIG" "$TEMP_CONFIG" \
                            "seed=${SEED}" \
                            "num_tables=${num_tables}" \
                            "num_groups=${num_tables}" \
                            "T_CAS.mean=${cas_latency}" \
                            "T_CAS.stddev=${cas_stddev}" \
                            "T_METADATA_ROOT.read.mean=${cas_latency}" \
                            "T_METADATA_ROOT.read.stddev=${cas_stddev}" \
                            "T_METADATA_ROOT.write.mean=${cas_latency}" \
                            "T_METADATA_ROOT.write.stddev=${cas_stddev}" \
                            "inter_arrival.scale=${load}"
                    fi

                    wait_for_job_slot
                    DESC="$PROGRESS Exp5.2 cas=${cas_latency}ms tables=$num_tables load=$load seed=$SEED"
                    log "  Starting: $DESC"
                    run_experiment_background "$EXP5_2_CONFIG" "$TEMP_CONFIG" "$DESC"
                done
            done
        done
    done

    log "All Experiment 5.2 jobs submitted"
fi

# ============================================================================
# Experiment 5.3: Transaction Partitioning with Varying Catalog Latency
# ============================================================================

if [ "$RUN_EXP5_3" = true ]; then
    log_section "EXPERIMENT 5.3: Transaction Partitioning with Varying Catalog Latency"
    log "Research Question: How does transaction partitioning interact with catalog latency?"
    log ""
    log "Sweeping T_CAS.mean: ${EXP5_3_CAS_LATENCIES[*]} ms"
    log "Sweeping num_groups: ${EXP5_3_GROUPS[*]}"
    log "Sweeping inter_arrival.scale: ${EXP5_3_LOADS[*]}"
    log ""

    for cas_latency in "${EXP5_3_CAS_LATENCIES[@]}"; do
        cas_stddev=$((cas_latency / 10))  # 10% stddev
        for num_groups in "${EXP5_3_GROUPS[@]}"; do
            for load in "${EXP5_3_LOADS[@]}"; do
                for seed_num in $(seq 1 $NUM_SEEDS); do
                    CURRENT_RUN=$((CURRENT_RUN + 1))
                    PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

                    # Generate deterministic seed
                    SEED=$(generate_deterministic_seed "exp5_3" "cas=${cas_latency}:groups=${num_groups}:load=${load}" "$seed_num")

                    TEMP_CONFIG=$(mktemp)

                    if [ "$QUICK_MODE" = true ]; then
                        create_config_variant "$EXP5_3_CONFIG" "$TEMP_CONFIG" \
                            "seed=${SEED}" \
                            "num_groups=${num_groups}" \
                            "T_CAS.mean=${cas_latency}" \
                            "T_CAS.stddev=${cas_stddev}" \
                            "T_METADATA_ROOT.read.mean=${cas_latency}" \
                            "T_METADATA_ROOT.read.stddev=${cas_stddev}" \
                            "T_METADATA_ROOT.write.mean=${cas_latency}" \
                            "T_METADATA_ROOT.write.stddev=${cas_stddev}" \
                            "inter_arrival.scale=${load}" \
                            "duration_ms=${QUICK_DURATION}"
                    else
                        create_config_variant "$EXP5_3_CONFIG" "$TEMP_CONFIG" \
                            "seed=${SEED}" \
                            "num_groups=${num_groups}" \
                            "T_CAS.mean=${cas_latency}" \
                            "T_CAS.stddev=${cas_stddev}" \
                            "T_METADATA_ROOT.read.mean=${cas_latency}" \
                            "T_METADATA_ROOT.read.stddev=${cas_stddev}" \
                            "T_METADATA_ROOT.write.mean=${cas_latency}" \
                            "T_METADATA_ROOT.write.stddev=${cas_stddev}" \
                            "inter_arrival.scale=${load}"
                    fi

                    wait_for_job_slot
                    DESC="$PROGRESS Exp5.3 cas=${cas_latency}ms groups=$num_groups load=$load seed=$SEED"
                    log "  Starting: $DESC"
                    run_experiment_background "$EXP5_3_CONFIG" "$TEMP_CONFIG" "$DESC"
                done
            done
        done
    done

    log "All Experiment 5.3 jobs submitted"
fi

# ============================================================================
# Wait for all jobs to complete
# ============================================================================

wait_for_all_jobs

# ============================================================================
# Final Summary
# ============================================================================

log_section "EXPERIMENTS COMPLETE"
log "Total runs: $TOTAL_RUNS"
log "  Successful: $JOB_SUCCESS"
log "  Failed: $JOB_FAILED"
log ""
log "Results stored in: experiments/"

if [ "$RUN_EXP2_1" = true ]; then
    log "  exp2_1_single_table_false-*/"
fi
if [ "$RUN_EXP2_2" = true ]; then
    log "  exp2_2_multi_table_false-*/"
fi
if [ "$RUN_EXP3_1" = true ]; then
    log "  exp3_1_single_table_real-*/"
fi
if [ "$RUN_EXP3_2" = true ]; then
    log "  exp3_2_manifest_count_distribution-*/"
fi
if [ "$RUN_EXP3_3" = true ]; then
    log "  exp3_3_multi_table_real-*/"
fi
if [ "$RUN_EXP5_1" = true ]; then
    log "  exp5_1_single_table_catalog_latency-*/"
fi
if [ "$RUN_EXP5_2" = true ]; then
    log "  exp5_2_multi_table_catalog_latency-*/"
fi
if [ "$RUN_EXP5_3" = true ]; then
    log "  exp5_3_transaction_partitioning_catalog_latency-*/"
fi

log ""
log "Next steps:"
log "  1. Verify results: find experiments/ -name 'results.parquet' | wc -l"
log "  2. Check for incomplete runs: find experiments/ -name '.running.parquet'"
log "  3. Run analysis:"
if [ "$RUN_EXP2_1" = true ]; then
    log "     python -m endive.saturation_analysis -i experiments -p 'exp2_1_*' -o plots/exp2_1"
fi
if [ "$RUN_EXP2_2" = true ]; then
    log "     python -m endive.saturation_analysis -i experiments -p 'exp2_2_*' -o plots/exp2_2 --group-by num_tables"
fi
if [ "$RUN_EXP3_1" = true ]; then
    log "     python -m endive.saturation_analysis -i experiments -p 'exp3_1_*' -o plots/exp3_1 --group-by real_conflict_probability"
fi
if [ "$RUN_EXP3_3" = true ]; then
    log "     python -m endive.saturation_analysis -i experiments -p 'exp3_3_*' -o plots/exp3_3 --group-by num_tables,real_conflict_probability"
fi

if [ "$DRY_RUN" = true ]; then
    log ""
    log "DRY RUN COMPLETE - No experiments were actually executed"
fi

exit 0
