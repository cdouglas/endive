#!/bin/bash
# run_exp8_experiments.sh
#
# Parallelized driver for Experiment 8: Metadata inlining and ML+ append optimizations
#
# This script runs saturation analysis for:
# - exp8_0: Baseline (inlined metadata, rewrite manifest lists)
# - exp8_1/8_2: Metadata inlining disabled
# - exp8_3/8_4: Manifest list append mode (ML+)
# - exp8_5/8_6: Combined optimizations
#
# Each experiment is run on both S3 Express One Zone (s3x) and Azure Premium (azurex).
#
# Usage:
#   ./run_exp8_experiments.sh [OPTIONS]
#
# Options:
#   --seeds N          Number of seeds per configuration (default: 3)
#   --parallel N       Number of concurrent experiments (default: 112)
#   --s3x-only         Run only S3 Express One Zone experiments
#   --azure-only       Run only Azure Premium experiments
#   --baseline         Run only baseline experiments (exp8_0)
#   --metadata         Run only metadata inlining experiments (exp8_1/8_2)
#   --mlplus           Run only ML+ experiments (exp8_3/8_4)
#   --combined         Run only combined experiments (exp8_5/8_6)
#   --quick            Quick test mode (fewer configs, shorter duration)
#   --dry-run          Show what would be run without executing
#   --help, -h         Show this help message
#
# Examples:
#   # Run all exp8 experiments with 112 parallel jobs
#   ./run_exp8_experiments.sh --parallel 112 --seeds 5
#
#   # Run only S3x experiments for quick validation
#   ./run_exp8_experiments.sh --s3x-only --quick --seeds 1
#
#   # Run in background on 144-core machine
#   nohup ./run_exp8_experiments.sh --parallel 112 --seeds 5 > exp8.log 2>&1 &

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Default values
NUM_SEEDS=3
NUM_PARALLEL=112  # Default for 144-core machine, leaving headroom for system
QUICK_MODE=false
DRY_RUN=false

# Provider selection (if none specified, run all append-supporting providers)
RUN_S3X=false
RUN_AZURE=false
RUN_AZUREX=false
ANY_PROVIDER_SPECIFIED=false

# Experiment selection (if none specified, run all)
RUN_BASELINE=false
RUN_METADATA=false
RUN_MLPLUS=false
RUN_COMBINED=false
ANY_EXP_SPECIFIED=false

# Load levels for saturation analysis (inter_arrival.scale in ms)
# Lower values = higher load = more transactions per unit time
EXP8_LOADS=(10 20 50 100 200 500 1000 2000 5000)

# Quick mode parameters
QUICK_LOADS=(100 500 2000)
QUICK_DURATION=10000  # 10 seconds

# Base config files for all append-supporting providers
declare -A CONFIGS
# S3 Express One Zone
CONFIGS["baseline_s3x"]="experiment_configs/exp8_0_baseline_s3x.toml"
CONFIGS["metadata_s3x"]="experiment_configs/exp8_1_metadata_inlining_s3x.toml"
CONFIGS["mlplus_s3x"]="experiment_configs/exp8_3_ml_append_s3x.toml"
CONFIGS["combined_s3x"]="experiment_configs/exp8_5_combined_s3x.toml"
# Azure Premium Block Blob
CONFIGS["baseline_azurex"]="experiment_configs/exp8_0_baseline_azurex.toml"
CONFIGS["metadata_azurex"]="experiment_configs/exp8_2_metadata_inlining_azurex.toml"
CONFIGS["mlplus_azurex"]="experiment_configs/exp8_4_ml_append_azurex.toml"
CONFIGS["combined_azurex"]="experiment_configs/exp8_6_combined_azurex.toml"
# Azure Blob Storage (Standard)
CONFIGS["baseline_azure"]="experiment_configs/exp8_0_baseline_azure.toml"
CONFIGS["metadata_azure"]="experiment_configs/exp8_7_metadata_inlining_azure.toml"
CONFIGS["mlplus_azure"]="experiment_configs/exp8_8_ml_append_azure.toml"
CONFIGS["combined_azure"]="experiment_configs/exp8_9_combined_azure.toml"

# Logging
LOG_DIR="experiment_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/exp8_${TIMESTAMP}.log"

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
        --s3x-only)
            RUN_S3X=true
            ANY_PROVIDER_SPECIFIED=true
            shift
            ;;
        --azure-only)
            RUN_AZURE=true
            ANY_PROVIDER_SPECIFIED=true
            shift
            ;;
        --azurex-only)
            RUN_AZUREX=true
            ANY_PROVIDER_SPECIFIED=true
            shift
            ;;
        --baseline)
            RUN_BASELINE=true
            ANY_EXP_SPECIFIED=true
            shift
            ;;
        --metadata)
            RUN_METADATA=true
            ANY_EXP_SPECIFIED=true
            shift
            ;;
        --mlplus)
            RUN_MLPLUS=true
            ANY_EXP_SPECIFIED=true
            shift
            ;;
        --combined)
            RUN_COMBINED=true
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
            head -n 40 "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If no provider specified, run all append-supporting providers
if [ "$ANY_PROVIDER_SPECIFIED" = false ]; then
    RUN_S3X=true
    RUN_AZURE=true
    RUN_AZUREX=true
fi

# If no experiment type specified, run all
if [ "$ANY_EXP_SPECIFIED" = false ]; then
    RUN_BASELINE=true
    RUN_METADATA=true
    RUN_MLPLUS=true
    RUN_COMBINED=true
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
    if [ -f "bin/activate" ]; then
        source bin/activate
    elif [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "Warning: Virtual environment not found. Assuming endive is installed globally."
    fi
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
    local nonce_file="experiments/.nonce_exp8"

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

if [ "$QUICK_MODE" = true ]; then
    log "Running in QUICK MODE (reduced parameters for testing)"
    EXP8_LOADS=("${QUICK_LOADS[@]}")
fi

# Count providers
NUM_PROVIDERS=0
if [ "$RUN_S3X" = true ]; then
    NUM_PROVIDERS=$((NUM_PROVIDERS + 1))
fi
if [ "$RUN_AZURE" = true ]; then
    NUM_PROVIDERS=$((NUM_PROVIDERS + 1))
fi
if [ "$RUN_AZUREX" = true ]; then
    NUM_PROVIDERS=$((NUM_PROVIDERS + 1))
fi

# Count experiment types
NUM_EXP_TYPES=0
if [ "$RUN_BASELINE" = true ]; then
    NUM_EXP_TYPES=$((NUM_EXP_TYPES + 1))
fi
if [ "$RUN_METADATA" = true ]; then
    NUM_EXP_TYPES=$((NUM_EXP_TYPES + 1))
fi
if [ "$RUN_MLPLUS" = true ]; then
    NUM_EXP_TYPES=$((NUM_EXP_TYPES + 1))
fi
if [ "$RUN_COMBINED" = true ]; then
    NUM_EXP_TYPES=$((NUM_EXP_TYPES + 1))
fi

# Total = providers × experiment_types × load_levels × seeds
TOTAL_RUNS=$((NUM_PROVIDERS * NUM_EXP_TYPES * ${#EXP8_LOADS[@]} * NUM_SEEDS))

# ============================================================================
# Print Summary
# ============================================================================

log_section "EXPERIMENT 8: METADATA INLINING AND ML+ APPEND"
log "Configuration:"
log "  Number of seeds per config: $NUM_SEEDS"
log "  Parallel jobs: $NUM_PARALLEL"
log "  Quick mode: $QUICK_MODE"
log "  Dry run: $DRY_RUN"
log ""
log "Providers selected:"
log "  S3 Express One Zone (s3x): $RUN_S3X"
log "  Azure Blob Standard (azure): $RUN_AZURE"
log "  Azure Premium Block Blob (azurex): $RUN_AZUREX"
log ""
log "Experiment types selected:"
log "  Baseline (exp8_0): $RUN_BASELINE"
log "  Metadata inlining disabled (exp8_1/8_2): $RUN_METADATA"
log "  ML+ manifest list append (exp8_3/8_4): $RUN_MLPLUS"
log "  Combined optimizations (exp8_5/8_6): $RUN_COMBINED"
log ""
log "Load levels (inter_arrival.scale): ${EXP8_LOADS[*]}"
log ""
log "Total simulations to run: $TOTAL_RUNS"

if [ "$QUICK_MODE" = true ]; then
    AVG_TIME=15  # 15 seconds per quick run
else
    AVG_TIME=4  # ~4 seconds per 1-hour simulation
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
# Run experiments function
# ============================================================================

run_experiment_sweep() {
    local exp_type=$1  # baseline, metadata, mlplus, combined
    local provider=$2  # s3x, azurex
    local config_key="${exp_type}_${provider}"
    local base_config="${CONFIGS[$config_key]}"

    if [ ! -f "$base_config" ]; then
        log "Warning: Config file not found: $base_config"
        return
    fi

    # Extract experiment label from config
    local exp_label=$(grep 'label\s*=' "$base_config" | sed 's/.*=\s*"\([^"]*\)".*/\1/')

    log ""
    log "Running: $exp_type on $provider (${exp_label})"
    log "  Config: $base_config"
    log "  Load levels: ${EXP8_LOADS[*]}"

    for load in "${EXP8_LOADS[@]}"; do
        for seed_num in $(seq 1 $NUM_SEEDS); do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

            # Generate deterministic seed
            SEED=$(generate_deterministic_seed "$exp_label" "load=${load}" "$seed_num")

            TEMP_CONFIG=$(mktemp)

            if [ "$QUICK_MODE" = true ]; then
                create_config_variant "$base_config" "$TEMP_CONFIG" \
                    "seed=${SEED}" \
                    "inter_arrival.scale=${load}" \
                    "duration_ms=${QUICK_DURATION}"
            else
                create_config_variant "$base_config" "$TEMP_CONFIG" \
                    "seed=${SEED}" \
                    "inter_arrival.scale=${load}"
            fi

            wait_for_job_slot
            DESC="$PROGRESS ${exp_label} load=$load seed=$SEED"
            log "  Starting: $DESC"
            run_experiment_background "$base_config" "$TEMP_CONFIG" "$DESC"
        done
    done
}

# ============================================================================
# Run all selected experiments
# ============================================================================

# Baseline experiments (exp8_0)
if [ "$RUN_BASELINE" = true ]; then
    log_section "BASELINE EXPERIMENTS (exp8_0)"
    log "Reference configuration: inlined metadata + manifest list rewrite"

    if [ "$RUN_S3X" = true ]; then
        run_experiment_sweep "baseline" "s3x"
    fi

    if [ "$RUN_AZURE" = true ]; then
        run_experiment_sweep "baseline" "azure"
    fi

    if [ "$RUN_AZUREX" = true ]; then
        run_experiment_sweep "baseline" "azurex"
    fi
fi

# Metadata inlining disabled experiments (exp8_1/8_2)
if [ "$RUN_METADATA" = true ]; then
    log_section "METADATA INLINING EXPERIMENTS (exp8_1/8_2)"
    log "Configuration: table_metadata_inlined=false (separate metadata files)"

    if [ "$RUN_S3X" = true ]; then
        run_experiment_sweep "metadata" "s3x"
    fi

    if [ "$RUN_AZURE" = true ]; then
        run_experiment_sweep "metadata" "azure"
    fi

    if [ "$RUN_AZUREX" = true ]; then
        run_experiment_sweep "metadata" "azurex"
    fi
fi

# ML+ manifest list append experiments (exp8_3/8_4)
if [ "$RUN_MLPLUS" = true ]; then
    log_section "ML+ MANIFEST LIST APPEND EXPERIMENTS (exp8_3/8_4)"
    log "Configuration: manifest_list_mode=\"append\" (ML+ mode)"

    if [ "$RUN_S3X" = true ]; then
        run_experiment_sweep "mlplus" "s3x"
    fi

    if [ "$RUN_AZURE" = true ]; then
        run_experiment_sweep "mlplus" "azure"
    fi

    if [ "$RUN_AZUREX" = true ]; then
        run_experiment_sweep "mlplus" "azurex"
    fi
fi

# Combined optimizations experiments (exp8_5/8_6)
if [ "$RUN_COMBINED" = true ]; then
    log_section "COMBINED OPTIMIZATION EXPERIMENTS (exp8_5/8_6)"
    log "Configuration: table_metadata_inlined=false + manifest_list_mode=\"append\""

    if [ "$RUN_S3X" = true ]; then
        run_experiment_sweep "combined" "s3x"
    fi

    if [ "$RUN_AZURE" = true ]; then
        run_experiment_sweep "combined" "azure"
    fi

    if [ "$RUN_AZUREX" = true ]; then
        run_experiment_sweep "combined" "azurex"
    fi
fi

# ============================================================================
# Wait for all jobs to complete
# ============================================================================

wait_for_all_jobs

# ============================================================================
# Final Summary
# ============================================================================

log_section "EXPERIMENT 8 COMPLETE"
log "Total runs: $TOTAL_RUNS"
log "  Successful: $JOB_SUCCESS"
log "  Failed: $JOB_FAILED"
log ""
log "Results stored in: experiments/"
log "  exp8_0_baseline_*/"
log "  exp8_1_metadata_inlining_s3x*/"
log "  exp8_2_metadata_inlining_azurex*/"
log "  exp8_3_ml_append_s3x*/"
log "  exp8_4_ml_append_azurex*/"
log "  exp8_5_combined_s3x*/"
log "  exp8_6_combined_azurex*/"
log ""
log "Next steps:"
log "  1. Verify results: find experiments/ -name 'results.parquet' -path '*exp8*' | wc -l"
log "  2. Check for incomplete runs: find experiments/ -name '.running.parquet' -path '*exp8*'"
log "  3. Consolidate results: python scripts/consolidate_all_experiments_incremental.py"
log "  4. Run analysis:"
log ""
log "     # S3 Express comparisons"
log "     python -m endive.saturation_analysis -i experiments -p 'exp8_*_s3x*' -o plots/exp8_s3x --group-by manifest_list_mode,table_metadata_inlined"
log ""
log "     # Azure Premium comparisons"
log "     python -m endive.saturation_analysis -i experiments -p 'exp8_*_azurex*' -o plots/exp8_azurex --group-by manifest_list_mode,table_metadata_inlined"
log ""
log "     # Cross-provider comparison"
log "     python -m endive.saturation_analysis -i experiments -p 'exp8_0_baseline*' -o plots/exp8_baseline_comparison --group-by storage_provider"

if [ "$DRY_RUN" = true ]; then
    log ""
    log "DRY RUN COMPLETE - No experiments were actually executed"
fi

exit 0
