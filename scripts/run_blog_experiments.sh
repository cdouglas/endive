#!/bin/bash
#
# Run blog post experiments: "Impossibly fast catalog" with real storage
#
# Usage:
#   ./scripts/run_blog_experiments.sh [--seeds N] [--parallel N]
#
# Default: 5 seeds, 4 parallel jobs

set -e

# Defaults
NUM_SEEDS=5
NUM_PARALLEL=4

# Parse arguments
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--seeds N] [--parallel N]"
            exit 1
            ;;
    esac
done

# Activate virtual environment if exists
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f bin/activate ]; then
    source bin/activate
fi

echo "=============================================="
echo "Blog Post Experiments: Instant Catalog + S3 Storage"
echo "=============================================="
echo "Seeds per config: $NUM_SEEDS"
echo "Parallel jobs: $NUM_PARALLEL"
echo ""

# Create temp directory for config variants
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Function to create config variant
create_config() {
    local base_config="$1"
    local output="$2"
    shift 2

    cp "$base_config" "$output"

    while [[ $# -gt 0 ]]; do
        local key="$1"
        local value="$2"
        shift 2

        # Handle different key formats
        case "$key" in
            inter_arrival.scale)
                sed -i "s/^inter_arrival.scale = .*/inter_arrival.scale = $value/" "$output"
                ;;
            real_conflict_probability)
                sed -i "s/^real_conflict_probability = .*/real_conflict_probability = $value/" "$output"
                ;;
            num_tables)
                sed -i "s/^num_tables = .*/num_tables = $value/" "$output"
                ;;
            seed)
                if grep -q "^seed = " "$output"; then
                    sed -i "s/^seed = .*/seed = $value/" "$output"
                else
                    sed -i "/^\[simulation\]/a seed = $value" "$output"
                fi
                ;;
        esac
    done
}

# Function to run experiment
run_experiment() {
    local config="$1"
    python -m endive.main "$config" --yes
}

export -f run_experiment

# Generate seeds
SEEDS=()
for i in $(seq 1 $NUM_SEEDS); do
    SEEDS+=($RANDOM$RANDOM)
done

echo "Seeds: ${SEEDS[*]}"
echo ""

# ============================================================================
# Experiment 1: Single Table, Trivial Conflicts
# ============================================================================
echo "=== Experiment 1: Single Table, Trivial Conflicts ==="
SCALES=(10 20 50 100 200 500 1000 2000 5000)

configs_1=()
for scale in "${SCALES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        config="$TEMP_DIR/blog_1tbl_trivial_s${scale}_seed${seed}.toml"
        create_config "experiment_configs/blog_1tbl_trivial.toml" "$config" \
            "inter_arrival.scale" "$scale" \
            "seed" "$seed"
        configs_1+=("$config")
    done
done

echo "Running ${#configs_1[@]} configurations..."
printf '%s\n' "${configs_1[@]}" | xargs -P $NUM_PARALLEL -I {} bash -c 'run_experiment "$@"' _ {}
echo "Experiment 1 complete!"
echo ""

# ============================================================================
# Experiment 2: Single Table, Non-Trivial Conflicts
# ============================================================================
echo "=== Experiment 2: Single Table, Non-Trivial Conflicts ==="
CONFLICT_PROBS=(0.0 0.1 0.3 0.5 1.0)

configs_2=()
for scale in "${SCALES[@]}"; do
    for prob in "${CONFLICT_PROBS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            config="$TEMP_DIR/blog_1tbl_nontrivial_s${scale}_p${prob}_seed${seed}.toml"
            create_config "experiment_configs/blog_1tbl_nontrivial.toml" "$config" \
                "inter_arrival.scale" "$scale" \
                "real_conflict_probability" "$prob" \
                "seed" "$seed"
            configs_2+=("$config")
        done
    done
done

echo "Running ${#configs_2[@]} configurations..."
printf '%s\n' "${configs_2[@]}" | xargs -P $NUM_PARALLEL -I {} bash -c 'run_experiment "$@"' _ {}
echo "Experiment 2 complete!"
echo ""

# ============================================================================
# Experiment 3: Multi-Table, Trivial Conflicts
# ============================================================================
echo "=== Experiment 3: Multi-Table, Trivial Conflicts ==="
NUM_TABLES=(1 2 5 10 20)

configs_3=()
for scale in "${SCALES[@]}"; do
    for ntbl in "${NUM_TABLES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            config="$TEMP_DIR/blog_ntbl_trivial_s${scale}_t${ntbl}_seed${seed}.toml"
            create_config "experiment_configs/blog_ntbl_trivial.toml" "$config" \
                "inter_arrival.scale" "$scale" \
                "num_tables" "$ntbl" \
                "seed" "$seed"
            configs_3+=("$config")
        done
    done
done

echo "Running ${#configs_3[@]} configurations..."
printf '%s\n' "${configs_3[@]}" | xargs -P $NUM_PARALLEL -I {} bash -c 'run_experiment "$@"' _ {}
echo "Experiment 3 complete!"
echo ""

# ============================================================================
# Experiment 4: Multi-Table, Non-Trivial Conflicts
# ============================================================================
echo "=== Experiment 4: Multi-Table, Non-Trivial Conflicts ==="
NUM_TABLES_4=(2 5 10)
CONFLICT_PROBS_4=(0.0 0.3 0.5)

configs_4=()
for scale in "${SCALES[@]}"; do
    for ntbl in "${NUM_TABLES_4[@]}"; do
        for prob in "${CONFLICT_PROBS_4[@]}"; do
            for seed in "${SEEDS[@]}"; do
                config="$TEMP_DIR/blog_ntbl_nontrivial_s${scale}_t${ntbl}_p${prob}_seed${seed}.toml"
                create_config "experiment_configs/blog_ntbl_nontrivial.toml" "$config" \
                    "inter_arrival.scale" "$scale" \
                    "num_tables" "$ntbl" \
                    "real_conflict_probability" "$prob" \
                    "seed" "$seed"
                configs_4+=("$config")
            done
        done
    done
done

echo "Running ${#configs_4[@]} configurations..."
printf '%s\n' "${configs_4[@]}" | xargs -P $NUM_PARALLEL -I {} bash -c 'run_experiment "$@"' _ {}
echo "Experiment 4 complete!"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=============================================="
echo "All Blog Experiments Complete!"
echo "=============================================="
echo ""
echo "Experiment counts:"
echo "  1. Single table, trivial:     ${#configs_1[@]} runs"
echo "  2. Single table, non-trivial: ${#configs_2[@]} runs"
echo "  3. Multi-table, trivial:      ${#configs_3[@]} runs"
echo "  4. Multi-table, non-trivial:  ${#configs_4[@]} runs"
echo ""
echo "Total: $((${#configs_1[@]} + ${#configs_2[@]} + ${#configs_3[@]} + ${#configs_4[@]})) runs"
echo ""
echo "Results in experiments/blog_*/"
