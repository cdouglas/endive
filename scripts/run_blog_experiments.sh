#!/bin/bash
# Run blog post experiments with parameter sweeps
# Each experiment runs 3 seeds for statistical confidence

set -e

SEEDS=${SEEDS:-"42 43 44"}
PARALLEL=${PARALLEL:-4}

# Create temp directory for variant configs
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

create_config_variant() {
    local base_config=$1
    local output=$2
    local seed=$3
    shift 3

    cp "$base_config" "$output"

    # Set seed
    if grep -q '^\[simulation\]' "$output"; then
        sed -i "/^\[simulation\]/a seed = $seed" "$output"
    fi

    # Apply parameter overrides (key=value pairs)
    while [[ $# -gt 0 ]]; do
        local param=$1
        local key="${param%%=*}"
        local value="${param#*=}"

        # Handle dotted keys (e.g., inter_arrival.scale)
        if [[ "$key" == *"."* ]]; then
            local section="${key%%.*}"
            local subkey="${key#*.}"
            sed -i "s/^${subkey} = .*/${subkey} = ${value}/" "$output"
        else
            sed -i "s/^${key} = .*/${key} = ${value}/" "$output"
        fi
        shift
    done
}

run_simulation() {
    local config=$1
    echo "[$(date '+%H:%M:%S')] Running: $config"
    python -m endive.main "$config" --yes 2>&1 | tail -20
}

export -f run_simulation

echo "========================================"
echo "Blog Post Experiment Runner"
echo "========================================"
echo "Seeds: $SEEDS"
echo "Parallel: $PARALLEL"
echo ""

# Question 1a: Single table trivial - sweep inter_arrival.scale
echo "=== Question 1a: Single Table Saturation ==="
SCALES="10 20 50 100 200 500 1000 2000 5000"
for scale in $SCALES; do
    for seed in $SEEDS; do
        config="$TMPDIR/single_table_trivial_scale${scale}_s${seed}.toml"
        create_config_variant experiment_configs/single_table_trivial.toml "$config" "$seed" "inter_arrival.scale=$scale"
        echo "$config"
    done
done | xargs -P $PARALLEL -I {} bash -c 'run_simulation "{}"'

echo ""
echo "=== Question 1a variant: With Backoff ==="
for scale in $SCALES; do
    for seed in $SEEDS; do
        config="$TMPDIR/single_table_trivial_backoff_scale${scale}_s${seed}.toml"
        create_config_variant experiment_configs/single_table_trivial_backoff.toml "$config" "$seed" "inter_arrival.scale=$scale"
        echo "$config"
    done
done | xargs -P $PARALLEL -I {} bash -c 'run_simulation "{}"'

echo ""
echo "=== Question 1b: Non-trivial Conflict Impact ==="
CONFLICT_PROBS="0.0 0.1 0.2 0.3 0.5 0.7 1.0"
for prob in $CONFLICT_PROBS; do
    for seed in $SEEDS; do
        config="$TMPDIR/single_table_mixed_p${prob}_s${seed}.toml"
        create_config_variant experiment_configs/single_table_mixed.toml "$config" "$seed" "real_conflict_probability=$prob"
        echo "$config"
    done
done | xargs -P $PARALLEL -I {} bash -c 'run_simulation "{}"'

echo ""
echo "=== Question 2a: Multi-table Scaling ==="
NUM_TABLES="1 2 5 10 20 50"
for nt in $NUM_TABLES; do
    for seed in $SEEDS; do
        config="$TMPDIR/multi_table_trivial_t${nt}_s${seed}.toml"
        create_config_variant experiment_configs/multi_table_trivial.toml "$config" "$seed" "num_tables=$nt"
        echo "$config"
    done
done | xargs -P $PARALLEL -I {} bash -c 'run_simulation "{}"'

echo ""
echo "=== Question 2b: Multi-table with Conflicts ==="
NUM_TABLES="2 5 10 20"
CONFLICT_PROBS="0.0 0.1 0.3 0.5"
for nt in $NUM_TABLES; do
    for prob in $CONFLICT_PROBS; do
        for seed in $SEEDS; do
            config="$TMPDIR/multi_table_mixed_t${nt}_p${prob}_s${seed}.toml"
            create_config_variant experiment_configs/multi_table_mixed.toml "$config" "$seed" "num_tables=$nt" "real_conflict_probability=$prob"
            echo "$config"
        done
    done
done | xargs -P $PARALLEL -I {} bash -c 'run_simulation "{}"'

echo ""
echo "========================================"
echo "All experiments complete!"
echo "========================================"
