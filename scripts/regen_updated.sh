#!/bin/bash
# Regenerate only the plots affected by the linear x-axis fix.
set -e

echo "=== workload_knee_vs_tables ==="
python -c "
from endive.saturation_analysis import generate_workload_knee_table
generate_workload_knee_table('experiments', 'exp4c_*', 'plots/exp4c_tables_providers/workload_knee')
"

echo ""
echo "=== op_type_vs_load ==="
python -c "
from endive.saturation_analysis import generate_operation_type_plots
generate_operation_type_plots(
    'experiments', 'exp4b_*', 'plots/exp4b_tables_mix',
    group_by='catalog_service_latency_ms',
)
generate_operation_type_plots(
    'experiments', 'exp4c_*', 'plots/exp4c_tables_providers/op_types_fa100_1t',
    group_by='storage_provider',
    config={'filters': ['num_tables==1', 'fast_append_ratio==1.0']},
)
generate_operation_type_plots(
    'experiments', 'exp4c_*', 'plots/exp4c_tables_providers/op_types_fa90_1t',
    group_by='storage_provider',
    config={'filters': ['num_tables==1', 'fast_append_ratio==0.9']},
)
generate_operation_type_plots(
    'experiments', 'exp4c_*', 'plots/exp4c_tables_providers/op_types_fa50_1t',
    group_by='storage_provider',
    config={'filters': ['num_tables==1', 'fast_append_ratio==0.5']},
)
"

echo ""
echo "Done."
