# Experiment Summary

| Exp | Description | Fixed | Swept | Configs |
|-----|-------------|-------|-------|---------|
| **1** | FA baseline, instant catalog | 1 table, 1 group, FA=100%, instant catalog (1ms), S3, conflicts=0% | `inter_arrival_scale`: [20, 50, 100, 200, 300, 400, 500, 1000, 2000, 5000] ms | 10 |
| **2** | FA/VO operation mix | 1 table, 1 group, instant catalog (1ms), S3, conflicts=0% | `fast_append_ratio`: [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.0]<br>`inter_arrival_scale`: 10 values | 80 |
| **3a** | Catalog CAS latency (FA) | 1 table, 1 group, FA=100%, S3, conflicts=0% | `catalog_latency_ms`: [1, 5, 10, 20, 50, 80, 120]<br>`inter_arrival_scale`: 10 values | 70 |
| **3b** | Catalog CAS latency (mix) | 1 table, 1 group, FA=90%/VO=10%, S3, conflicts=0% | `catalog_latency_ms`: [1, 5, 10, 20, 50, 80, 120]<br>`inter_arrival_scale`: 10 values | 70 |
| **4a** | Multi-table contention (FA) | 1 group, FA=100%, S3, conflicts=0% | `num_tables`: [1, 2, 5, 10, 20, 50]<br>`catalog_latency_ms`: [1, 10, 50, 120]<br>`inter_arrival_scale`: 10 values | 240 |
| **4b** | Multi-table contention (mix) | 1 group, FA=90%/VO=10%, S3, conflicts=0% | `num_tables`: [1, 2, 5, 10, 20, 50]<br>`catalog_latency_ms`: [1, 10, 50, 120]<br>`inter_arrival_scale`: 10 values | 240 |
| **4c** | Multi-table, real providers | 1 group, conflicts=0%, backend=storage | `provider`: [s3x, s3, azurex, azure, gcp]<br>`num_tables`: [1, 2, 5, 10, 20, 50]<br>`fast_append_ratio`: [1.0, 0.9, 0.5]<br>`inter_arrival_scale`: 10 values | 900 |

All experiments use 5 seeds, `inter_arrival_scale` = [20, 50, 100, 200, 300, 400, 500, 1000, 2000, 5000] ms, retry=10, txn runtime mean=180s. Each run simulates 1 hour, with the first and last 15 minutes excluded as warmup/cooldown. Config counts exclude seeds. Total: 1,610 configs × 5 seeds = 8,050 runs.
