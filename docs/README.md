# Documentation Index

Authoritative simulator reference lives at [`../SPEC.md`](../SPEC.md). Project overview in [`../README.md`](../README.md).

## Core

| Document | Description |
|----------|-------------|
| [model.md](model.md) | Simplifications made vs. real Iceberg implementations |
| [CONSOLIDATED_FORMAT.md](CONSOLIDATED_FORMAT.md) | Layout of `experiments/consolidated.parquet` |

## Experiment Designs

| Document | Description |
|----------|-------------|
| [EXP5.md](EXP5.md) | Partition-aware single table (exp5a/5b) |
| [EXP6.md](EXP6.md) | Inlined table metadata (exp6a/6b) |

See [`../experiment_configs/`](../experiment_configs/) for all runnable configs (exp1–exp6) and [`scripts/run_all_experiments.py`](../scripts/run_all_experiments.py) for the experiment groups.

## Analysis & Reference Data

| Document | Description |
|----------|-------------|
| [analysis/IO_CONVOY_ANALYSIS.md](analysis/IO_CONVOY_ANALYSIS.md) | I/O convoy pattern in partition-level OCC |
| [analysis/des_profiling_report.md](analysis/des_profiling_report.md) | SimPy DES engine profiling: ~75–107k events/s |
| [analysis/latency_verification.md](analysis/latency_verification.md) | Provider latency parameters vs. measured sources |
| [analysis/simulation_summary.md](analysis/simulation_summary.md) | YCSB June 2025 benchmark measurements (raw) |
| [analysis/dr_put_get.md](analysis/dr_put_get.md) | PUT/GET latency research across cloud providers |
| [analysis/dr_s3x.md](analysis/dr_s3x.md) | S3 Express One Zone PUT/GET research |
| [analysis/dr_iceberg_metadata.md](analysis/dr_iceberg_metadata.md) | Iceberg metadata artifact size research |
| [analysis/distributions.json](analysis/distributions.json) | Raw per-provider latency distribution fits |

## Research Notes (not part of the simulator)

The `review/` subdirectory contains design documents for a separate catalog file-format effort (LCF/PB) and paper notes — not implemented in the simulator.
