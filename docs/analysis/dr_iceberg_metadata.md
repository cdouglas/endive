# Estimating Typical Sizes of Apache Iceberg Metadata Artifacts in v1.10.1

## Executive summary

Iceberg’s on-disk metadata for a snapshot is a small “metadata tree” rooted at a **table metadata JSON** file (e.g., `v<N>.metadata.json`) that points to a per-snapshot **manifest list** file, which in turn points to one or more **manifest files** that contain per–data-file (or per–delete-file) entries. The Iceberg table spec explicitly defines table metadata as **JSON** and defines manifests as **immutable Avro** files; manifest lists are also stored as Iceberg “data files” and, in the reference implementation (Iceberg 1.10.1), are read/written as **Avro** as well. citeturn9view0turn4view0turn12view0turn24view0turn11view0

For benchmarking **physical file sizes** (bytes on storage, uncompressed unless stated), the most practically stable “typical size knobs” in Iceberg are:

- **Manifest files** tend to cluster around the **merge target** size `commit.manifest.target-size-bytes`, which defaults to **8 MiB**, because Iceberg can automatically merge many small manifests into fewer larger ones once enough have accumulated (default min count to merge: **100**). citeturn23view0turn3view0  
- **Manifest lists** are “one per committed snapshot attempt” and contain *one row per manifest file* plus some summary fields; they are generally **much smaller than manifests** unless a snapshot references thousands of manifests. citeturn6view1turn27view0  
- **Table metadata JSON** is usually **small (<1 MiB)** for most real tables because it stores *pointers* (to manifest lists, etc.) rather than enumerating all data files, but it can grow with the number of snapshots retained, schemas/specs/orders stored, and log history retained. It may optionally be **GZIP-compressed** per spec and is supported by the 1.10.1 JSON parser/reader. citeturn9view0turn9view1turn12view5

A practical benchmark recommendation (assuming **format-version 2 default**, typical metrics defaults, and “reasonably maintained” tables where manifest merging is enabled):

- **Table metadata JSON (`v*.metadata.json`, uncompressed):** **~100 KiB typical** (often 10–300 KiB), with a heavy tail into the low MiB range for very long histories or very wide schema catalogs. (Below ~1 MiB, distributions are typically insensitive for most system benchmarks.) citeturn9view1turn23view0  
- **Manifest list (`snap-*.avro`):** **~16–64 KiB typical**, scaling roughly linearly in the number of manifest files referenced by the snapshot; crosses 1 MiB only when there are on the order of **thousands of manifests** in one snapshot. citeturn6view1turn27view0turn24view0  
- **Manifest file (`*.avro`):** **~8 MiB typical/median in tables that undergo merging/compaction of manifests**, but with many **small manifests** (tens/hundreds of KiB) possible in streaming or frequent small commits *before* merges kick in. citeturn23view0turn3view0turn11view0  

Where an artifact is usually under ~1 MiB (table metadata JSON and many manifest lists), the *exact* min/median/mean/90th you choose rarely changes benchmark outcomes unless you are explicitly modeling metadata-cache pressure, network fanout, or control-plane throughput. Under that threshold, modeling **count of files** and **frequency of access** often dominates modeling byte-level size.

## Formats and schemas in Iceberg 1.10.1

### How the files relate

```mermaid
flowchart TD
  A[Catalog pointer<br/>to current metadata] --> B[vN.metadata.json<br/>table metadata JSON]
  B -->|snapshots[].manifest-list| C[snap-...avro<br/>manifest list (Avro)]
  C -->|each manifest_file.manifest_path| D[manifest-....avro<br/>manifest (Avro)]
  D -->|each manifest_entry.data_file.file_path| E[data files / delete files<br/>Parquet/ORC/Avro/etc]
```

This “pointer fanout” is in the spec’s terminology: snapshot → manifest list → manifests. citeturn4view0turn9view1turn6view1

### Table metadata JSON (`v<N>.metadata.json`)

**What it is.** The Iceberg spec defines table metadata as a JSON object; every metadata update writes a new file and commits it via an atomic swap mechanism (e.g., renames for filesystem tables, pointer swaps in metastores). citeturn9view0turn8view0

**Key fields that affect size.** The spec enumerates the top-level fields; the most size-relevant are:

- `schemas` (list) and `current-schema-id` citeturn9view0turn8view2  
- `partition-specs`, `default-spec-id`, `last-partition-id` citeturn9view0turn9view1  
- `sort-orders`, `default-sort-order-id` citeturn9view1  
- `properties` (string map) citeturn9view1turn23view0  
- `snapshots` (list) where each snapshot includes a `manifest-list` location citeturn9view1turn9view2  
- `snapshot-log` and `metadata-log` history lists citeturn9view1turn23view0  
- `refs` (branches/tags) citeturn9view1turn12view4  

**Compression.** The spec explicitly states a metadata JSON file *may be compressed with GZIP* and the 1.10.1 parser reads a file through a codec chosen from the filename, using `GZIPInputStream` when the codec is GZIP. citeturn9view1turn12view5  
Separately, the table property `write.metadata.compression-codec` (default `none`) is documented in Iceberg configuration (values `none` or `gzip`). citeturn23view0turn3view0

**Representative snippet (spec-shaped).** The spec includes an example of how snapshots appear within table metadata, including `manifest-list`. citeturn9view1turn9view2  
A minimal-ish (illustrative) excerpt:

```json
{
  "format-version": 2,
  "table-uuid": "fb072c92-a02b-11e9-ae9c-1bb7bc9eca94",
  "location": "s3://bucket/warehouse/db/table",
  "last-updated-ms": 1515100955770,
  "schemas": [ /* schema objects */ ],
  "partition-specs": [ /* spec objects */ ],
  "properties": { "write.format.default": "parquet" },
  "current-snapshot-id": 3051729675574597004,
  "snapshots": [{
    "snapshot-id": 3051729675574597004,
    "timestamp-ms": 1515100955770,
    "summary": { "operation": "append" },
    "manifest-list": "s3://bucket/.../snap-...avro",
    "schema-id": 0
  }]
}
```

### Manifest list (`snap-*.avro`) schema

**What it is.** The spec describes manifest lists as separate files (one per snapshot commit attempt) that store a list of `manifest_file` structs and summary metadata. citeturn6view1turn5view2

**On-disk format in Iceberg 1.10.1.** In the 1.10.1 source, the helper `ManifestLists.read(...)` uses `InternalData.read(FileFormat.AVRO, ...)` and `ManifestListWriter` uses `InternalData.write(FileFormat.AVRO, ...)` to write manifest lists, i.e., the reference implementation treats manifest lists as **Avro**. citeturn12view0turn24view0

**Exact fields.** The spec enumerates the `manifest_file` fields (IDs 500+), including manifest path, length, spec id, counts, partition summaries, and optional key metadata / row lineage. citeturn6view1turn7view1  
The Iceberg 1.10.1 API cements the same schema in code as `ManifestFile.SCHEMA`, including partition summary sub-struct and optional `first_row_id`. citeturn27view0  
The 1.10.1 core defines per-version “manifest list schema” constants, e.g., `V2Metadata.MANIFEST_LIST_SCHEMA` or `V4Metadata.MANIFEST_LIST_SCHEMA`, built directly from `ManifestFile.*` fields. citeturn24view1turn25view0turn24view0

**Representative manifest-list entry snippet (JSON-shaped).** While the file is Avro, a logical row looks like:

```json
{
  "manifest_path": "s3://bucket/.../manifest-00001.avro",
  "manifest_length": 8388608,
  "partition_spec_id": 0,
  "content": 0,
  "sequence_number": 42,
  "min_sequence_number": 42,
  "added_snapshot_id": 3051729675574597004,
  "added_files_count": 1200,
  "existing_files_count": 0,
  "deleted_files_count": 0,
  "added_rows_count": 120000000,
  "partitions": [
    { "contains_null": false, "lower_bound": "…bytes…", "upper_bound": "…bytes…" }
  ]
}
```

(The field set is per spec and per the API interface.) citeturn6view1turn27view0

### Manifest file (`manifest-*.avro`) schema

**What it is.** The spec states “a manifest is an immutable Avro file” that lists data or delete files and stores partition tuples, metrics, and tracking info. citeturn4view0turn19search2

**On-disk format in Iceberg 1.10.1.** In 1.10.1, `ManifestWriter` writes manifests using `InternalData.write(FileFormat.AVRO, ...)` and attaches metadata such as the JSON table schema and partition spec. citeturn11view0turn4view0

**Exact fields.** The manifest entry schema is defined in the spec as `manifest_entry` with fields `status`, `snapshot_id`, and (v2+) sequence numbers plus a nested `data_file` struct. citeturn6view0turn7view0  
In the 1.10.1 API, `ManifestEntry` defines those same fields (IDs 0–4) and requires a nested `data_file` struct. citeturn26view1  
In 1.10.1, the `DataFile` interface defines the nested struct fields that appear inside manifests: file path, format, partition tuple, record count, file size, and a set of optional per-column metrics maps (`column_sizes`, `value_counts`, `null_value_counts`, `nan_value_counts`, `lower_bounds`, `upper_bounds`), plus optional encryption and split metadata. citeturn27view1turn7view0

**Column metrics defaults that materially affect manifest size.** Iceberg’s documented defaults matter a lot for byte sizes:

- `write.metadata.metrics.max-inferred-column-defaults` default **100**: metrics collected for up to the first 100 columns (traversal defined). citeturn23view0turn21search5  
- `write.metadata.metrics.default` default **`truncate(16)`**: default metrics mode for all columns (options include `none`, `counts`, `truncate(length)`, `full`). citeturn23view0  

These properties primarily control *how much data is populated into the `DataFile` metrics maps*, which dominates the per-entry size of manifests for wide tables.

**Representative manifest entry snippet (JSON-shaped).** A logical row (not the physical Avro bytes) looks like:

```json
{
  "status": 1,
  "snapshot_id": 3051729675574597004,
  "sequence_number": 42,
  "file_sequence_number": 42,
  "data_file": {
    "content": 0,
    "file_path": "s3://bucket/.../data/00001.parquet",
    "file_format": "parquet",
    "partition": { "p_day": 20210103 },
    "record_count": 1200000,
    "file_size_in_bytes": 536870912,
    "value_counts": { "1": 1200000, "2": 1200000 },
    "null_value_counts": { "1": 0, "2": 0 },
    "lower_bounds": { "1": "…bytes…", "2": "…bytes…" },
    "upper_bounds": { "1": "…bytes…", "2": "…bytes…" }
  }
}
```

(The presence/absence of the map fields is the major size driver.) citeturn7view0turn26view1turn27view1

## Size modeling approach and assumptions

Your request asks for “compute or measure sizes.” The Iceberg 1.10.1 source and the spec precisely define the *fields* and show that manifests/lists are written as Avro in the reference code. citeturn11view0turn12view0turn4view0  
However, without executing Iceberg writers against real tables here, what is feasible in a rigorous report is a **schema-driven byte model** that:

- Treats **table metadata** as UTF-8 JSON (compact) and sizes it by serialized length. (Spec defines JSON format; parser supports optional gzip.) citeturn9view0turn12view5  
- Treats **manifest list** and **manifest** as Avro container files (as per spec + 1.10.1 writer code) and sizes them by:
  - a **header term** (Avro container header + embedded schema/metadata strings), plus
  - **(number of rows × average row payload)**, where row payload is dominated by string lengths and (for manifests) the presence/size of metrics maps. citeturn11view0turn24view0turn27view1  

All resulting sizes are **explicitly conditional** on assumptions you can change.

Baseline assumptions chosen for “typical” numbers below (and called out in the comparison table):

- **Iceberg format version:** default 2 (as documented). citeturn23view0  
- **Data file paths:** average **~180 bytes** (typical of S3/ADLS URIs with UUID-ish filenames).  
- **Partitioning:** 1–2 partition fields, small bounds (e.g., integer/date).  
- **Metrics defaults:** up to **100 columns** with `truncate(16)` bounds by default. citeturn23view0turn27view1  
- **Compression:** unspecified ⇒ assume *uncompressed* physical sizes. (Iceberg supports gzip for metadata JSON via spec/property; manifests themselves are Avro in code, but the metadata compression property is explicitly for “metadata compression codec” and is commonly applied to metadata JSON; you should state whether you enable it.) citeturn23view0turn12view5  

## Estimated sizes for minimal, typical, and large examples

### Table metadata JSON (`v*.metadata.json`)

**Minimal example (empty or near-empty table).** The smallest “realistic” metadata JSON still includes schema/spec/order arrays, properties, and empty lists for snapshots/logs. The spec shows the required presence and naming of these fields (depending on version) and the serializer in 1.10.1 writes substantial structure even when lists are empty. citeturn9view0turn12view4  
A minimal table metadata JSON is typically on the order of **a few KiB** (≈ 2–10 KiB).

**Typical example (moderate history).** For a table with:
- 50–200 columns,
- 10–100 snapshots retained,
- metadata-log tracking on the order of 10–100 prior versions (default max tracked is 100), citeturn23view0turn9view1  

a compact JSON serialization is typically **~10–200 KiB**, often clustering around **~50–150 KiB**.

**Large example (very wide + long history).** With ~1000 columns and ~1000 snapshots, it is plausible for uncompressed metadata JSON to land in the **~0.5–3 MiB** range depending on how much schema/spec history is retained and how large the per-snapshot `summary` maps are. (The spec allows many optional summary keys; engines can include lots of strings.) citeturn9view2turn9view1  

**Benchmark recommendation and distribution.**
- **Recommended “typical” size:** **100 KiB** (uncompressed).
- **Plausible distribution (uncompressed):**
  - min: **4 KiB**
  - median: **80 KiB**
  - mean: **140 KiB** (heavy tail)
  - 90th percentile: **400 KiB**

**Sensitivity note (<1 MiB).** For many benchmarks, replacing (say) 80 KiB vs 200 KiB changes little unless:
- you are saturating a metadata cache by bytes rather than objects,
- you model control-plane bandwidth at very high commit rates, or
- you include many historical metadata fetches per query.

If you only need a single “average,” using **100 KiB** is usually sufficient.

**Scaling rule of thumb.** Let:
- `C` = number of columns stored in the schema JSON,
- `S` = number of snapshots retained in `snapshots`,
- `L` = number of entries in `metadata-log` and `snapshot-log`,
- `P` = number of partition specs + sort orders retained.

A practical linear model for compact JSON is:

\[
\text{size(metadata.json)} \approx A \;+\; b_C \cdot C \;+\; b_S \cdot S \;+\; b_L \cdot L \;+\; b_P \cdot P
\]

where (empirically reasonable for compact JSON with moderate key lengths):
- \(A \approx 2\text{–}10\text{ KiB}\) base structure,
- \(b_C \approx 40\text{–}120\) bytes/column (field objects),
- \(b_S \approx 150\text{–}500\) bytes/snapshot (dominated by the `manifest-list` URI and summary keys),
- \(b_L \approx 120\text{–}250\) bytes/log entry (URI + timestamp).  

The spec’s inclusion of `manifest-list` URIs inside each snapshot is the key per-snapshot contributor. citeturn9view1turn9view2

### Manifest list (`snap-*.avro`)

**Minimal example.** A manifest list with a single manifest entry is usually **header-dominated**: you pay the Avro container header + embedded schema + a small amount of metadata for snapshot ids/sequence numbers (written as Avro file metadata by the writer code). citeturn24view0turn12view0  
Estimated minimal size: **~5–10 KiB**.

**Typical example.** For snapshots that reference **10–200 manifest files**, a manifest list is usually **~10–100 KiB**.

**Large example.** Manifest lists grow linearly with the number of manifests in the snapshot. If a snapshot references **~5000 manifests**, it is plausible for the manifest list to reach **~1–2 MiB** (still often smaller than the manifests they point to).

**Benchmark recommendation and distribution.**
- **Recommended “typical” size:** **32 KiB**
- **Plausible distribution (uncompressed Avro):**
  - min: **6 KiB**
  - median: **24 KiB**
  - mean: **64 KiB**
  - 90th percentile: **200 KiB**

**Sensitivity note (<1 MiB).** Manifest lists are frequently <1 MiB; for many systems they are more sensitive to:
- *count of manifest lists read per query* (often 1), and
- *latency of the object store GET*,
than to the difference between 20 KiB and 200 KiB.

**Scaling rule of thumb.** Using the `ManifestFile` schema:

\[
\text{size(manifest-list)} \approx H_{ml} \;+\; M \cdot r_{ml}
\]

- \(H_{ml}\): Avro header + schema + fixed metadata (often a few KiB).
- \(M\): number of manifest files referenced by the snapshot.
- \(r_{ml}\): average row size, dominated by `manifest_path` length plus partition summaries. citeturn6view1turn27view0  

A practical row estimate:
- \(r_{ml} \approx 80 + \text{len(manifest\_path)} + 10 \cdot F_p\) bytes,  
where \(F_p\) is the number of partition fields (each adds a small `field_summary` struct with booleans and optional bounds). citeturn7view1turn27view0  

### Manifest file (`manifest-*.avro`)

**Minimal example.** A manifest with just a few entries is often **header-dominated** because Iceberg writes JSON strings for the table schema and partition spec into Avro key-value metadata when writing the file. citeturn11view0turn4view0  
Estimated minimal size for 1–5 entries: **~15–50 KiB**.

**Typical example (with manifest merging).** The key stability lever is Iceberg’s manifest merging behavior:

- Iceberg’s table properties include `commit.manifest.target-size-bytes` with default **8 MiB** and merging enabled by default (`commit.manifest-merge.enabled = true`) with default min-count-to-merge **100**. citeturn23view0turn3view0  

For tables where manifests are periodically merged, many “steady-state” manifests cluster around **~8 MiB** (some smaller, some larger depending on entry payload size).

**Large example.** The manifest itself can exceed 8 MiB if:
- entries are large (many metrics maps populated, long paths, large bound values),
- merges produce larger-than-target results (e.g., imperfect packing),
- you disable merging or set larger targets.

It is plausible to see **~16–32 MiB** manifests in some environments, especially when metrics are “full” on many columns.

**Benchmark recommendation and distribution.**
- **Recommended “typical” size:** **8 MiB** (8,388,608 bytes) for a “maintained” table
- **Plausible distribution (uncompressed Avro, maintained table):**
  - min: **20 KiB** (tiny commits / first snapshot)
  - median: **8 MiB**
  - mean: **6 MiB**
  - 90th percentile: **10 MiB**

If you are modeling a **streaming / micro-batch** workload *before merges*, use a different “typical”:
- **typical (streaming pre-merge):** **256–512 KiB**, with many small manifests.

**Sensitivity note.** Unlike the other two artifacts, manifests frequently exceed 1 MiB; benchmark results can be sensitive to manifest size when:
- planning reads must scan many manifest rows,
- metadata caching is bytes-constrained,
- you read manifests over a relatively low-bandwidth control plane.

**Scaling rule of thumb.** Using the `manifest_entry` + `DataFile` schema:

\[
\text{size(manifest)} \approx H_{m} \;+\; N \cdot r_{e}
\]

- \(H_{m}\): Avro header + schema + embedded JSON metadata strings  
  (notably the JSON table schema + partition spec stored as Avro file metadata). citeturn11view0turn4view0  
- \(N\): number of manifest entries (≈ number of data files + delete files tracked in this manifest).
- \(r_e\): average entry size, dominated by:
  - `file_path` string length, citeturn27view1turn7view0  
  - number of partition fields, citeturn7view0turn27view1  
  - number of columns for which metrics are written (default up to 100), and the chosen metrics mode (default `truncate(16)`). citeturn23view0turn27view1  

A practical per-entry estimate (order-of-magnitude) for Parquet data files with default metrics:

- **No metrics written:** ~**200–400 B** per entry (mostly strings + a few longs).  
- **Counts-only metrics for ~100 columns:** ~**1–3 KiB** per entry.  
- **Counts + truncated bounds (`truncate(16)`) for ~100 columns:** ~**4–8 KiB** per entry (bounds dominate).  

These regimes correspond directly to the presence/absence of the `DataFile` metric maps (`value_counts`, `null_value_counts`, `lower_bounds`, `upper_bounds`, etc.). citeturn27view1turn7view0turn23view0  

From this, entries per 8 MiB manifest are roughly:
- ~20k–40k entries (no metrics),
- ~3k–8k entries (counts-only),
- ~1k–2k entries (counts + truncated bounds).

## Variability factors and what to parametrize in a benchmark

### Factors that expand or shrink table metadata JSON

- **Snapshot retention / history.** `snapshots`, `snapshot-log`, and `metadata-log` scale with retained history. The spec spells out these structures and the `metadata-log` stores prior metadata file paths. citeturn9view1turn23view0  
- **Schema/spec/order evolution.** Each new schema/spec/order is stored in arrays; wide schemas increase size linearly in the number of fields. citeturn9view0turn9view1  
- **Properties bloat.** Large property maps can add meaningful text, but are usually small. citeturn23view0turn9view1  
- **Compression.** If you enable gzip for metadata JSON (spec allows gzip; implementation supports it), your physical downloaded bytes can drop substantially. citeturn9view1turn12view5turn23view0  

### Factors that expand or shrink manifest lists

- **Number of manifests per snapshot (`M`).** Dominant linear factor: one `manifest_file` row per manifest. citeturn6view1turn27view0  
- **Partition spec complexity / partition-field summaries.** Each manifest list row can include `partitions` summaries: one `field_summary` per partition field (contains_null + optional bounds). For many partition fields or large bounds (e.g., long strings), this can add noticeable bytes. citeturn7view1turn27view0  
- **Path lengths.** `manifest_path` is a string and can dominate the row if it’s long. citeturn27view0turn6view1  

### Factors that expand or shrink manifests

- **Metrics mode and number of columns with metrics.** `write.metadata.metrics.max-inferred-column-defaults` (default 100) and `write.metadata.metrics.default` (default `truncate(16)`) strongly influence how many `DataFile` metric maps are populated and how large bounds are. citeturn23view0turn27view1  
- **Data file format.** The spec notes `column_sizes` should be null for row-oriented formats (Avro); Parquet/ORC typically have richer stats, while Avro data files may have fewer relevant maps populated. citeturn6view0turn7view0  
- **File path length distribution.** S3 URIs with deep prefixes and UUID file names can add hundreds of bytes per entry. `file_path` is required. citeturn27view1turn7view0  
- **Partition schema width.** Each entry’s `partition` struct grows with the number of partition fields in the spec. citeturn7view0turn27view1  
- **Manifest merging.** The existence of an 8 MiB default target for merging (`commit.manifest.target-size-bytes`) plus default min-count-to-merge (100) changes the steady-state file size distribution. citeturn23view0turn3view0  
- **Delete files and row lineage fields.** In newer format versions, fields like `first_row_id`, `referenced_data_file`, `content_offset`, and `content_size_in_bytes` can appear for delete vectors and row lineage, increasing per-entry payload in some workloads. citeturn7view0turn27view1turn25view2  

## Comparison table of recommended benchmark sizes

Assumptions: uncompressed bytes; typical URI lengths ~180B; format-version 2 typical; default metrics (`truncate(16)`, up to 100 columns).

| Artifact | Physical format in Iceberg 1.10.1 | Minimal (bytes) | Typical recommendation | Large (bytes) | Notes |
|---|---|---:|---:|---:|---|
| Table metadata `v*.metadata.json` | JSON (optionally gzip) citeturn9view0turn12view5 | ~4–10 KiB | **100 KiB** | ~0.5–3 MiB | Usually <1 MiB; size ∝ (#snapshots + schema width). citeturn9view1turn23view0 |
| Manifest list `snap-*.avro` | Avro (in reference code) citeturn12view0turn24view0 | ~5–10 KiB | **32 KiB** | ~1–2 MiB | Size ∝ number of manifests in snapshot; often <1 MiB. citeturn6view1turn27view0 |
| Manifest `manifest-*.avro` | Avro (spec + code) citeturn4view0turn11view0 | ~20–50 KiB | **8 MiB** | ~16–32 MiB | Steady-state often near 8 MiB due to merge target; per-entry size dominated by column metrics maps. citeturn23view0turn27view1 |

## Sources and repository pointers used

- Iceberg table spec (table metadata JSON fields, snapshots’ `manifest-list`, manifest list and manifest schema definitions). citeturn9view0turn9view1turn9view2turn6view1turn7view0turn4view0  
- Iceberg 1.10.1 source (tag `apache-iceberg-1.10.1`) for defaults and writer formats:
  - `TableProperties`: `commit.manifest.target-size-bytes` default 8 MiB; metadata compression default; merge toggles. citeturn3view0turn23view0  
  - `TableMetadataParser`: gzip codec handling, JSON serialization behavior. citeturn12view4turn12view5  
  - `ManifestLists` and `ManifestListWriter`: manifest list read/write uses `FileFormat.AVRO`. citeturn12view0turn24view0  
  - `ManifestWriter`: manifest write uses `FileFormat.AVRO` and writes schema/spec metadata. citeturn11view0turn4view0  
  - API schemas: `ManifestFile`, `ManifestEntry`, `DataFile` field IDs and types. citeturn27view0turn26view1turn27view1  
- Iceberg configuration docs (`write.metadata.metrics.*` defaults and `commit.manifest.*` defaults). citeturn23view0