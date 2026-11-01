"""Streaming Parquet shards for Craftax trajectories."""
from pathlib import Path
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def arrays_to_table(data):
    arrays, shapes = {}, {}
    for key, value in data.items():
        value = np.asarray(value)
        if value.ndim < 1:
            raise ValueError(f"{key}: expected at least one dimension")
        if value.dtype.kind not in "biuf":
            raise ValueError(f"{key}: unsupported dtype {value.dtype}")
        shapes[key] = list(value.shape[1:])
        arr = pa.array(np.ascontiguousarray(value).reshape(-1))
        for size in reversed(value.shape[1:]):
            if size <= 0:
                raise ValueError(f"{key}: empty dimensions are unsupported")
            arr = pa.FixedSizeListArray.from_arrays(arr, int(size))
        arrays[key] = arr
    table = pa.table(arrays)
    return table.replace_schema_metadata({b"mappo_shapes": json.dumps(shapes).encode(), b"format": b"mappo_parquet_v1"})


def table_to_arrays(table):
    metadata = table.schema.metadata or {}
    if b"mappo_shapes" not in metadata:
        raise ValueError("Parquet file is missing MAPPO shape metadata")
    shapes = json.loads(metadata[b"mappo_shapes"])
    result = {}
    for key in table.column_names:
        array = table[key].combine_chunks()
        for _ in shapes[key]:
            array = array.values
        result[key] = np.asarray(array.to_numpy(zero_copy_only=False)).reshape((len(table), *shapes[key]))
    return result


def iter_parquet(path, batch_size=8192):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield table_to_arrays(pa.Table.from_batches([batch], schema=pf.schema_arrow))


class ShardWriter:
    def __init__(self, parts_dir, max_uncompressed_bytes, on_upload=None):
        self.parts_dir = Path(parts_dir)
        self.limit = max_uncompressed_bytes
        self.on_upload = on_upload
        self.writer = None
        self.path = None
        self.shard_no = 0
        self.bytes = 0
        self.rows = 0
        self.parts = []

    def write(self, data):
        table = arrays_to_table(data)
        if self.writer is None:
            self.parts_dir.mkdir(parents=True, exist_ok=True)
            self.shard_no += 1
            self.path = self.parts_dir / f"part_{self.shard_no:06d}.parquet"
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)
        self.bytes += sum(np.asarray(v).nbytes for v in data.values())
        self.rows += len(table)

    def maybe_flush(self):
        # Called at episode boundaries so a trajectory is not split across shards.
        if self.writer is not None and self.bytes >= self.limit:
            self.flush()

    def flush(self):
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        path = self.path
        entry = {"path": str(path), "num_rows": self.rows, "size_bytes": path.stat().st_size}
        print(f"Saved Parquet shard {path} (rows={self.rows}, bytes={entry['size_bytes']})", flush=True)
        if self.on_upload is not None:
            self.on_upload(path)
        self.parts.append(entry)
        self.bytes = 0
        self.rows = 0


def write_embeddings(path, embeddings):
    pq.write_table(arrays_to_table({"embeddings": np.asarray(embeddings, dtype=np.float32)}), path, compression="zstd")


def read_embeddings(path):
    return table_to_arrays(pq.read_table(path))["embeddings"]
