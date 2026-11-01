"""Iterate over MAPPO Parquet shards or legacy NPZ parts as NumPy arrays."""
import json
from pathlib import Path
import numpy as np


def iter_parts(meta_path, restore_obs=True):
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    embeddings = None
    if meta.get("compact_observations") and restore_obs:
        name = Path(meta["embeddings_path"]).name
        embedding_path = meta_path.parent / name
        if embedding_path.suffix == ".parquet":
            if __package__:
                from .trajectory_parquet import read_embeddings
            else:
                from trajectory_parquet import read_embeddings
            embeddings = read_embeddings(embedding_path)
        else:
            with np.load(embedding_path, allow_pickle=False) as f:
                embeddings = f["embeddings"]
    parts_dir = meta_path.parent / Path(meta["parts_dir"]).name
    for entry in meta["parts"]:
        path = parts_dir / Path(entry["path"]).name
        if path.suffix == ".parquet":
            if __package__:
                from .trajectory_parquet import iter_parquet
            else:
                from trajectory_parquet import iter_parquet
            blocks = iter_parquet(path)
        else:
            with np.load(path, allow_pickle=False) as f:
                blocks = [{key: f[key] for key in f.files}]
        for data in blocks:
            if embeddings is not None:
                values = embeddings[data["instruction_idx"]]
                values = np.broadcast_to(values[:, None, :], (*data["obs"].shape[:2], values.shape[-1]))
                data["obs"] = np.concatenate((data["obs"], values), axis=-1)
            yield data
