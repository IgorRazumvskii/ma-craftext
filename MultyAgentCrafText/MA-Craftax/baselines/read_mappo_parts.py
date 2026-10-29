"""Read v2 MAPPO NPZ blocks, optionally restoring full actor observations."""
import json
from pathlib import Path
import numpy as np


def iter_parts(meta_path, restore_obs=True):
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    embeddings = None
    if meta.get("compact_observations") and restore_obs:
        # Embeddings are next to the metadata; supports moving the whole dataset.
        name = Path(meta["embeddings_path"]).name
        with np.load(meta_path.parent / name, allow_pickle=False) as f:
            embeddings = f["embeddings"]
    parts_dir = meta_path.parent / Path(meta["parts_dir"]).name
    for entry in meta["parts"]:
        with np.load(parts_dir / Path(entry["path"]).name, allow_pickle=False) as f:
            data = {key: f[key] for key in f.files}
        if embeddings is not None:
            values = embeddings[data["instruction_idx"]]
            values = np.broadcast_to(values[:, None, :], (*data["obs"].shape[:2], values.shape[-1]))
            data["obs"] = np.concatenate((data["obs"], values), axis=-1)
        yield data
