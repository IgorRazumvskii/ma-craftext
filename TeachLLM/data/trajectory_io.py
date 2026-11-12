"""Read MAPPO Parquet schema and scenario definitions without importing JAX."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import numpy as np
import pyarrow as pa


def table_to_arrays(table):
    metadata = table.schema.metadata or {}
    if b'mappo_shapes' not in metadata or metadata.get(b'format') != b'mappo_parquet_v1':
        raise ValueError('Parquet is missing MAPPO v1 shape metadata')
    shapes = json.loads(metadata[b'mappo_shapes'])
    result = {}
    for key in table.column_names:
        array = table[key].combine_chunks()
        for _ in shapes[key]:
            array = array.values
        result[key] = np.asarray(array.to_numpy(zero_copy_only=False)).reshape((len(table), *shapes[key]))
    return result


def arrays_to_table(data):
    """Construct a MAPPO v1 table; useful for small fixtures."""
    arrays, shapes = {}, {}
    for name, value in data.items():
        value = np.asarray(value)
        shapes[name] = list(value.shape[1:])
        array = pa.array(np.ascontiguousarray(value).reshape(-1))
        for size in reversed(value.shape[1:]):
            array = pa.FixedSizeListArray.from_arrays(array, int(size))
        arrays[name] = array
    return pa.table(arrays).replace_schema_metadata({b'mappo_shapes': json.dumps(shapes).encode(), b'format': b'mappo_parquet_v1'})


def load_scenarios(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    easy = next((node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'easy' for t in node.targets)), None)
    if not isinstance(easy, ast.Dict):
        raise ValueError(f'Cannot find easy task dictionary in {path}')
    tasks, instructions, mapping = [], [], []
    for task_id, (key, value) in enumerate(zip(easy.keys, easy.values)):
        name = ast.literal_eval(key)
        if not isinstance(value, ast.Dict):
            raise ValueError(f'Task {name} is not a dictionary')
        fields = {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(value.keys, value.values)
                  if isinstance(k, ast.Constant) and k.value in ('instruction', 'instruction_paraphrases')}
        variants = [fields['instruction']] + fields.get('instruction_paraphrases', [])
        if not all(isinstance(text, str) for text in variants):
            raise ValueError(f'Unexpected text for task {name}')
        tasks.append(name); instructions.extend(variants); mapping.extend([task_id] * len(variants))
    return tasks, instructions, mapping


def read_layout(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    classes = {}; dimensions = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'OBS_DIM' for t in node.targets):
            dimensions = ast.literal_eval(node.value)
        if isinstance(node, ast.ClassDef) and node.name in {'BlockType', 'ItemType', 'Action'}:
            members = {}
            for assignment in node.body:
                if isinstance(assignment, ast.Assign) and len(assignment.targets)==1 and isinstance(assignment.targets[0], ast.Name):
                    try: value = ast.literal_eval(assignment.value)
                    except (ValueError, TypeError): continue
                    if isinstance(value, int): members[value] = assignment.targets[0].id
            classes[node.name] = members
    if dimensions != (9,11) or set(classes) != {'BlockType','ItemType','Action'}:
        raise ValueError('Unsupported Craftax symbolic observation layout')
    for name in ('BlockType', 'ItemType'):
        if set(classes[name]) != set(range(len(classes[name]))):
            raise ValueError(f'Non-contiguous {name}')
    return dimensions, classes
