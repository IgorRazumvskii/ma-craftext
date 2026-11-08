#!/usr/bin/env python3
"""Download one fixed HF trajectory snapshot, then stream SFT output to HF."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('SFT_DATA_DIR', '/data'))


def setting_bool(name: str, default: str) -> bool:
    value = os.environ.get(name, default).lower()
    if value not in ('0', '1'):
        raise ValueError(f'{name} must be 0 or 1')
    return value == '1'


def run() -> None:
    source = os.environ.get('SFT_SOURCE_REPO', 'igor69raz/ma-craftax-dataset')
    requested_revision = os.environ.get('SFT_SOURCE_REVISION', 'main')
    folder = os.environ.get('SFT_SOURCE_FOLDER', 'mappo_100k_20261101_210728_parts')
    output = os.environ.get('SFT_OUTPUT_REPO', '')
    if not os.environ.get('HF_TOKEN'):
        raise ValueError('HF_TOKEN is required')
    if not output or '/' not in output or output == source:
        raise ValueError('SFT_OUTPUT_REPO must be a new owner/name, different from the source')
    if not folder or PurePosixPath(folder).name != folder:
        raise ValueError('SFT_SOURCE_FOLDER must be a directory name, not a path')
    shard_mib = int(os.environ.get('SFT_SHARD_MIB', '64'))
    max_shards = int(os.environ.get('SFT_MAX_SHARDS', '0'))
    if shard_mib < 1 or max_shards < 0:
        raise ValueError('SFT_SHARD_MIB must be positive and SFT_MAX_SHARDS nonnegative')
    public = setting_bool('SFT_PUBLIC', '0')
    skip_unreadable = setting_bool('SFT_SKIP_UNREADABLE', '1')

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / 'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(DATA / 'tmp')
    api = HfApi()
    revision = api.repo_info(source, repo_type='dataset', revision=requested_revision).sha
    if not revision or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('HF returned an invalid source commit SHA')
    remote_files = api.list_repo_files(source, repo_type='dataset', revision=revision)
    parents = sorted({str(PurePosixPath(f).parent) for f in remote_files
                      if f.endswith('.parquet') and PurePosixPath(f).parent.name == folder})
    if len(parents) != 1:
        raise ValueError(f'Expected one source folder named {folder!r}; found {parents}')
    parent = parents[0]
    expected = {PurePosixPath(f).name for f in remote_files
                if str(PurePosixPath(f).parent) == parent and f.endswith('.parquet')}
    # A separate local directory per commit prevents leftover shards from older snapshots.
    snapshot_dir = DATA / 'snapshots' / revision
    print(f'Source {source} at {revision}: downloading {len(expected)} shards', flush=True)
    snapshot_download(repo_id=source, repo_type='dataset', revision=revision,
                      local_dir=snapshot_dir, allow_patterns=[f'{parent}/*.parquet'])
    parts = snapshot_dir / parent
    actual = {f.name for f in parts.glob('*.parquet')}
    if actual != expected:
        raise RuntimeError(f'Snapshot file mismatch: {len(expected - actual)} missing, {len(actual - expected)} unexpected')

    cmd = [sys.executable, str(ROOT / 'scripts/build_mappo_sft.py'),
           '--parts', str(parts), '--hf-repo-id', output,
           '--source-hf-repo-id', source, '--source-hf-revision', revision,
           '--hf-shard-mib', str(shard_mib), '--max-shards', str(max_shards)]
    if skip_unreadable:
        cmd.append('--skip-unreadable')
    if public:
        cmd.append('--hf-public')
    print(f'Converting to new dataset repository {output}', flush=True)
    subprocess.run(cmd, check=True)
    print(f'Completed: https://huggingface.co/datasets/{output}', flush=True)


if __name__ == '__main__':
    run()
