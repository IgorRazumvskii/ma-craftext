#!/usr/bin/env python3
"""Download one fixed HF trajectory snapshot, then stream SFT output to HF."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

from huggingface_hub import HfApi, snapshot_download
from requests.exceptions import HTTPError

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('SFT_DATA_DIR', '/data'))


def setting_bool(name: str, default: str) -> bool:
    value = os.environ.get(name, default).lower()
    if value not in ('0', '1'):
        raise ValueError(f'{name} must be 0 or 1')
    return value == '1'


def retry_rate_limit(operation, description: str):
    """Wait out HF's five-minute API window; keep completed local downloads."""
    attempts = 12
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except HTTPError as exc:
            response = exc.response
            if response is None or response.status_code != 429 or attempt == attempts:
                raise
            try:
                retry_after = int(response.headers.get('Retry-After', '0'))
            except ValueError:
                retry_after = 0
            delay = max(310, retry_after)
            print(f'HF rate limit during {description}; waiting {delay}s before '
                  f'retry {attempt + 1}/{attempts}', file=sys.stderr, flush=True)
            time.sleep(delay)


def run() -> None:
    source = os.environ.get('SFT_SOURCE_REPO', 'igor69raz/ma-craftax-dataset')
    requested_revision = os.environ.get('SFT_SOURCE_REVISION', 'main')
    folder = os.environ.get('SFT_SOURCE_FOLDER', 'mappo_100k_20261101_210728/parts')
    output = os.environ.get('SFT_OUTPUT_REPO', '')
    if not os.environ.get('HF_TOKEN'):
        raise ValueError('HF_TOKEN is required')
    if not output or '/' not in output or output == source:
        raise ValueError('SFT_OUTPUT_REPO must be a new owner/name, different from the source')
    folder_path = PurePosixPath(folder)
    if (not folder or folder_path.is_absolute() or '..' in folder_path.parts
            or any(ch in folder for ch in '*?[')):
        raise ValueError('SFT_SOURCE_FOLDER must be a relative folder path without glob patterns')
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
    revision = retry_rate_limit(
        lambda: api.repo_info(source, repo_type='dataset', revision=requested_revision),
        'reading source revision').sha
    if not revision or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('HF returned an invalid source commit SHA')
    remote_files = retry_rate_limit(
        lambda: api.list_repo_files(source, repo_type='dataset', revision=revision),
        'listing source files')
    parents = sorted({str(PurePosixPath(f).parent) for f in remote_files
                      if f.endswith('.parquet') and (
                          str(PurePosixPath(f).parent) == str(folder_path)
                          if len(folder_path.parts) > 1
                          else PurePosixPath(f).parent.name == folder)})
    if len(parents) != 1:
        parquet_dirs = sorted({str(PurePosixPath(f).parent) for f in remote_files if f.endswith('.parquet')})
        raise ValueError(f'Expected one source folder named {folder!r}; found {parents}. '
                         f'Parquet folders in {source}: {parquet_dirs}')
    parent = parents[0]
    expected = {PurePosixPath(f).name for f in remote_files
                if str(PurePosixPath(f).parent) == parent and f.endswith('.parquet')}
    # A separate local directory per commit prevents leftover shards from older snapshots.
    snapshot_dir = DATA / 'snapshots' / revision
    print(f'Source {source} at {revision}: downloading {len(expected)} shards', flush=True)
    retry_rate_limit(
        lambda: snapshot_download(repo_id=source, repo_type='dataset', revision=revision,
                                  local_dir=snapshot_dir, allow_patterns=[f'{parent}/*.parquet'],
                                  max_workers=2),
        'downloading source files')
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
