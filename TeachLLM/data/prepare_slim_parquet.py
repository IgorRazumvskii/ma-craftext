from __future__ import annotations
import argparse
from pathlib import Path

def list_files(data_dir: Path, pattern: str, exclude_prefix: str) -> list[Path]:
    out: list[Path] = []
    for p in sorted(data_dir.glob(pattern)):
        if not p.is_file():
            continue
        if p.name.startswith('.'):
            continue
        if exclude_prefix and p.name.startswith(exclude_prefix):
            continue
        if p.stat().st_size == 0:
            continue
        out.append(p)
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description='Convert JSONL shards into slim parquet shards.')
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--glob', type=str, default='*.jsonl')
    parser.add_argument('--exclude_prefix', type=str, default='.')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--keep_columns', type=str, default='prompt,expert_action_label', help='Comma-separated columns to keep.')
    parser.add_argument('--max_files', type=int, default=0, help='Process first N files only (0=all).')
    parser.add_argument('--datasets_cache_dir', type=Path, default=None)
    parser.add_argument('--keep_in_memory', action='store_true')
    args = parser.parse_args()
    from datasets import Dataset
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f'input_dir is not a directory: {input_dir}')
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list_files(input_dir, args.glob, args.exclude_prefix)
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise SystemExit('No input files matched.')
    keep_columns = [c.strip() for c in args.keep_columns.split(',') if c.strip()]
    if not keep_columns:
        raise SystemExit('--keep_columns is empty.')
    load_kw: dict = {}
    if args.datasets_cache_dir is not None:
        dc = args.datasets_cache_dir.expanduser().resolve()
        dc.mkdir(parents=True, exist_ok=True)
        load_kw['cache_dir'] = str(dc)
    if args.keep_in_memory:
        load_kw['keep_in_memory'] = True
    for i, fp in enumerate(files, start=1):
        print(f'[{i}/{len(files)}] {fp.name}')
        ds = Dataset.from_json(str(fp), **load_kw)
        present = [c for c in keep_columns if c in ds.column_names]
        if not present:
            print('  skip: none of requested columns present')
            continue
        ds = ds.select_columns(present)
        out_path = output_dir / f'{fp.stem}.parquet'
        ds.to_parquet(str(out_path))
        print(f'  -> {out_path.name}')
if __name__ == '__main__':
    main()
