from __future__ import annotations
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

def _open_text(path: Path):
    return gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else open(path, 'rt', encoding='utf-8')

def _iter_rows(paths: list[Path], *, max_rows: int=0) -> Iterable[dict[str, Any]]:
    seen = 0
    for path in paths:
        with _open_text(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                seen += 1
                if max_rows > 0 and seen > max_rows:
                    return
                yield row

def _to_int(value: Any, *, default: int | None=None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default

def main() -> None:
    p = argparse.ArgumentParser(description='Plot expert trajectory and per-agent action distributions.')
    p.add_argument('--data_dir', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt/out/expert_prompts_all_agents_gz'), help='Directory with JSONL/JSONL.GZ shards.')
    p.add_argument('--glob', type=str, default='*.jsonl.gz')
    p.add_argument('--max_files', type=int, default=0, help='0 = all files.')
    p.add_argument('--max_rows', type=int, default=0, help='0 = all rows.')
    p.add_argument('--top_k_actions', type=int, default=20)
    p.add_argument('--output_dir', type=Path, default=Path('/home/irazumovskii/project/runs/expert-distributions'))
    p.add_argument('--title_prefix', type=str, default='Expert dataset')
    args = p.parse_args()
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise SystemExit('This script requires matplotlib and numpy in the active environment.') from exc
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f'data_dir is not a directory: {data_dir}')
    paths = sorted((x for x in data_dir.glob(args.glob) if x.is_file() and x.stat().st_size > 0))
    if args.max_files > 0:
        paths = paths[:args.max_files]
    if not paths:
        raise SystemExit(f'No files matched {args.glob!r} under {data_dir}')
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_row_count: dict[int, int] = defaultdict(int)
    traj_min_step: dict[int, int] = {}
    traj_max_step: dict[int, int] = {}
    traj_agents: dict[int, set[int]] = defaultdict(set)
    action_global: Counter[str] = Counter()
    action_by_agent: dict[int, Counter[str]] = defaultdict(Counter)
    rows_total = 0
    rows_missing_keys = 0
    for row in _iter_rows(paths, max_rows=int(args.max_rows)):
        rows_total += 1
        tid = _to_int(row.get('trajectory_id'))
        sid = _to_int(row.get('step_id'))
        aid = _to_int(row.get('agent_id'))
        act = row.get('expert_action_label')
        if tid is None or sid is None or aid is None or (act is None):
            rows_missing_keys += 1
            continue
        act_s = str(act)
        traj_row_count[tid] += 1
        traj_agents[tid].add(aid)
        if tid not in traj_min_step:
            traj_min_step[tid] = sid
            traj_max_step[tid] = sid
        else:
            if sid < traj_min_step[tid]:
                traj_min_step[tid] = sid
            if sid > traj_max_step[tid]:
                traj_max_step[tid] = sid
        action_global[act_s] += 1
        action_by_agent[aid][act_s] += 1
    if not traj_row_count:
        raise SystemExit('No usable rows with trajectory_id/step_id/agent_id/expert_action_label.')
    traj_lengths = []
    rows_per_step_values = []
    for tid, n_rows in traj_row_count.items():
        min_s = traj_min_step[tid]
        max_s = traj_max_step[tid]
        est_steps = max(1, int(max_s - min_s + 1))
        traj_lengths.append(est_steps)
        rows_per_step_values.append(float(n_rows) / float(est_steps))
    traj_lengths_np = np.asarray(traj_lengths, dtype=np.float64)
    rows_per_step_np = np.asarray(rows_per_step_values, dtype=np.float64)
    fig = plt.figure(figsize=(10, 5))
    bins = min(60, max(10, int(np.sqrt(len(traj_lengths_np)))))
    plt.hist(traj_lengths_np, bins=bins, color='#4C78A8', edgecolor='white')
    plt.title(f'{args.title_prefix}: trajectory length distribution')
    plt.xlabel('Estimated trajectory length (steps)')
    plt.ylabel('Number of trajectories')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    path_len_hist = out_dir / 'trajectory_length_hist.png'
    fig.savefig(path_len_hist, dpi=170)
    plt.close(fig)
    top_k = max(1, int(args.top_k_actions))
    agent_ids = sorted(action_by_agent.keys())
    fig, axes = plt.subplots(len(agent_ids), 1, figsize=(12, max(4, 3.0 * len(agent_ids))), squeeze=False)
    for i, aid in enumerate(agent_ids):
        ax = axes[i, 0]
        top = action_by_agent[aid].most_common(top_k)
        labels = [x[0] for x in top][::-1]
        values = [x[1] for x in top][::-1]
        ax.barh(labels, values, color='#72B7B2')
        ax.set_title(f'agent_id={aid}: top-{top_k} actions (count)')
        ax.set_xlabel('Count')
        ax.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    path_agent_counts = out_dir / 'action_distribution_by_agent.png'
    fig.savefig(path_agent_counts, dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(len(agent_ids), 1, figsize=(12, max(4, 3.0 * len(agent_ids))), squeeze=False)
    for i, aid in enumerate(agent_ids):
        ax = axes[i, 0]
        top = action_by_agent[aid].most_common(top_k)
        total_a = sum(action_by_agent[aid].values())
        labels = [x[0] for x in top][::-1]
        shares = [float(x[1]) / max(total_a, 1) for x in top][::-1]
        ax.barh(labels, shares, color='#F28E2B')
        ax.set_title(f'agent_id={aid}: top-{top_k} actions (share)')
        ax.set_xlabel('Share')
        ax.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    path_agent_shares = out_dir / 'action_distribution_by_agent_share.png'
    fig.savefig(path_agent_shares, dpi=170)
    plt.close(fig)
    summary = {'data_dir': str(data_dir), 'glob': args.glob, 'files_used': len(paths), 'rows_total': int(rows_total), 'rows_missing_required_keys': int(rows_missing_keys), 'num_trajectories': int(len(traj_row_count)), 'trajectory_length_steps': {'min': int(np.min(traj_lengths_np)), 'max': int(np.max(traj_lengths_np)), 'mean': float(np.mean(traj_lengths_np)), 'p50': float(np.percentile(traj_lengths_np, 50)), 'p90': float(np.percentile(traj_lengths_np, 90)), 'p95': float(np.percentile(traj_lengths_np, 95)), 'p99': float(np.percentile(traj_lengths_np, 99))}, 'rows_per_step_estimated': {'min': float(np.min(rows_per_step_np)), 'max': float(np.max(rows_per_step_np)), 'mean': float(np.mean(rows_per_step_np)), 'p50': float(np.percentile(rows_per_step_np, 50)), 'p90': float(np.percentile(rows_per_step_np, 90))}, 'action_global_top': action_global.most_common(top_k), 'action_by_agent_top': {str(aid): action_by_agent[aid].most_common(top_k) for aid in agent_ids}, 'plots': {'trajectory_length_hist': str(path_len_hist), 'action_distribution_by_agent': str(path_agent_counts), 'action_distribution_by_agent_share': str(path_agent_shares)}}
    summary_path = out_dir / 'distribution_summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Saved summary: {summary_path}')
if __name__ == '__main__':
    main()
