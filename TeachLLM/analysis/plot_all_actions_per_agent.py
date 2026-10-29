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

def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None

def _sanitize_filename(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ('-', '_'):
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out)

def main() -> None:
    parser = argparse.ArgumentParser(description='Plot all action labels per agent_id.')
    parser.add_argument('--data_dir', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt/out/expert_prompts_all_agents_gz'), help='Directory with JSONL/JSONL.GZ shards.')
    parser.add_argument('--glob', type=str, default='*.jsonl.gz')
    parser.add_argument('--max_files', type=int, default=0, help='0 = all files')
    parser.add_argument('--max_rows', type=int, default=0, help='0 = all rows')
    parser.add_argument('--output_dir', type=Path, default=Path('/home/irazumovskii/project/runs/all-actions-per-agent'))
    parser.add_argument('--normalize', action='store_true', help='Plot shares instead of raw counts.')
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit('This script requires matplotlib in the active environment.') from exc
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
    action_global: Counter[str] = Counter()
    action_by_agent: dict[int, Counter[str]] = defaultdict(Counter)
    role_by_agent: dict[int, Counter[str]] = defaultdict(Counter)
    rows_total = 0
    rows_used = 0
    rows_missing = 0
    for row in _iter_rows(paths, max_rows=int(args.max_rows)):
        rows_total += 1
        aid = _to_int(row.get('agent_id'))
        action = row.get('expert_action_label')
        role = row.get('agent_role')
        if aid is None or action is None:
            rows_missing += 1
            continue
        action_s = str(action)
        action_global[action_s] += 1
        action_by_agent[aid][action_s] += 1
        if role is not None:
            role_by_agent[aid][str(role)] += 1
        rows_used += 1
    if not action_global:
        raise SystemExit('No usable rows with agent_id + expert_action_label.')
    all_actions = [k for k, _ in action_global.most_common()]
    agent_ids = sorted(action_by_agent.keys())
    saved_plots: dict[str, str] = {}
    for aid in agent_ids:
        counter = action_by_agent[aid]
        values = [counter.get(action, 0) for action in all_actions]
        total = max(sum(values), 1)
        y_values = [v / total for v in values] if args.normalize else values
        fig, ax = plt.subplots(figsize=(20, 6))
        x = list(range(len(all_actions)))
        ax.bar(x, y_values, color='#4C78A8')
        ax.set_xticks(x)
        ax.set_xticklabels(all_actions, rotation=90)
        ylabel = 'Share' if args.normalize else 'Count'
        ax.set_ylabel(ylabel)
        role_label = 'unknown'
        if aid in role_by_agent and role_by_agent[aid]:
            role_label = role_by_agent[aid].most_common(1)[0][0]
        ax.set_title(f'agent_id={aid} role={role_label}: all actions ({ylabel.lower()})')
        ax.grid(axis='y', alpha=0.25)
        plt.tight_layout()
        suffix = 'share' if args.normalize else 'count'
        filename = f'agent_{aid}_{_sanitize_filename(role_label)}_all_actions_{suffix}.png'
        plot_path = out_dir / filename
        fig.savefig(plot_path, dpi=170)
        plt.close(fig)
        saved_plots[str(aid)] = str(plot_path)
    overall_values = [action_global.get(action, 0) for action in all_actions]
    overall_total = max(sum(overall_values), 1)
    overall_y = [v / overall_total for v in overall_values] if args.normalize else overall_values
    fig, ax = plt.subplots(figsize=(20, 6))
    x = list(range(len(all_actions)))
    ax.bar(x, overall_y, color='#F28E2B')
    ax.set_xticks(x)
    ax.set_xticklabels(all_actions, rotation=90)
    ylabel = 'Share' if args.normalize else 'Count'
    ax.set_ylabel(ylabel)
    ax.set_title(f'overall: all actions ({ylabel.lower()})')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    overall_plot_path = out_dir / f"overall_all_actions_{('share' if args.normalize else 'count')}.png"
    fig.savefig(overall_plot_path, dpi=170)
    plt.close(fig)
    summary = {'data_dir': str(data_dir), 'glob': args.glob, 'files_used': len(paths), 'rows_total': rows_total, 'rows_used': rows_used, 'rows_missing_required': rows_missing, 'num_agents': len(agent_ids), 'num_unique_actions': len(all_actions), 'all_actions_order': all_actions, 'agent_totals': {str(aid): int(sum(action_by_agent[aid].values())) for aid in agent_ids}, 'agent_role_mode': {str(aid): role_by_agent[aid].most_common(1)[0][0] if role_by_agent[aid] else None for aid in agent_ids}, 'plots_by_agent': saved_plots, 'overall': {'total_rows': int(sum(action_global.values())), 'actions_count': action_global.most_common(), 'actions_share': [[k, float(v) / float(max(sum(action_global.values()), 1))] for k, v in action_global.most_common()], 'plot': str(overall_plot_path)}}
    summary_path = out_dir / 'summary_all_actions_per_agent.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Saved summary: {summary_path}')
if __name__ == '__main__':
    main()
