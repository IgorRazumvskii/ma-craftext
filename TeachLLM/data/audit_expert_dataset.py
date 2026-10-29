from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
REQUIRED_COLUMNS = ('prompt', 'expert_action_label')
OPTIONAL_COLUMNS = ('agent_id', 'agent_role', 'task_idx', 'instruction_idx', 'trajectory_id', 'step_id')

@dataclass
class RunningStats:
    n: int = 0
    sum_v: float = 0.0
    min_v: float = math.inf
    max_v: float = -math.inf
    values_for_quantiles: list[float] | None = None

    def add(self, v: float) -> None:
        self.n += 1
        self.sum_v += float(v)
        self.min_v = min(self.min_v, float(v))
        self.max_v = max(self.max_v, float(v))
        if self.values_for_quantiles is not None:
            self.values_for_quantiles.append(float(v))

    def summary(self) -> dict[str, float]:
        if self.n == 0:
            return {'count': 0}
        out = {'count': float(self.n), 'mean': self.sum_v / float(self.n), 'min': self.min_v, 'max': self.max_v}
        if self.values_for_quantiles:
            vals = sorted(self.values_for_quantiles)
            out['p50'] = _percentile(vals, 50.0)
            out['p90'] = _percentile(vals, 90.0)
            out['p95'] = _percentile(vals, 95.0)
            out['p99'] = _percentile(vals, 99.0)
        return out

def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (p / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_values[lo])
    frac = rank - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)

def _iter_jsonl(paths: list[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    yield (path, line_idx, {'__json_error__': str(exc), '__raw_line__': line})
                    continue
                if not isinstance(row, dict):
                    yield (path, line_idx, {'__type_error__': f'row type is {type(row).__name__}'})
                    continue
                yield (path, line_idx, row)

def _as_key(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[Any, ...] | None:
    vals = []
    for k in keys:
        if k not in row:
            return None
        vals.append(row.get(k))
    return tuple(vals)

def _maybe_load_tokenizer(tokenizer_name_or_path: str | None):
    if not tokenizer_name_or_path:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise SystemExit('Tokenizer analysis requested but transformers is unavailable. Install transformers or run without --tokenizer.') from exc
    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    return tok

def _recommendations(*, total_rows: int, action_counts: Counter[str], invalid_rows: int, empty_prompts: int, empty_actions: int, missing_required_rows: int, duplicate_key_rows: int, prompt_hash_dupes: int, role_counts: Counter[str]) -> list[str]:
    recs: list[str] = []
    if total_rows == 0:
        recs.append('Dataset has zero usable rows.')
        return recs
    if invalid_rows > 0:
        recs.append(f'Found {invalid_rows} invalid JSON/type rows; clean shards before training.')
    if missing_required_rows > 0:
        recs.append(f'{missing_required_rows} rows missing required columns {REQUIRED_COLUMNS}.')
    if empty_prompts > 0:
        recs.append(f'{empty_prompts} rows have empty prompts.')
    if empty_actions > 0:
        recs.append(f'{empty_actions} rows have empty expert_action_label.')
    if duplicate_key_rows > 0:
        recs.append(f'{duplicate_key_rows} duplicate (trajectory_id, step_id, agent_id) rows detected; possible repeated samples.')
    if prompt_hash_dupes > 0:
        recs.append(f'{prompt_hash_dupes} duplicate prompt hashes detected; check over-repetition and leakage risk.')
    if len(action_counts) <= 1:
        recs.append('Only one unique action label found; policy collapse is likely.')
    elif len(action_counts) < 10:
        recs.append(f'Only {len(action_counts)} unique actions found; ensure export includes full action diversity.')
    if action_counts:
        max_count = max(action_counts.values())
        min_count = min(action_counts.values())
        if min_count > 0 and max_count / float(min_count) > 50.0:
            recs.append('Action imbalance is very high (>50x max/min). Consider stronger balancing or weighted sampling.')
    if role_counts and len(role_counts) < 3:
        recs.append(f'Only {len(role_counts)} unique roles found; multi-agent behavior may not generalize.')
    if not recs:
        recs.append('No critical data-quality issues detected by this audit.')
    return recs

def main() -> None:
    p = argparse.ArgumentParser(description='Audit expert JSONL(.gz) dataset for SFT readiness.')
    p.add_argument('--data_dir', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt/out/expert_prompts_all_agents_gz'), help='Directory with jsonl/jsonl.gz shards.')
    p.add_argument('--glob', type=str, default='*.jsonl.gz')
    p.add_argument('--max_files', type=int, default=0, help='0 = all.')
    p.add_argument('--max_rows', type=int, default=0, help='0 = scan all rows. Useful for quick smoke checks.')
    p.add_argument('--top_k', type=int, default=30, help='How many top actions/tasks/roles to print.')
    p.add_argument('--tokenizer', type=str, default=None, help='Optional HF tokenizer name/path to measure prompt token lengths.')
    p.add_argument('--quantiles_sample_rows', type=int, default=250000, help='Rows to store for exact quantiles (0 = all rows).')
    p.add_argument('--seed', type=int, default=42, help='Seed for any sampling logic.')
    p.add_argument('--output_json', type=Path, default=None, help='Optional path to save full report as JSON.')
    args = p.parse_args()
    rng = random.Random(int(args.seed))
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f'data_dir is not a directory: {data_dir}')
    files = sorted([x for x in data_dir.glob(args.glob) if x.is_file() and x.stat().st_size > 0])
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise SystemExit(f'No files matched {args.glob!r} under {data_dir}')
    tokenizer = _maybe_load_tokenizer(args.tokenizer)
    total_rows = 0
    invalid_rows = 0
    missing_required_rows = 0
    empty_prompts = 0
    empty_actions = 0
    files_scanned = 0
    rows_per_file: dict[str, int] = {}
    action_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    instruction_counts: Counter[str] = Counter()
    agent_id_counts: Counter[str] = Counter()
    action_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    action_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    prompt_char_stats = RunningStats(values_for_quantiles=[])
    prompt_tok_stats = RunningStats(values_for_quantiles=[]) if tokenizer is not None else None
    dup_key_seen: set[tuple[Any, ...]] = set()
    duplicate_key_rows = 0
    prompt_hash_seen: set[str] = set()
    prompt_hash_dupes = 0
    example_bad_rows: list[dict[str, Any]] = []
    quantile_cap = int(args.quantiles_sample_rows)
    seen_for_quantile_sampling = 0
    for path, line_idx, row in _iter_jsonl(files):
        if rows_per_file.get(path.name) is None:
            rows_per_file[path.name] = 0
            files_scanned += 1
        rows_per_file[path.name] += 1
        if args.max_rows > 0 and total_rows >= int(args.max_rows):
            break
        total_rows += 1
        if '__json_error__' in row or '__type_error__' in row:
            invalid_rows += 1
            if len(example_bad_rows) < 20:
                example_bad_rows.append({'file': path.name, 'line': line_idx, 'error': row.get('__json_error__') or row.get('__type_error__')})
            continue
        missing_here = [c for c in REQUIRED_COLUMNS if c not in row]
        if missing_here:
            missing_required_rows += 1
            if len(example_bad_rows) < 20:
                example_bad_rows.append({'file': path.name, 'line': line_idx, 'error': f'missing columns {missing_here}'})
            continue
        prompt = str(row.get('prompt', ''))
        action = str(row.get('expert_action_label', ''))
        role = str(row.get('agent_role', 'unknown'))
        task = str(row.get('task_idx', 'unknown'))
        instr = str(row.get('instruction_idx', 'unknown'))
        agent_id = str(row.get('agent_id', 'unknown'))
        if not prompt.strip():
            empty_prompts += 1
        if not action.strip():
            empty_actions += 1
        action_counts[action] += 1
        role_counts[role] += 1
        task_counts[task] += 1
        instruction_counts[instr] += 1
        agent_id_counts[agent_id] += 1
        action_by_role[role][action] += 1
        action_by_task[task][action] += 1
        char_len = len(prompt)
        if prompt_char_stats.values_for_quantiles is not None:
            vals = prompt_char_stats.values_for_quantiles
            if quantile_cap <= 0 or len(vals) < quantile_cap:
                prompt_char_stats.add(char_len)
            else:
                prompt_char_stats.n += 1
                prompt_char_stats.sum_v += float(char_len)
                prompt_char_stats.min_v = min(prompt_char_stats.min_v, float(char_len))
                prompt_char_stats.max_v = max(prompt_char_stats.max_v, float(char_len))
                seen_for_quantile_sampling += 1
                j = rng.randrange(seen_for_quantile_sampling + quantile_cap)
                if j < quantile_cap:
                    vals[j] = float(char_len)
        else:
            prompt_char_stats.add(char_len)
        if tokenizer is not None and prompt_tok_stats is not None:
            tok_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            tok_vals = prompt_tok_stats.values_for_quantiles
            if tok_vals is not None:
                if quantile_cap <= 0 or len(tok_vals) < quantile_cap:
                    prompt_tok_stats.add(tok_len)
                else:
                    prompt_tok_stats.n += 1
                    prompt_tok_stats.sum_v += float(tok_len)
                    prompt_tok_stats.min_v = min(prompt_tok_stats.min_v, float(tok_len))
                    prompt_tok_stats.max_v = max(prompt_tok_stats.max_v, float(tok_len))
                    j = rng.randrange(seen_for_quantile_sampling + quantile_cap)
                    if j < quantile_cap:
                        tok_vals[j] = float(tok_len)
            else:
                prompt_tok_stats.add(tok_len)
        key = _as_key(row, ('trajectory_id', 'step_id', 'agent_id'))
        if key is not None:
            if key in dup_key_seen:
                duplicate_key_rows += 1
            else:
                dup_key_seen.add(key)
        prompt_hash = hashlib.sha1(prompt.encode('utf-8')).hexdigest()
        if prompt_hash in prompt_hash_seen:
            prompt_hash_dupes += 1
        else:
            prompt_hash_seen.add(prompt_hash)
    top_k = max(int(args.top_k), 1)
    actions_top = action_counts.most_common(top_k)
    roles_top = role_counts.most_common(top_k)
    tasks_top = task_counts.most_common(top_k)
    action_probs = [cnt / float(max(total_rows, 1)) for _, cnt in action_counts.items()]
    action_entropy = -sum((p * math.log(max(p, 1e-12), 2) for p in action_probs)) if action_probs else 0.0
    max_action_share = max(action_counts.values()) / float(max(total_rows, 1)) if action_counts else 0.0
    min_action_share = min(action_counts.values()) / float(max(total_rows, 1)) if action_counts else 0.0
    recs = _recommendations(total_rows=total_rows, action_counts=action_counts, invalid_rows=invalid_rows, empty_prompts=empty_prompts, empty_actions=empty_actions, missing_required_rows=missing_required_rows, duplicate_key_rows=duplicate_key_rows, prompt_hash_dupes=prompt_hash_dupes, role_counts=role_counts)
    report = {'dataset_path': str(data_dir), 'glob': args.glob, 'files_scanned': files_scanned, 'rows_scanned': total_rows, 'rows_per_file_top': sorted(rows_per_file.items(), key=lambda kv: kv[1], reverse=True)[:top_k], 'required_columns': list(REQUIRED_COLUMNS), 'optional_columns': list(OPTIONAL_COLUMNS), 'quality_issues': {'invalid_rows': invalid_rows, 'missing_required_rows': missing_required_rows, 'empty_prompts': empty_prompts, 'empty_actions': empty_actions, 'duplicate_key_rows': duplicate_key_rows, 'duplicate_prompt_hash_rows': prompt_hash_dupes}, 'length_stats': {'prompt_chars': prompt_char_stats.summary(), 'prompt_tokens': prompt_tok_stats.summary() if prompt_tok_stats is not None else None}, 'distributions': {'unique_actions': len(action_counts), 'unique_roles': len(role_counts), 'unique_tasks': len(task_counts), 'unique_instructions': len(instruction_counts), 'action_entropy_bits': action_entropy, 'max_action_share': max_action_share, 'min_action_share': min_action_share, 'actions_top': actions_top, 'roles_top': roles_top, 'tasks_top': tasks_top, 'agent_id_top': agent_id_counts.most_common(top_k)}, 'action_by_role_top': {role: cnts.most_common(min(15, top_k)) for role, cnts in sorted(action_by_role.items())}, 'action_by_task_top': {task: cnts.most_common(min(10, top_k)) for task, cnts in sorted(action_by_task.items(), key=lambda kv: task_counts.get(kv[0], 0), reverse=True)[:min(20, top_k)]}, 'example_bad_rows': example_bad_rows, 'recommendations': recs}
    print('=' * 88)
    print('EXPERT DATASET AUDIT')
    print('=' * 88)
    print(f'Dataset: {data_dir}')
    print(f'Files scanned: {files_scanned}')
    print(f'Rows scanned: {total_rows}')
    print()
    print('Quality issues:')
    for k, v in report['quality_issues'].items():
        print(f'- {k}: {v}')
    print()
    print('Prompt length stats:')
    print(f"- chars: {report['length_stats']['prompt_chars']}")
    if report['length_stats']['prompt_tokens'] is not None:
        print(f"- tokens: {report['length_stats']['prompt_tokens']}")
    else:
        print('- tokens: not computed (use --tokenizer)')
    print()
    print('Distribution snapshot:')
    print(f"- unique actions/roles/tasks: {report['distributions']['unique_actions']} / {report['distributions']['unique_roles']} / {report['distributions']['unique_tasks']}")
    print(f"- action entropy (bits): {report['distributions']['action_entropy_bits']:.3f}; max/min action share: {report['distributions']['max_action_share']:.4f} / {report['distributions']['min_action_share']:.4f}")
    print('- top actions:')
    for act, cnt in actions_top[:min(15, len(actions_top))]:
        print(f'  {act}: {cnt}')
    print()
    print('Recommendations:')
    for r in recs:
        print(f'- {r}')
    print('=' * 88)
    if args.output_json is not None:
        out = args.output_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'Saved report JSON: {out}')
if __name__ == '__main__':
    main()
