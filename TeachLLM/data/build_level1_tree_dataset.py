from __future__ import annotations
import argparse
import gzip
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
DEFAULT_TREE_TASK_NAMES = ['COLLECT_WOOD', 'PLACE_TABLE', 'MAKE_WOOD_PICKAXE', 'MAKE_WOOD_SWORD']
DEFAULT_LEVEL1_ACTIONS = ['NOOP', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'DO', 'PLACE_TABLE', 'MAKE_WOOD_PICKAXE', 'MAKE_WOOD_SWORD', 'REQUEST_WOOD', 'REQUEST_PICKAXE', 'REQUEST_SWORD', 'GIVE', 'GIVE_TO_PLAYER_1']
ACTION_NAME_TO_ID = {'NOOP': 0, 'LEFT': 1, 'RIGHT': 2, 'UP': 3, 'DOWN': 4, 'DO': 5, 'SLEEP': 6, 'PLACE_STONE': 7, 'PLACE_TABLE': 8, 'PLACE_FURNACE': 9, 'PLACE_PLANT': 10, 'MAKE_WOOD_PICKAXE': 11, 'MAKE_STONE_PICKAXE': 12, 'MAKE_IRON_PICKAXE': 13, 'MAKE_WOOD_SWORD': 14, 'MAKE_STONE_SWORD': 15, 'MAKE_IRON_SWORD': 16, 'REST': 17, 'DESCEND': 18, 'ASCEND': 19, 'MAKE_DIAMOND_PICKAXE': 20, 'MAKE_DIAMOND_SWORD': 21, 'MAKE_IRON_ARMOUR': 22, 'MAKE_DIAMOND_ARMOUR': 23, 'SHOOT_ARROW': 24, 'MAKE_ARROW': 25, 'CAST_SPELL': 26, 'PLACE_TORCH': 28, 'DRINK_POTION_RED': 29, 'DRINK_POTION_GREEN': 30, 'DRINK_POTION_BLUE': 31, 'DRINK_POTION_PINK': 32, 'DRINK_POTION_CYAN': 33, 'DRINK_POTION_YELLOW': 34, 'READ_BOOK': 35, 'ENCHANT_SWORD': 36, 'ENCHANT_ARMOUR': 37, 'MAKE_TORCH': 38, 'LEVEL_UP_DEXTERITY': 39, 'LEVEL_UP_STRENGTH': 40, 'LEVEL_UP_INTELLIGENCE': 41, 'ENCHANT_BOW': 42, 'REQUEST_FOOD': 43, 'REQUEST_DRINK': 44, 'REQUEST_WOOD': 45, 'REQUEST_STONE': 46, 'REQUEST_IRON': 47, 'REQUEST_COAL': 48, 'REQUEST_DIAMOND': 49, 'REQUEST_RUBY': 50, 'REQUEST_SAPPHIRE': 51, 'GIVE': 52, 'REQUEST_PICKAXE': 54, 'REQUEST_SWORD': 55, 'REQUEST_ARMOUR': 56, 'REQUEST_BOW': 57, 'REQUEST_ARROWS': 58, 'REQUEST_TORCHES': 59}
ACTION_ID_GIVE = 52
ACTION_ID_REQUEST_PICKAXE = 54
ALLOW_PLAYER_DESCEND = False

def _normalize_role(role: Any) -> str:
    s = str(role).strip().upper()
    if s == 'CRAFTER':
        return 'WARRIOR'
    return s

def _action_label_to_id(label: str) -> int | None:
    label_s = str(label).strip()
    if label_s.startswith('GIVE_TO_PLAYER_'):
        suffix = label_s.removeprefix('GIVE_TO_PLAYER_')
        idx = _to_int(suffix)
        if idx is None or idx < 1:
            return None
        return ACTION_ID_GIVE + idx
    return ACTION_NAME_TO_ID.get(label_s)

def _is_role_action_allowed(role: str, action_label: str) -> bool:
    action_id = _action_label_to_id(action_label)
    if action_id is None:
        return False
    common = {0, 1, 2, 3, 4, 5, 6, 17, 19}
    if ALLOW_PLAYER_DESCEND:
        common.add(18)
    give_allowed = ACTION_ID_GIVE <= action_id < ACTION_ID_REQUEST_PICKAXE
    common_allowed = action_id in common or give_allowed
    if common_allowed:
        return True
    role_n = _normalize_role(role)
    if role_n == 'MINER':
        return action_id == 54
    if role_n == 'WARRIOR':
        return action_id == 55
    if role_n == 'BUILDER':
        return action_id in {45, 46, 47, 48, 49, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
    return False

def _role_allowed_action_list(role: str, actions: list[str]) -> list[str]:
    return [a for a in actions if _is_role_action_allowed(role, a)]

def _smooth_within_role_target(count: int, role_min: int, strength: float) -> int:
    c = max(int(count), 1)
    m = max(int(role_min), 1)
    s = max(0.0, min(1.0, float(strength)))
    if s <= 0.0:
        return c
    if s >= 1.0:
        return m
    tgt = int(round(c ** (1.0 - s) * m ** s))
    return max(1, min(c, tgt))

def _open_in(path: Path):
    return gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else open(path, 'rt', encoding='utf-8')

def _open_out(path: Path):
    return gzip.open(path, 'wt', encoding='utf-8') if path.suffix == '.gz' else open(path, 'wt', encoding='utf-8')

def _iter_rows(paths: list[Path], *, max_input_rows: int=0) -> Iterable[dict[str, Any]]:
    seen = 0
    for p in paths:
        with _open_in(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                seen += 1
                if max_input_rows > 0 and seen > max_input_rows:
                    return
                yield row

def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None

def _rewrite_prompt_actions(prompt: str, allowed_actions: list[str]) -> str:
    marker = 'The possible list of actions you can take:'
    if marker not in prompt:
        return prompt
    lines = prompt.splitlines()
    marker_idx = None
    for i, line in enumerate(lines):
        if marker in line:
            marker_idx = i
            break
    if marker_idx is None:
        return prompt
    start = marker_idx + 1
    end = start
    while end < len(lines):
        s = lines[end].strip()
        if s.startswith('- '):
            end += 1
            continue
        break
    new_lines = list(lines[:start])
    new_lines.extend([f'- {a}' for a in allowed_actions])
    new_lines.extend(lines[end:])
    return '\n'.join(new_lines)

def main() -> None:
    parser = argparse.ArgumentParser(description='Build tree-only level-1 balanced dataset.')
    parser.add_argument('--input_dir', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt/out/expert_prompts_all_agents_gz'))
    parser.add_argument('--glob', type=str, default='*.jsonl.gz')
    parser.add_argument('--max_files', type=int, default=0, help='0 = all files')
    parser.add_argument('--max_input_rows', type=int, default=0, help='0 = all rows')
    parser.add_argument('--meta_json', type=Path, default=Path('/home/irazumovskii/project/MultyAgentCrafText/MA-Craftax/datasets/expert_mappo.meta.json'))
    parser.add_argument('--tree_task_names', type=str, default=','.join(DEFAULT_TREE_TASK_NAMES), help='Comma-separated task names to keep (matched via task_idx -> meta.task_names).')
    parser.add_argument('--level1_actions', type=str, default=','.join(DEFAULT_LEVEL1_ACTIONS), help='Comma-separated actions to keep and inject into prompt action list.')
    parser.add_argument('--balance_roles', action='store_true', help='Balance kept rows by role to min role count.')
    parser.add_argument('--balance_actions_within_role', action='store_true', default=True, help='Balance each role by action with smoothing (see --within_role_balance_strength).')
    parser.add_argument('--no_balance_actions_within_role', dest='balance_actions_within_role', action='store_false', help='Disable balance by (role, action).')
    parser.add_argument('--within_role_balance_strength', type=float, default=0.6, help='Smoothing strength for within-role action balancing in [0,1]: 0=no balancing, 1=hard min balancing.')
    parser.add_argument('--drop_forbidden_role_actions', action='store_true', default=True, help='Drop rows where action is forbidden for role by MA-Craftax specialization masks.')
    parser.add_argument('--no_drop_forbidden_role_actions', dest='drop_forbidden_role_actions', action='store_false', help='Disable role-action forbidden filtering.')
    parser.add_argument('--rewrite_prompt_actions_per_role', action='store_true', default=True, help='Rewrite action list in prompt to role-specific allowed actions.')
    parser.add_argument('--no_rewrite_prompt_actions_per_role', dest='rewrite_prompt_actions_per_role', action='store_false', help='Disable role-specific prompt action list rewriting.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rows_per_shard', type=int, default=200000)
    parser.add_argument('--output_dir', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt/out/expert_prompts_tree_level1_balanced'))
    args = parser.parse_args()
    inp = args.input_dir.expanduser().resolve()
    if not inp.is_dir():
        raise SystemExit(f'input_dir not found: {inp}')
    paths = sorted((x for x in inp.glob(args.glob) if x.is_file() and x.stat().st_size > 0))
    if args.max_files > 0:
        paths = paths[:args.max_files]
    if not paths:
        raise SystemExit(f'No files matched {args.glob!r} under {inp}')
    meta_path = args.meta_json.expanduser().resolve()
    if not meta_path.is_file():
        raise SystemExit(f'meta_json not found: {meta_path}')
    with meta_path.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    task_names = meta.get('task_names')
    if not isinstance(task_names, list) or not task_names:
        raise SystemExit('meta_json must contain non-empty task_names list')
    keep_task_names = {x.strip() for x in str(args.tree_task_names).split(',') if x.strip()}
    allowed_actions = [x.strip() for x in str(args.level1_actions).split(',') if x.strip()]
    allowed_action_set = set(allowed_actions)
    if not keep_task_names:
        raise SystemExit('tree_task_names is empty')
    if not allowed_actions:
        raise SystemExit('level1_actions is empty')
    role_counts = Counter()
    role_action_counts = Counter()
    counters = Counter()
    for row in _iter_rows(paths, max_input_rows=int(args.max_input_rows)):
        counters['rows_seen'] += 1
        task_idx = _to_int(row.get('task_idx'))
        role = row.get('agent_role')
        action = row.get('expert_action_label')
        if task_idx is None or role is None or action is None:
            counters['rows_missing_required'] += 1
            continue
        if task_idx < 0 or task_idx >= len(task_names):
            counters['rows_bad_task_idx'] += 1
            continue
        task_name = str(task_names[task_idx])
        if task_name not in keep_task_names:
            counters['rows_drop_task'] += 1
            continue
        action_s = str(action)
        if action_s not in allowed_action_set:
            counters['rows_drop_action'] += 1
            continue
        role_s = str(role)
        if args.drop_forbidden_role_actions and (not _is_role_action_allowed(role_s, action_s)):
            counters['rows_drop_forbidden_role_action'] += 1
            continue
        role_counts[role_s] += 1
        role_action_counts[role_s, action_s] += 1
        counters['rows_pass_filters'] += 1
    if not role_counts:
        raise SystemExit('No rows left after task/action filters.')
    target_per_role_action: dict[tuple[str, str], int] = {}
    if args.balance_actions_within_role:
        counts_by_role: dict[str, list[int]] = {}
        for (role_s, _action_s), cnt in role_action_counts.items():
            counts_by_role.setdefault(role_s, []).append(int(cnt))
        role_min_by_role = {role_s: min(cnts) for role_s, cnts in counts_by_role.items() if cnts}
        strength = max(0.0, min(1.0, float(args.within_role_balance_strength)))
        for (role_s, action_s), cnt in role_action_counts.items():
            role_min = role_min_by_role.get(role_s, int(cnt))
            target_per_role_action[role_s, action_s] = _smooth_within_role_target(int(cnt), int(role_min), strength)
    else:
        for key, cnt in role_action_counts.items():
            target_per_role_action[key] = int(cnt)
    target_per_role = None
    if args.balance_roles:
        role_target_candidates = Counter()
        for (role_s, _action_s), cnt in target_per_role_action.items():
            role_target_candidates[role_s] += int(cnt)
        if role_target_candidates:
            target_per_role = min(role_target_candidates.values())
    keep_prob_by_role: dict[str, float] = {}
    if target_per_role is not None:
        role_after_role_action = Counter()
        for (role, _), cnt in target_per_role_action.items():
            role_after_role_action[role] += int(cnt)
        for role, cnt in role_after_role_action.items():
            keep_prob_by_role[role] = min(1.0, float(target_per_role) / float(max(cnt, 1)))
    else:
        for role in role_counts:
            keep_prob_by_role[role] = 1.0
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    kept_by_role = Counter()
    shard_idx = 1
    rows_in_shard = 0
    out_path = out_dir / f'part_{shard_idx:06d}.jsonl.gz'
    out_f = _open_out(out_path)
    remaining_by_role_action = dict(role_action_counts)
    kept_by_role_action = Counter()

    def _rotate_shard() -> None:
        nonlocal shard_idx, rows_in_shard, out_path, out_f
        out_f.close()
        shard_idx += 1
        rows_in_shard = 0
        out_path = out_dir / f'part_{shard_idx:06d}.jsonl.gz'
        out_f = _open_out(out_path)
    for row in _iter_rows(paths, max_input_rows=int(args.max_input_rows)):
        task_idx = _to_int(row.get('task_idx'))
        role = row.get('agent_role')
        action = row.get('expert_action_label')
        prompt = row.get('prompt')
        if task_idx is None or role is None or action is None or (prompt is None):
            continue
        if task_idx < 0 or task_idx >= len(task_names):
            continue
        task_name = str(task_names[task_idx])
        if task_name not in keep_task_names:
            continue
        action_s = str(action)
        if action_s not in allowed_action_set:
            continue
        if args.drop_forbidden_role_actions and (not _is_role_action_allowed(str(role), action_s)):
            continue
        role_s = str(role)
        role_action_key = (role_s, action_s)
        quota = int(target_per_role_action.get(role_action_key, 0))
        remaining = int(remaining_by_role_action.get(role_action_key, 0))
        if remaining <= 0 or quota <= 0:
            continue
        remaining_need = max(quota - int(kept_by_role_action[role_action_key]), 0)
        if remaining_need <= 0:
            continue
        p_action = min(1.0, float(remaining_need) / float(remaining))
        remaining_by_role_action[role_action_key] = remaining - 1
        if rng.random() > p_action:
            continue
        p_role = keep_prob_by_role.get(role_s, 0.0)
        if p_role <= 0.0:
            continue
        if p_role < 1.0 and rng.random() > p_role:
            continue
        if target_per_role is not None and kept_by_role[role_s] >= int(target_per_role):
            continue
        out_row = dict(row)
        prompt_actions = allowed_actions
        if args.rewrite_prompt_actions_per_role:
            prompt_actions = _role_allowed_action_list(role_s, allowed_actions)
        out_row['prompt'] = _rewrite_prompt_actions(str(prompt), prompt_actions)
        out_row['task_name'] = task_name
        out_f.write(json.dumps(out_row, ensure_ascii=False) + '\n')
        rows_in_shard += 1
        kept_by_role[role_s] += 1
        kept_by_role_action[role_action_key] += 1
        counters['rows_written'] += 1
        if rows_in_shard >= int(args.rows_per_shard):
            _rotate_shard()
    out_f.close()
    if out_path.exists() and out_path.stat().st_size == 0:
        out_path.unlink()
    summary = {'input_dir': str(inp), 'glob': args.glob, 'files_used': len(paths), 'max_input_rows': int(args.max_input_rows), 'tree_task_names': sorted(keep_task_names), 'level1_actions': allowed_actions, 'balance_roles': bool(args.balance_roles), 'balance_actions_within_role': bool(args.balance_actions_within_role), 'within_role_balance_strength': float(args.within_role_balance_strength), 'drop_forbidden_role_actions': bool(args.drop_forbidden_role_actions), 'rewrite_prompt_actions_per_role': bool(args.rewrite_prompt_actions_per_role), 'seed': int(args.seed), 'role_counts_after_task_action_filter': dict(sorted(role_counts.items())), 'role_action_counts_after_filters': {f'{k[0]}::{k[1]}': int(v) for k, v in sorted(role_action_counts.items())}, 'target_per_role_action': {f'{k[0]}::{k[1]}': int(v) for k, v in sorted(target_per_role_action.items())}, 'target_per_role': int(target_per_role) if target_per_role is not None else None, 'keep_prob_by_role': keep_prob_by_role, 'kept_by_role': dict(sorted(kept_by_role.items())), 'kept_by_role_action': {f'{k[0]}::{k[1]}': int(v) for k, v in sorted(kept_by_role_action.items())}, 'counters': dict(counters), 'output_dir': str(out_dir)}
    summary_path = out_dir / 'build_summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Saved summary: {summary_path}')
if __name__ == '__main__':
    main()
