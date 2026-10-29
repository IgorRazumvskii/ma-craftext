from __future__ import annotations
import argparse
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional
_ACTION_RE = re.compile('<action>\\s*(.*?)\\s*</action>', re.DOTALL | re.IGNORECASE)
_MOVE_ACTIONS = {'LEFT', 'RIGHT', 'UP', 'DOWN'}

def _extract_action_tag(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub('(?<![</])action>', '<action>', text, flags=re.IGNORECASE)
    m = _ACTION_RE.search(cleaned)
    if not m:
        return None
    action = (m.group(1) or '').strip().replace('\n', ' ').strip()
    return action or None

def _iter_jsonl(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        open_fn = gzip.open if path.suffix == '.gz' else open
        with open_fn(path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                if row.get('prompt') is None or row.get('expert_action_label') is None:
                    continue
                if row.get('trajectory_id') is None or row.get('step_id') is None:
                    continue
                yield row

def _collect_trajectory_rows(paths: list[Path], *, trajectory_id: int | None, seed: int, max_scan_rows: int) -> tuple[int, list[dict[str, Any]]]:
    scan_rows = 0
    if trajectory_id is None:
        chosen_tid: int | None = None
        seen_tids: list[int] = []
        seen_set: set[int] = set()
        for row in _iter_jsonl(paths):
            scan_rows += 1
            if max_scan_rows > 0 and scan_rows > max_scan_rows:
                break
            tid = int(row['trajectory_id'])
            if tid not in seen_set:
                seen_set.add(tid)
                seen_tids.append(tid)
        if not seen_tids:
            raise SystemExit('No usable trajectory_id found while scanning dataset.')
        rng = random.Random(seed)
        chosen_tid = rng.choice(seen_tids)
    else:
        chosen_tid = int(trajectory_id)
    rows: list[dict[str, Any]] = []
    scan_rows = 0
    for row in _iter_jsonl(paths):
        scan_rows += 1
        if max_scan_rows > 0 and scan_rows > max_scan_rows:
            break
        if int(row['trajectory_id']) == chosen_tid:
            rows.append(row)
    if not rows:
        raise SystemExit(f'No rows found for trajectory_id={chosen_tid}.')
    rows.sort(key=lambda r: (int(r.get('step_id', 0)), int(r.get('agent_id', 0))))
    return (chosen_tid, rows)

def _generate_action(model: Any, tokenizer: Any, prompt: str, *, max_new_tokens: int, device: Any) -> tuple[Optional[str], str]:
    messages = [{'role': 'user', 'content': prompt}]
    chat_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt', add_generation_prompt=True)
    if isinstance(chat_ids, torch.Tensor):
        inputs = {'input_ids': chat_ids.to(device)}
    else:
        inputs = {k: v.to(device) for k, v in chat_ids.items()}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kw: dict[str, Any] = {**inputs, 'max_new_tokens': int(max_new_tokens), 'pad_token_id': pad_id, 'eos_token_id': tokenizer.eos_token_id, 'do_sample': False}
    import torch
    with torch.inference_mode():
        out = model.generate(**gen_kw)
    gen_ids = out[0][inputs['input_ids'].shape[-1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    end_tag = '</action>'
    pos = raw.find(end_tag)
    if pos != -1:
        raw = raw[:pos + len(end_tag)]
    return (_extract_action_tag(raw), raw)

def _pct(num: int, den: int) -> float:
    return float(num) / max(int(den), 1)

def main() -> None:
    p = argparse.ArgumentParser(description='Evaluate checkpoint on one training trajectory.')
    p.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    p.add_argument('--adapter_path', type=Path, default=None)
    p.add_argument('--data_dir', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt/out/expert_prompts_all_agents_gz'))
    p.add_argument('--glob', type=str, default='*.jsonl.gz')
    p.add_argument('--trajectory_id', type=int, default=None, help='Specific trajectory_id. If omitted, one random id is selected.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_scan_rows', type=int, default=0, help='0 = scan full dataset. Use >0 for quicker selection in very large datasets.')
    p.add_argument('--max_new_tokens', type=int, default=32)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--output_json', type=Path, default=None)
    p.add_argument('--trace_jsonl', type=Path, default=None, help='Optional per-row prediction trace.')
    args = p.parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise SystemExit('This script requires torch + transformers in the active environment.') from exc
    data_dir = args.data_dir.expanduser().resolve()
    paths = sorted((x for x in data_dir.glob(args.glob) if x.is_file() and x.stat().st_size > 0))
    if not paths:
        raise SystemExit(f'No files matched {args.glob!r} under {data_dir}')
    traj_id, rows = _collect_trajectory_rows(paths, trajectory_id=args.trajectory_id, seed=int(args.seed), max_scan_rows=int(args.max_scan_rows))
    adapter_path = args.adapter_path.expanduser().resolve() if args.adapter_path is not None else None
    tok_src = str(adapter_path) if adapter_path is not None and (adapter_path / 'tokenizer.json').is_file() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_cuda = torch.cuda.is_available() and (not bool(args.cpu))
    use_bf16 = bool(use_cuda and getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype, device_map='auto' if use_cuda else None)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
    if not use_cuda:
        model = model.to('cpu')
    model.eval()
    device = next(model.parameters()).device
    total = 0
    correct = 0
    parse_fail = 0
    move_target = 0
    move_pred = 0
    target_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    by_agent: dict[str, Counter[str]] = defaultdict(Counter)
    by_step: dict[int, Counter[str]] = defaultdict(Counter)
    wrong_examples: list[dict[str, Any]] = []
    pred_streak_max = 0
    pred_streak_cur = 0
    prev_pred = None
    trace_rows: list[dict[str, Any]] = []
    for row in rows:
        target = str(row['expert_action_label'])
        pred, raw = _generate_action(model, tokenizer, str(row['prompt']), max_new_tokens=int(args.max_new_tokens), device=device)
        pred_s = pred if pred is not None else '<PARSE_FAIL>'
        ok = pred == target
        step_id = int(row.get('step_id', -1))
        agent_id = str(row.get('agent_id', 'unknown'))
        role = str(row.get('agent_role', 'unknown'))
        total += 1
        correct += int(ok)
        parse_fail += int(pred is None)
        move_target += int(target in _MOVE_ACTIONS)
        move_pred += int(pred_s in _MOVE_ACTIONS)
        target_counts[target] += 1
        pred_counts[pred_s] += 1
        confusion[target, pred_s] += 1
        by_role[role]['total'] += 1
        by_role[role]['correct'] += int(ok)
        by_agent[agent_id]['total'] += 1
        by_agent[agent_id]['correct'] += int(ok)
        by_step[step_id]['total'] += 1
        by_step[step_id]['correct'] += int(ok)
        if pred_s == prev_pred:
            pred_streak_cur += 1
        else:
            pred_streak_cur = 1
            prev_pred = pred_s
        pred_streak_max = max(pred_streak_max, pred_streak_cur)
        if not ok and len(wrong_examples) < 30:
            wrong_examples.append({'step_id': step_id, 'agent_id': row.get('agent_id'), 'agent_role': row.get('agent_role'), 'task_idx': row.get('task_idx'), 'instruction_idx': row.get('instruction_idx'), 'target': target, 'pred': pred_s, 'raw_generation': raw})
        if args.trace_jsonl is not None:
            trace_rows.append({'trajectory_id': int(row['trajectory_id']), 'step_id': step_id, 'agent_id': row.get('agent_id'), 'agent_role': row.get('agent_role'), 'task_idx': row.get('task_idx'), 'instruction_idx': row.get('instruction_idx'), 'target_action': target, 'pred_action': pred_s, 'correct': int(ok), 'parse_fail': int(pred is None), 'raw_generation': raw})
    unique_steps = sorted(by_step.keys())
    result = {'trajectory_id': int(traj_id), 'rows_evaluated': int(total), 'unique_steps': int(len(unique_steps)), 'first_step_id': int(unique_steps[0]) if unique_steps else None, 'last_step_id': int(unique_steps[-1]) if unique_steps else None, 'exact_action_accuracy': _pct(correct, total), 'parse_failure_rate': _pct(parse_fail, total), 'move_action_rate_target': _pct(move_target, total), 'move_action_rate_pred': _pct(move_pred, total), 'max_same_prediction_streak': int(pred_streak_max), 'target_top': target_counts.most_common(25), 'prediction_top': pred_counts.most_common(25), 'confusion_top': [{'target': t, 'pred': pr, 'count': int(c)} for (t, pr), c in confusion.most_common(40)], 'by_role': {k: {'total': int(v['total']), 'correct': int(v['correct']), 'accuracy': _pct(int(v['correct']), int(v['total']))} for k, v in sorted(by_role.items())}, 'by_agent_id': {k: {'total': int(v['total']), 'correct': int(v['correct']), 'accuracy': _pct(int(v['correct']), int(v['total']))} for k, v in sorted(by_agent.items(), key=lambda kv: kv[0])}, 'step_accuracy_tail': [{'step_id': int(step), 'total': int(by_step[step]['total']), 'correct': int(by_step[step]['correct']), 'accuracy': _pct(int(by_step[step]['correct']), int(by_step[step]['total']))} for step in unique_steps[-30:]], 'wrong_examples': wrong_examples}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json is not None:
        out = args.output_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f'Saved metrics JSON: {out}')
    if args.trace_jsonl is not None:
        tpath = args.trace_jsonl.expanduser().resolve()
        tpath.parent.mkdir(parents=True, exist_ok=True)
        with tpath.open('w', encoding='utf-8') as f:
            for tr in trace_rows:
                f.write(json.dumps(tr, ensure_ascii=False) + '\n')
        print(f'Saved trace JSONL: {tpath}')
if __name__ == '__main__':
    main()
