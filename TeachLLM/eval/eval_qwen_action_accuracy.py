from __future__ import annotations
import argparse
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
_ACTION_RE = re.compile('<action>\\s*(.*?)\\s*</action>', re.DOTALL | re.IGNORECASE)

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
                if row.get('prompt') is None or row.get('expert_action_label') is None:
                    continue
                yield row

def _reservoir_sample(paths: list[Path], *, sample_size: int, seed: int, max_rows: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    seen = 0
    for row in _iter_jsonl(paths):
        seen += 1
        if max_rows > 0 and seen > max_rows:
            break
        if len(sample) < sample_size:
            sample.append(row)
            continue
        j = rng.randrange(seen)
        if j < sample_size:
            sample[j] = row
    return sample

class _StopOnSubseq(StoppingCriteria):

    def __init__(self, stop_ids: list[int]):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist()
        s = self.stop_ids
        return len(seq) >= len(s) and seq[-len(s):] == s

def _generate_action(model: Any, tokenizer: Any, prompt: str, *, max_new_tokens: int, device: torch.device) -> tuple[Optional[str], str]:
    messages = [{'role': 'user', 'content': prompt}]
    chat_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt', add_generation_prompt=True)
    if isinstance(chat_ids, torch.Tensor):
        inputs = {'input_ids': chat_ids.to(device)}
    else:
        inputs = {k: v.to(device) for k, v in chat_ids.items()}
    stop_tok = tokenizer.encode('</action>', add_special_tokens=False)
    stopping = StoppingCriteriaList([_StopOnSubseq(stop_tok)]) if stop_tok else None
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kw: dict[str, Any] = {**inputs, 'max_new_tokens': int(max_new_tokens), 'pad_token_id': pad_id, 'eos_token_id': tokenizer.eos_token_id, 'do_sample': False}
    if stopping is not None:
        gen_kw['stopping_criteria'] = stopping
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
    p = argparse.ArgumentParser(description='Exact action accuracy for Qwen SFT adapters.')
    p.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    p.add_argument('--adapter_path', type=Path, default=None)
    p.add_argument('--data_dir', type=Path, required=True)
    p.add_argument('--glob', type=str, default='*.jsonl.gz')
    p.add_argument('--sample_rows', type=int, default=512)
    p.add_argument('--max_scan_rows', type=int, default=0, help='0 = scan all rows for reservoir sampling.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_new_tokens', type=int, default=32)
    p.add_argument('--output_json', type=Path, default=None)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    paths = sorted((x for x in data_dir.glob(args.glob) if x.is_file() and x.stat().st_size > 0))
    if not paths:
        raise SystemExit(f'No files matched {args.glob!r} under {data_dir}')
    rows = _reservoir_sample(paths, sample_size=int(args.sample_rows), seed=int(args.seed), max_rows=int(args.max_scan_rows))
    if not rows:
        raise SystemExit('No usable rows sampled.')
    adapter_path = args.adapter_path.expanduser().resolve() if args.adapter_path is not None else None
    tok_src = str(adapter_path) if adapter_path is not None and (adapter_path / 'tokenizer.json').is_file() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_cuda = torch.cuda.is_available() and (not bool(args.cpu))
    dtype = torch.bfloat16 if use_cuda else torch.float32
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
    target_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: Counter[tuple[str, str]] = Counter()
    examples: list[dict[str, Any]] = []
    for row in rows:
        target = str(row['expert_action_label'])
        pred, raw = _generate_action(model, tokenizer, str(row['prompt']), max_new_tokens=int(args.max_new_tokens), device=device)
        pred_s = pred if pred is not None else '<PARSE_FAIL>'
        ok = pred == target
        total += 1
        correct += int(ok)
        parse_fail += int(pred is None)
        target_counts[target] += 1
        pred_counts[pred_s] += 1
        confusion[target, pred_s] += 1
        role = str(row.get('agent_role', 'unknown'))
        task = str(row.get('task_idx', 'unknown'))
        by_role[role]['total'] += 1
        by_role[role]['correct'] += int(ok)
        by_task[task]['total'] += 1
        by_task[task]['correct'] += int(ok)
        if not ok and len(examples) < 20:
            examples.append({'target': target, 'pred': pred_s, 'agent_role': row.get('agent_role'), 'task_idx': row.get('task_idx'), 'instruction_idx': row.get('instruction_idx'), 'raw_generation': raw})
    result = {'sample_rows': int(total), 'exact_action_accuracy': _pct(correct, total), 'parse_failure_rate': _pct(parse_fail, total), 'target_top': target_counts.most_common(20), 'prediction_top': pred_counts.most_common(20), 'by_role': {k: {'total': int(v['total']), 'correct': int(v['correct']), 'accuracy': _pct(int(v['correct']), int(v['total']))} for k, v in sorted(by_role.items())}, 'by_task_top': [{'task_idx': k, 'total': int(v['total']), 'correct': int(v['correct']), 'accuracy': _pct(int(v['correct']), int(v['total']))} for k, v in sorted(by_task.items(), key=lambda kv: int(kv[1]['total']), reverse=True)[:30]], 'confusion_top': [{'target': t, 'pred': pr, 'count': int(c)} for (t, pr), c in confusion.most_common(30)], 'wrong_examples': examples}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json is not None:
        out = args.output_json.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
if __name__ == '__main__':
    main()
