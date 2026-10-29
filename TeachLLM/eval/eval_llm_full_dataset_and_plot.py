from __future__ import annotations
import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional
_ACTION_RE = re.compile('<action>\\s*(.*?)\\s*</action>', re.DOTALL | re.IGNORECASE)
COLOR_DATASET = '#F28E2B'
COLOR_AGENT = '#4E79A7'

def _extract_action_tag(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub('(?<![</])action>', '<action>', text, flags=re.IGNORECASE)
    m = _ACTION_RE.search(cleaned)
    if not m:
        return None
    action = (m.group(1) or '').strip().replace('\n', ' ').strip()
    return action or None

def _iter_jsonl(paths: list[Path], max_rows: int=0) -> Iterable[dict[str, Any]]:
    seen = 0
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
                seen += 1
                if max_rows > 0 and seen > max_rows:
                    return
                yield row

def _pct(num: int, den: int) -> float:
    return float(num) / max(int(den), 1)

def _build_overall_plot(target_counts: Counter[str], pred_counts: Counter[str], *, top_k: int, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [a for a, _ in target_counts.most_common(top_k)]
    for a, _ in pred_counts.most_common(top_k):
        if a not in labels:
            labels.append(a)
        if len(labels) >= top_k:
            break
    if not labels:
        labels = sorted(set(target_counts.keys()) | set(pred_counts.keys()))[:top_k]
    x = np.arange(len(labels))
    w = 0.42
    target_vals = np.array([target_counts.get(a, 0) for a in labels], dtype=np.float64)
    pred_vals = np.array([pred_counts.get(a, 0) for a in labels], dtype=np.float64)
    fig = plt.figure(figsize=(max(11, len(labels) * 0.55), 6))
    plt.bar(x - w / 2, target_vals, width=w, color=COLOR_DATASET, label='данные')
    plt.bar(x + w / 2, pred_vals, width=w, color=COLOR_AGENT, label='агент')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(axis='y', alpha=0.2)
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)

def _build_by_role_plot(target_by_role: dict[str, Counter[str]], pred_by_role: dict[str, Counter[str]], *, top_k: int, output_path: Path, title_prefix: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    roles = sorted(set(target_by_role.keys()) | set(pred_by_role.keys()))
    if not roles:
        return
    fig, axes = plt.subplots(len(roles), 1, figsize=(12, max(4, 4 * len(roles))), squeeze=False)
    for i, role in enumerate(roles):
        ax = axes[i, 0]
        tcnt = target_by_role.get(role, Counter())
        pcnt = pred_by_role.get(role, Counter())
        labels = [a for a, _ in tcnt.most_common(top_k)]
        for a, _ in pcnt.most_common(top_k):
            if a not in labels:
                labels.append(a)
            if len(labels) >= top_k:
                break
        if not labels:
            labels = sorted(set(tcnt.keys()) | set(pcnt.keys()))[:top_k]
        x = np.arange(len(labels))
        w = 0.42
        tvals = np.array([tcnt.get(a, 0) for a in labels], dtype=np.float64)
        pvals = np.array([pcnt.get(a, 0) for a in labels], dtype=np.float64)
        ax.bar(x - w / 2, tvals, width=w, color=COLOR_DATASET, label='данные')
        ax.bar(x + w / 2, pvals, width=w, color=COLOR_AGENT, label='агент')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title(f'{title_prefix}: role={role}')
        ax.grid(axis='y', alpha=0.2)
        ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)

def main() -> None:
    p = argparse.ArgumentParser(description='Run LLM over prompt dataset and plot label distributions.')
    p.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    p.add_argument('--adapter_path', type=Path, default=None)
    p.add_argument('--data_dir', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt/out/expert_prompts_all_agents_gz'))
    p.add_argument('--glob', type=str, default='*.jsonl.gz')
    p.add_argument('--max_rows', type=int, default=0, help='0 = whole dataset')
    p.add_argument('--max_new_tokens', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=8, help='LLM inference batch size.')
    p.add_argument('--dtype', type=str, default='auto', choices=['auto', 'fp32', 'fp16', 'bf16'], help='Model dtype. Use fp32 for maximum CUDA stability.')
    p.add_argument('--device_map', type=str, default='none', choices=['none', 'auto'], help="HuggingFace device_map. 'none' runs on a single device and is more stable.")
    p.add_argument('--top_k_plot', type=int, default=20)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--output_dir', type=Path, default=Path('/home/irazumovskii/project/runs/llm-vs-dataset'))
    p.add_argument('--predictions_gz', type=Path, default=None, help='Optional custom path for .jsonl.gz predictions. Defaults to output_dir.')
    args = p.parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise SystemExit('This script requires torch + transformers in active env.') from exc
    data_dir = args.data_dir.expanduser().resolve()
    paths = sorted((x for x in data_dir.glob(args.glob) if x.is_file() and x.stat().st_size > 0))
    if not paths:
        raise SystemExit(f'No files matched {args.glob!r} under {data_dir}')
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.predictions_gz.expanduser().resolve() if args.predictions_gz is not None else out_dir / 'llm_vs_dataset_predictions.jsonl.gz'
    adapter_path = args.adapter_path.expanduser().resolve() if args.adapter_path is not None else None
    tok_src = str(adapter_path) if adapter_path is not None and (adapter_path / 'tokenizer.json').is_file() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    use_cuda = torch.cuda.is_available() and (not bool(args.cpu))
    if args.dtype == 'fp32':
        dtype = torch.float32
    elif args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        if not use_cuda or not bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)()):
            raise SystemExit('Requested --dtype bf16, but this GPU/runtime does not support bf16.')
        dtype = torch.bfloat16
    else:
        use_bf16 = bool(use_cuda and getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
        dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_cuda else torch.float32
    device_map_opt = 'auto' if use_cuda and args.device_map == 'auto' else None
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype, device_map=device_map_opt)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
    if use_cuda and device_map_opt is None:
        model = model.to('cuda')
    elif not use_cuda:
        model = model.to('cpu')
    model.eval()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    total = 0
    correct = 0
    parse_fail = 0
    target_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    target_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    pred_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    by_role_correct: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: Counter[tuple[str, str]] = Counter()
    batch_size = max(int(args.batch_size), 1)
    pending_rows: list[dict[str, Any]] = []
    pending_prompts: list[str] = []

    def _flush_batch(out_f) -> None:
        nonlocal total, correct, parse_fail, pending_rows, pending_prompts
        if not pending_rows:
            return
        texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in pending_prompts]
        model_inputs = tokenizer(texts, return_tensors='pt', padding=True)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        gen_kw: dict[str, Any] = {**model_inputs, 'max_new_tokens': int(args.max_new_tokens), 'pad_token_id': pad_id, 'eos_token_id': tokenizer.eos_token_id, 'do_sample': False}
        with torch.inference_mode():
            out = model.generate(**gen_kw)
        attn = model_inputs.get('attention_mask')
        if attn is None:
            input_lens = [model_inputs['input_ids'].shape[1]] * len(pending_rows)
        else:
            input_lens = attn.sum(dim=1).tolist()
        for i, row in enumerate(pending_rows):
            target = str(row['expert_action_label'])
            role = str(row.get('agent_role', 'unknown'))
            prompt = pending_prompts[i]
            gen_ids = out[i][int(input_lens[i]):]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
            end_tag = '</action>'
            pos = raw.find(end_tag)
            if pos != -1:
                raw = raw[:pos + len(end_tag)]
            pred = _extract_action_tag(raw)
            pred_s = pred if pred is not None else '<PARSE_FAIL>'
            total += 1
            ok = pred == target
            correct += int(ok)
            parse_fail += int(pred is None)
            target_counts[target] += 1
            pred_counts[pred_s] += 1
            target_by_role[role][target] += 1
            pred_by_role[role][pred_s] += 1
            by_role_correct[role]['total'] += 1
            by_role_correct[role]['correct'] += int(ok)
            confusion[target, pred_s] += 1
            rec = {'prompt': prompt, 'llm_action': pred_s, 'dataset_action': target, 'agent_role': row.get('agent_role'), 'agent_id': row.get('agent_id'), 'trajectory_id': row.get('trajectory_id'), 'step_id': row.get('step_id'), 'task_idx': row.get('task_idx'), 'instruction_idx': row.get('instruction_idx'), 'correct': int(ok)}
            out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        pending_rows = []
        pending_prompts = []
    with gzip.open(pred_path, 'wt', encoding='utf-8') as out_f:
        for row in _iter_jsonl(paths, max_rows=int(args.max_rows)):
            pending_rows.append(row)
            pending_prompts.append(str(row['prompt']))
            if len(pending_rows) >= batch_size:
                _flush_batch(out_f)
        _flush_batch(out_f)
    overall_plot = out_dir / 'action_distribution_overall.png'
    by_role_plot = out_dir / 'action_distribution_by_role.png'
    _build_overall_plot(target_counts, pred_counts, top_k=max(1, int(args.top_k_plot)), output_path=overall_plot, title='Action distribution: данные (orange) vs агент (blue)')
    _build_by_role_plot(target_by_role, pred_by_role, top_k=max(1, int(args.top_k_plot)), output_path=by_role_plot, title_prefix='Action distribution')
    summary = {'rows_evaluated': int(total), 'exact_action_accuracy': _pct(correct, total), 'parse_failure_rate': _pct(parse_fail, total), 'prediction_file': str(pred_path), 'plots': {'overall': str(overall_plot), 'by_role': str(by_role_plot)}, 'target_top': target_counts.most_common(30), 'prediction_top': pred_counts.most_common(30), 'by_role': {role: {'total': int(cnt['total']), 'correct': int(cnt['correct']), 'accuracy': _pct(int(cnt['correct']), int(cnt['total']))} for role, cnt in sorted(by_role_correct.items())}, 'confusion_top': [{'target': t, 'pred': p, 'count': int(c)} for (t, p), c in confusion.most_common(40)]}
    summary_path = out_dir / 'summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Saved summary: {summary_path}')
if __name__ == '__main__':
    main()
