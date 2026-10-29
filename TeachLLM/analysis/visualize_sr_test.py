from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from craftax_llm_eval import CraftaxEvalConfig, run_craftax_instruction_suite

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run SR test and save visualizations.')
    p.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    p.add_argument('--adapter_path', type=Path, default=None, help='Path to PEFT LoRA checkpoint dir, e.g. runs/.../checkpoint-8900.')
    p.add_argument('--ma_craftax_root', type=Path, default=Path('/home/irazumovskii/project/MultyAgentCrafText/MA-Craftax'))
    p.add_argument('--megaprompt_root', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt'))
    p.add_argument('--meta_json', type=Path, default=Path('/home/irazumovskii/project/MultyAgentCrafText/MA-Craftax/datasets/expert_mappo.meta.json'))
    p.add_argument('--render_config', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt/craftext_prompt/templates/expert_npz/craftext.yaml'))
    p.add_argument('--env_name', type=str, default='')
    p.add_argument('--craftext_config_name', type=str, default='')
    p.add_argument('--max_steps', type=int, default=64)
    p.add_argument('--max_instructions', type=int, default=16)
    p.add_argument('--max_new_tokens', type=int, default=64)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--obs_include_legend', action=argparse.BooleanOptionalAction, default=True, help='Whether to include map legend text in rendered observations.')
    p.add_argument('--epoch', type=int, default=0)
    p.add_argument('--output_dir', type=Path, default=Path('./runs/sr-viz'))
    p.add_argument('--run_name', type=str, default='')
    p.add_argument('--trust_remote_code', action='store_true', default=True)
    p.add_argument('--cpu', action='store_true', help='Force CPU instead of CUDA.')
    return p

def _save_outputs(output_dir: Path, run_name: str, metrics: dict[str, Any], rows: list[list[int]], task_names: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'{run_name}_rows.csv'
    summary_path = output_dir / f'{run_name}_summary.json'
    cumulative_plot = output_dir / f'{run_name}_cumulative_sr.png'
    per_task_plot = output_dir / f'{run_name}_per_task_sr.png'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['order_idx', 'instruction_idx', 'task_idx', 'task_name', 'success', 'cumulative_successes', 'cumulative_sr'])
        w.writeheader()
        solved = 0
        for i, row in enumerate(rows):
            instr_idx, task_idx, success = (int(row[0]), int(row[1]), int(row[2]))
            solved += success
            task_name = task_names[task_idx] if 0 <= task_idx < len(task_names) else f'task_{task_idx}'
            w.writerow({'order_idx': i, 'instruction_idx': instr_idx, 'task_idx': task_idx, 'task_name': task_name, 'success': success, 'cumulative_successes': solved, 'cumulative_sr': solved / (i + 1)})
    summary = {'metrics': metrics, 'num_rows': len(rows), 'rows_csv': str(csv_path)}
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if rows:
        y = np.cumsum(np.array([int(r[2]) for r in rows], dtype=np.float32)) / np.arange(1, len(rows) + 1)
        x = np.arange(1, len(rows) + 1, dtype=np.int32)
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, linewidth=2)
        plt.xlabel('Instruction eval order')
        plt.ylabel('Cumulative SR')
        plt.title('Cumulative Success Rate')
        plt.grid(True, alpha=0.3)
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(cumulative_plot, dpi=160)
        plt.close()
    if rows:
        task_total: dict[int, int] = {}
        task_solved: dict[int, int] = {}
        for _instr_idx, task_idx, success in rows:
            t = int(task_idx)
            task_total[t] = task_total.get(t, 0) + 1
            task_solved[t] = task_solved.get(t, 0) + int(success)
        items: list[tuple[str, float]] = []
        for t in sorted(task_total.keys()):
            name = task_names[t] if 0 <= t < len(task_names) else f'task_{t}'
            sr = task_solved[t] / max(task_total[t], 1)
            items.append((f'{t}:{name[:28]}', float(sr)))
        items.sort(key=lambda kv: kv[1])
        labels = [x[0] for x in items]
        values = [x[1] for x in items]
        plt.figure(figsize=(12, max(4, len(values) * 0.28)))
        y_pos = np.arange(len(values))
        plt.barh(y_pos, values)
        plt.yticks(y_pos, labels)
        plt.xlabel('SR')
        plt.title('Per-task Success Rate')
        plt.xlim(0.0, 1.0)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(per_task_plot, dpi=160)
        plt.close()
    print(f'Saved summary: {summary_path}')
    print(f'Saved rows CSV: {csv_path}')
    if rows:
        print(f'Saved cumulative plot: {cumulative_plot}')
        print(f'Saved per-task plot: {per_task_plot}')

def main() -> None:
    args = _build_parser().parse_args()
    with args.meta_json.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    task_names = [str(x) for x in meta.get('task_names', [])] if isinstance(meta.get('task_names'), list) else []
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    dtype = torch.float32 if device == 'cpu' else torch.bfloat16
    adapter_path = args.adapter_path.expanduser().resolve() if args.adapter_path is not None else None
    tok_src = str(adapter_path) if adapter_path is not None and (adapter_path / 'tokenizer.json').is_file() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, torch_dtype=dtype, device_map='auto' if device == 'cuda' else None)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
    if device == 'cpu':
        model = model.to('cpu')
    cfg = CraftaxEvalConfig(ma_craftax_root=args.ma_craftax_root.expanduser().resolve(), megaprompt_root=args.megaprompt_root.expanduser().resolve(), meta_json=args.meta_json.expanduser().resolve(), render_config=args.render_config.expanduser().resolve(), env_name=args.env_name.strip(), craftext_config_name=args.craftext_config_name.strip(), max_steps_per_episode=int(args.max_steps), max_new_tokens=int(args.max_new_tokens), eval_temperature=float(args.temperature), max_instructions=int(args.max_instructions), eval_seed=int(args.seed), task_names=tuple(task_names), obs_include_legend=bool(args.obs_include_legend))
    metrics = run_craftax_instruction_suite(model, tokenizer, cfg, epoch=int(args.epoch))
    rows = metrics.pop('_comet_rows', [])
    clean_metrics = {k: v for k, v in metrics.items() if not str(k).startswith('_')}
    print(clean_metrics)
    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name.strip() or f'sr_test_{ts}'
    _save_outputs(args.output_dir.expanduser().resolve(), run_name, clean_metrics, rows, task_names)
if __name__ == '__main__':
    main()
