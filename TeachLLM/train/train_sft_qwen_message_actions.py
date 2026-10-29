from __future__ import annotations
import argparse
import gzip
import inspect
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List
import numpy as np
from torch.utils.data import IterableDataset
from transformers.trainer_callback import TrainerCallback
_TEACHLLM_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _TEACHLLM_ROOT.parent
if (_PROJECT_ROOT / 'MegaPrompt').exists() is False and (_PROJECT_ROOT / 'MultyAgentCrafText').exists() is False:
    _PROJECT_ROOT = _PROJECT_ROOT.parent
logger = logging.getLogger(__name__)

def _list_data_files(data_dir: Path, pattern: str, exclude_prefix: str) -> List[Path]:
    files = sorted(data_dir.glob(pattern))
    out: List[Path] = []
    for p in files:
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

def _load_dataset_files(file_paths: List[Path], data_format: str='jsonl', cache_dir: str | None=None, keep_in_memory: bool=False):
    from datasets import Dataset, concatenate_datasets
    if not file_paths:
        raise SystemExit('No JSONL files matched. Check --data_dir and --glob.')
    load_kw: dict = {}
    if cache_dir:
        load_kw['cache_dir'] = cache_dir
    if keep_in_memory:
        load_kw['keep_in_memory'] = True
    parts = []
    for fp in file_paths:
        if data_format == 'parquet':
            ds = Dataset.from_parquet(str(fp), **load_kw)
        else:
            ds = Dataset.from_json(str(fp), **load_kw)
        parts.append(ds)
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)

def _load_paraphrase_pools(path: Path) -> tuple[Dict[str, List[str]], List[str]]:
    request_pool: Dict[str, List[str]] = {}
    global_replies: List[str] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = str(row.get('expert_action_label', '')).strip()
            req = str(row.get('request', '')).strip()
            rep = str(row.get('reply', '')).strip()
            if label and req:
                request_pool.setdefault(label, []).append(req)
            if rep:
                global_replies.append(rep)
    if not request_pool:
        raise SystemExit(f'No request paraphrases found in: {path}')
    if not global_replies:
        raise SystemExit(f'No reply texts found in: {path}')
    return (request_pool, global_replies)

def _count_jsonl_records(path: Path) -> int:
    open_fn = gzip.open if path.suffix == '.gz' else open
    cnt = 0
    with open_fn(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            cnt += chunk.count(b'\n')
    return cnt

def _assistant_for_label(label: str, request_pool: Dict[str, List[str]], global_replies: List[str], transfer_labels: set[str], rng: random.Random) -> str:
    if label in request_pool:
        msg = rng.choice(request_pool[label])
        return f'<message>{msg}</message><action>{label}</action>'
    if label in transfer_labels:
        msg = rng.choice(global_replies)
        return f'<message>{msg}</message><action>{label}</action>'
    return f'<action>{label}</action>'

class JsonlMessagesIterableDataset(IterableDataset):

    def __init__(self, file_paths: List[Path], *, request_pool: Dict[str, List[str]], global_replies: List[str], transfer_labels: set[str], seed: int, world_size: int=1, rank: int=0):
        self.file_paths = list(file_paths)
        self.request_pool = request_pool
        self.global_replies = global_replies
        self.transfer_labels = transfer_labels
        self.seed = int(seed)
        self.world_size = max(int(world_size), 1)
        self.rank = max(int(rank), 0)
        self.column_names = None
        self._total_rows = sum((_count_jsonl_records(p) for p in self.file_paths))
        self._local_rows = int(math.ceil(self._total_rows / self.world_size))
        self._transforms: list[Any] = []
        self._selected_columns: set[str] | None = None

    def map(self, function: Any, fn_kwargs: dict[str, Any] | None=None, batched: bool=False, remove_columns: str | list[str] | None=None, **kwargs: Any) -> 'JsonlMessagesIterableDataset':
        if batched:
            raise SystemExit('streaming_jsonl: batched map is not supported in lazy iterable mode.')
        fn_kwargs = fn_kwargs or {}
        rm_cols = [remove_columns] if isinstance(remove_columns, str) else list(remove_columns or [])

        def _apply(example: dict[str, Any]) -> dict[str, Any]:
            out = function(dict(example), **fn_kwargs)
            merged = dict(example)
            if isinstance(out, dict):
                merged.update(out)
            for c in rm_cols:
                merged.pop(c, None)
            return merged
        self._transforms.append(_apply)
        return self

    def shuffle(self, seed: int | None=None, **kwargs: Any) -> 'JsonlMessagesIterableDataset':
        if seed is None:
            return self
        rng = np.random.default_rng(int(seed))
        rng.shuffle(self.file_paths)
        return self

    def select_columns(self, columns: list[str] | set[str]) -> 'JsonlMessagesIterableDataset':
        self._selected_columns = set(columns)
        return self

    def __len__(self) -> int:
        return self._local_rows

    def __iter__(self) -> Iterator[dict[str, Any]]:
        from torch.utils.data import get_worker_info
        import torch.distributed as dist
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        rank = self.rank
        world_size = self.world_size
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        global_workers = max(world_size * num_workers, 1)
        global_worker_id = rank * num_workers + worker_id
        rng = random.Random(self.seed + 997 * global_worker_id)
        for file_idx, path in enumerate(self.file_paths):
            if file_idx % global_workers != global_worker_id:
                continue
            open_fn = gzip.open if path.suffix == '.gz' else open
            with open_fn(path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    prompt = row.get('prompt')
                    label = row.get('expert_action_label')
                    if prompt is None or label is None:
                        continue
                    lab_s = str(label)
                    assistant = _assistant_for_label(lab_s, self.request_pool, self.global_replies, self.transfer_labels, rng)
                    item: dict[str, Any] = {'messages': [{'role': 'user', 'content': str(prompt)}, {'role': 'assistant', 'content': assistant}]}
                    for tr in self._transforms:
                        item = tr(item)
                    if self._selected_columns is not None:
                        item = {k: v for k, v in item.items() if k in self._selected_columns}
                    yield item

def _build_mapper(request_pool: Dict[str, List[str]], global_replies: List[str], transfer_labels: set[str], seed: int):
    rng = random.Random(seed)

    def _to_messages(batch: dict) -> dict:
        prompts = batch['prompt']
        labels = batch['expert_action_label']
        messages = []
        for p, lab in zip(prompts, labels):
            lab_s = str(lab)
            assistant = _assistant_for_label(lab_s, request_pool, global_replies, transfer_labels, rng)
            messages.append([{'role': 'user', 'content': p}, {'role': 'assistant', 'content': assistant}])
        return {'messages': messages}
    return _to_messages

def _build_sft_trainer(SFTTrainer, sig, **common):
    kwargs = {k: v for k, v in common.items() if k in sig.parameters}
    return SFTTrainer(**kwargs)

def _make_sft_config(SFTConfig, cfg_kwargs: dict, eos_token: str):
    allowed = set(inspect.signature(SFTConfig.__init__).parameters) - {'self'}
    merged = {'eos_token': eos_token, 'assistant_only_loss': True, **cfg_kwargs}
    filtered = {k: v for k, v in merged.items() if k in allowed}
    return SFTConfig(**filtered)

class CraftaxStepEvalCallback(TrainerCallback):

    def __init__(self, eval_cfg: Any, every_steps: int, start_step: int=0, run_on_epoch_end: bool=False):
        self.eval_cfg = eval_cfg
        self.every_steps = max(int(every_steps), 0)
        self.start_step = max(int(start_step), 0)
        self.run_on_epoch_end = run_on_epoch_end
        self.trainer: Any = None
        self._sr_cumulative: dict[str, Any] = {}
        self._last_eval_step: int = -1

    def _should_skip_rank(self) -> bool:
        import torch.distributed as dist
        return dist.is_initialized() and dist.get_rank() != 0

    def _run_eval(self, state: Any, model=None, processing_class=None) -> None:
        if self.trainer is None or self._should_skip_rank():
            return
        from comet_sr_utils import get_hf_comet_experiment, log_mappo_style_sr_to_comet
        from craftax_llm_eval import run_craftax_instruction_suite
        step = int(getattr(state, 'global_step', 0) or 0)
        if step == self._last_eval_step:
            return
        self._last_eval_step = step
        model = model if model is not None else self.trainer.model
        tokenizer = processing_class if processing_class is not None else self.trainer.processing_class
        was_training = model.training
        model.eval()
        try:
            metrics = run_craftax_instruction_suite(model, tokenizer, self.eval_cfg, epoch=int(getattr(state, 'epoch', 0) or 0))
            rows = metrics.pop('_comet_rows', [])
            metrics['sr_eval/step'] = float(step)
            metrics['sr_eval/epoch'] = float(getattr(state, 'epoch', 0) or 0)
            experiment = get_hf_comet_experiment(self.trainer)
            if experiment is not None and rows:
                log_mappo_style_sr_to_comet(experiment, rows, self.eval_cfg.task_names, self._sr_cumulative, step=step, epoch=float(getattr(state, 'epoch', 0) or 0))
            for k in list(metrics.keys()):
                if str(k).startswith('_'):
                    metrics.pop(k, None)
            self.trainer.log(metrics)
            logger.info('Step SR eval @step=%d: SR=%.4f (%d/%d instructions)', step, metrics.get('env/success_rate', 0.0), int(metrics.get('env/successes', 0)), int(metrics.get('env/num_instructions', 0)))
        except Exception as exc:
            logger.exception('Step SR eval failed at step %s: %s', step, exc)
        finally:
            if was_training:
                model.train()

    def on_step_end(self, args: Any, state: Any, control: Any, model=None, processing_class=None, **kwargs: Any):
        step = int(getattr(state, 'global_step', 0) or 0)
        if self.every_steps <= 0:
            return control
        if step < self.start_step:
            return control
        if step % self.every_steps != 0:
            return control
        self._run_eval(state, model=model, processing_class=processing_class)
        return control

    def on_epoch_end(self, args: Any, state: Any, control: Any, model=None, processing_class=None, **kwargs: Any):
        if self.run_on_epoch_end:
            self._run_eval(state, model=model, processing_class=processing_class)
        return control

def main() -> None:
    _extra_paths = (_PROJECT_ROOT / 'TeachLLM', _PROJECT_ROOT / 'TeachLLM' / 'train', _PROJECT_ROOT / 'TeachLLM' / 'eval', _PROJECT_ROOT / 'TeachLLM' / 'data')
    for _p in _extra_paths:
        _p_s = str(_p)
        if _p_s not in sys.path:
            sys.path.insert(0, _p_s)
    parser = argparse.ArgumentParser(description='SFT with <message> targets for REQUEST/GIVE actions.')
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--glob', type=str, default='*.jsonl')
    parser.add_argument('--data_format', type=str, default='auto', choices=['auto', 'jsonl', 'parquet'], help='Input file format. auto infers from --glob suffix.')
    parser.add_argument('--keep_columns', type=str, default='prompt,expert_action_label', help='Comma-separated columns to keep right after load (reduces RAM/disk pressure).')
    parser.add_argument('--exclude_prefix', type=str, default='.')
    parser.add_argument('--max_files', type=int, default=0)
    parser.add_argument('--paraphrase_jsonl', type=Path, required=True)
    parser.add_argument('--transfer_labels', type=str, default='GIVE,GIVE_TO_PLAYER_1')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--output_dir', type=Path, default=Path('./runs/sft-qwen-message-actions'))
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=4, help='Per-device batch; effective batch = batch_size × grad_accum (default 4×4=16, use 8 for 32).')
    parser.add_argument('--grad_accum', type=int, default=4, help='Gradient accumulation steps (default 4; with batch_size=4 → effective batch 16).')
    parser.add_argument('--max_seq_len', type=int, default=4096)
    parser.add_argument('--val_ratio', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--report_to', type=str, default='none')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--comet_project', type=str, default='teach-llm')
    parser.add_argument('--comet_workspace', type=str, default='igorrazumvskii', help='Comet workspace (sets COMET_WORKSPACE if unset). Empty string = Comet default workspace.')
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--datasets_cache_dir', type=Path, default=None)
    parser.add_argument('--keep_jsonl_in_memory', action='store_true')
    parser.add_argument('--streaming_jsonl', action='store_true', help='Stream JSONL/JSONL.GZ directly without HF Arrow cache (disables val split).')
    parser.add_argument('--sr_eval_steps', type=int, default=0, help='Run Craftax SR eval every N train steps; 0 disables.')
    parser.add_argument('--sr_eval_start_step', type=int, default=0, help='Start periodic SR eval from this global step.')
    parser.add_argument('--sr_eval_on_epoch_end', action='store_true', help='Also run SR eval on epoch end.')
    parser.add_argument('--craftax_root', type=Path, default=_PROJECT_ROOT / 'MultyAgentCrafText' / 'MA-Craftax', help='MA-Craftax repo root (directory containing craftax/).')
    parser.add_argument('--megaprompt_root', type=Path, default=_PROJECT_ROOT / 'MegaPrompt', help='MegaPrompt repo root.')
    parser.add_argument('--eval_meta_json', type=Path, default=_PROJECT_ROOT / 'MultyAgentCrafText' / 'MA-Craftax' / 'datasets' / 'expert_mappo.meta.json')
    parser.add_argument('--eval_render_config', type=Path, default=_PROJECT_ROOT / 'MegaPrompt' / 'craftext_prompt' / 'templates' / 'expert_npz' / 'craftext.yaml')
    parser.add_argument('--eval_env_name', type=str, default='', help='Override env_name from meta (optional).')
    parser.add_argument('--eval_craftext_config', type=str, default='', help='Override craftext config name from meta.')
    parser.add_argument('--eval_max_steps', type=int, default=512, help='Max env steps per instruction episode.')
    parser.add_argument('--eval_max_instructions', type=int, default=0, help='Max items per SR eval (0 = all). Default mode: one random paraphrase per task (57 tasks). With --eval_all_variants: max instruction indices (0 = all 333 variants).')
    parser.add_argument('--eval_all_variants', action='store_true', help='Evaluate every instruction paraphrase variant instead of one random variant per task.')
    parser.add_argument('--eval_max_new_tokens', type=int, default=64)
    parser.add_argument('--eval_temperature', type=float, default=0.0)
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_obs_include_legend', action=argparse.BooleanOptionalAction, default=True, help='Whether to include map legend text in rendered observations during env eval.')
    args = parser.parse_args()
    if args.datasets_cache_dir is not None:
        dc = args.datasets_cache_dir.expanduser().resolve()
        dc.mkdir(parents=True, exist_ok=True)
        os.environ['HF_DATASETS_CACHE'] = str(dc)
    if args.report_to in ('comet_ml', 'all'):
        if 'COMET_PROJECT_NAME' not in os.environ and args.comet_project.strip():
            os.environ['COMET_PROJECT_NAME'] = args.comet_project.strip()
        if args.comet_workspace.strip() and 'COMET_WORKSPACE' not in os.environ:
            os.environ['COMET_WORKSPACE'] = args.comet_workspace.strip()
        comet_cfg = Path.home() / '.comet.config'
        if not os.environ.get('COMET_API_KEY', '').strip() and (not comet_cfg.is_file()):
            raise SystemExit('Comet: set COMET_API_KEY in the environment (export COMET_API_KEY=...) or run `comet login`.\nDo not put the key in the repo or on the command line. To train without Comet: --report_to none')
    if args.sr_eval_steps > 0:
        cr = args.craftax_root.expanduser().resolve()
        mp = args.megaprompt_root.expanduser().resolve()
        if not (cr / 'craftax').is_dir():
            raise SystemExit(f'--sr_eval_steps enabled: craftax_root must contain craftax/: {cr}')
        if not (mp / 'megaprompt').is_dir():
            raise SystemExit(f'--sr_eval_steps enabled: megaprompt_root must contain megaprompt/: {mp}')
        if not args.eval_meta_json.is_file():
            raise SystemExit(f'--eval_meta_json not found: {args.eval_meta_json}')
        if not args.eval_render_config.is_file():
            raise SystemExit(f'--eval_render_config not found: {args.eval_render_config}')
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f'data_dir is not a directory: {data_dir}')
    para_path = args.paraphrase_jsonl.expanduser().resolve()
    if not para_path.is_file():
        raise SystemExit(f'paraphrase_jsonl not found: {para_path}')
    transfer_labels = {x.strip() for x in args.transfer_labels.split(',') if x.strip()}
    request_pool, global_replies = _load_paraphrase_pools(para_path)
    file_paths = _list_data_files(data_dir, args.glob, args.exclude_prefix)
    if args.max_files and args.max_files > 0:
        file_paths = file_paths[:args.max_files]
    if not file_paths:
        raise SystemExit('No input JSONL files found.')
    ds_cache = os.environ.get('HF_DATASETS_CACHE')
    data_format = args.data_format
    if data_format == 'auto':
        data_format = 'parquet' if '.parquet' in args.glob else 'jsonl'
    eval_ds = None
    if args.streaming_jsonl:
        if data_format != 'jsonl':
            raise SystemExit('--streaming_jsonl supports only JSONL/JSONL.GZ input.')
        if args.val_ratio > 0:
            logger.warning('--streaming_jsonl: validation split is disabled (setting val_ratio=0).')
            args.val_ratio = 0.0
        train_ds = JsonlMessagesIterableDataset(file_paths, request_pool=request_pool, global_replies=global_replies, transfer_labels=transfer_labels, seed=args.seed)
        print(f'Streaming mode: {len(file_paths)} file(s), ~{len(train_ds)} row(s) per rank, no HF Arrow cache writes.')
    else:
        raw = _load_dataset_files(file_paths, data_format=data_format, cache_dir=ds_cache, keep_in_memory=args.keep_jsonl_in_memory)
        keep_columns = [c.strip() for c in args.keep_columns.split(',') if c.strip()]
        if keep_columns:
            present = [c for c in keep_columns if c in raw.column_names]
            if present:
                raw = raw.select_columns(present)
        required = {'prompt', 'expert_action_label'}
        missing = required - set(raw.column_names)
        if missing:
            raise SystemExit(f'Dataset missing columns {missing}; have {raw.column_names}')
        mapper = _build_mapper(request_pool, global_replies, transfer_labels, seed=args.seed)
        _cols = list(raw.column_names)
        raw = raw.map(mapper, batched=True, remove_columns=_cols, load_from_cache_file=not args.keep_jsonl_in_memory, keep_in_memory=args.keep_jsonl_in_memory)
        raw = raw.shuffle(seed=args.seed)
        if args.val_ratio > 0:
            split = raw.train_test_split(test_size=args.val_ratio, seed=args.seed)
            train_ds, eval_ds = (split['train'], split['test'])
        else:
            train_ds = raw
    try:
        import torch
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise SystemExit('Install required training deps first (transformers/trl/peft/torch).') from e
    if args.report_to in ('comet_ml', 'all'):
        try:
            import comet_ml
        except ImportError as e:
            raise SystemExit(f'Comet: pip install comet_ml; set COMET_API_KEY.\n{e}') from e
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch_dtype, device_map='auto')
    peft_config = None
    effective_batch = args.batch_size * args.grad_accum
    print(f'Effective batch size: {effective_batch} (batch_size={args.batch_size} × grad_accum={args.grad_accum})')
    if not args.no_lora:
        peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias='none', task_type=TaskType.CAUSAL_LM, inference_mode=False, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    use_bf16 = bool(torch.cuda.is_available())
    save_total = None if args.save_total_limit <= 0 else args.save_total_limit
    run_name = args.run_name or args.output_dir.name
    cfg_kwargs = dict(output_dir=str(args.output_dir), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, logging_steps=args.logging_steps, save_strategy='steps', save_steps=args.save_steps, save_total_limit=save_total, save_safetensors=True, bf16=use_bf16, fp16=not use_bf16, max_length=args.max_seq_len, report_to=args.report_to, run_name=run_name, seed=args.seed, gradient_checkpointing=True, eval_strategy='steps' if eval_ds is not None else 'no')
    if eval_ds is not None:
        cfg_kwargs['eval_steps'] = min(args.save_steps, 200)
    sft_args = _make_sft_config(SFTConfig, cfg_kwargs, tokenizer.eos_token or '</s>')
    sig = inspect.signature(SFTTrainer.__init__)
    callbacks_list: list[Any] = []
    if args.sr_eval_steps > 0:
        from craftax_llm_eval import CraftaxEvalConfig
        with args.eval_meta_json.open('r', encoding='utf-8') as f:
            meta_eval = json.load(f)
        tn = meta_eval.get('task_names')
        task_names_tuple = tuple((str(x) for x in tn)) if isinstance(tn, list) else ()
        eval_cfg = CraftaxEvalConfig(ma_craftax_root=args.craftax_root.expanduser().resolve(), megaprompt_root=args.megaprompt_root.expanduser().resolve(), meta_json=args.eval_meta_json.expanduser().resolve(), render_config=args.eval_render_config.expanduser().resolve(), env_name=(args.eval_env_name or '').strip(), craftext_config_name=(args.eval_craftext_config or '').strip(), max_steps_per_episode=int(args.eval_max_steps), max_new_tokens=int(args.eval_max_new_tokens), eval_temperature=float(args.eval_temperature), max_instructions=int(args.eval_max_instructions), eval_per_task=not args.eval_all_variants, eval_seed=int(args.eval_seed), task_names=task_names_tuple, obs_include_legend=bool(args.eval_obs_include_legend))
        callbacks_list.append(CraftaxStepEvalCallback(eval_cfg=eval_cfg, every_steps=args.sr_eval_steps, start_step=args.sr_eval_start_step, run_on_epoch_end=args.sr_eval_on_epoch_end))
    trainer = _build_sft_trainer(SFTTrainer, sig, model=model, args=sft_args, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer, tokenizer=tokenizer, peft_config=peft_config, callbacks=callbacks_list if callbacks_list else None)
    for cb in callbacks_list:
        if isinstance(cb, CraftaxStepEvalCallback):
            cb.trainer = trainer
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
if __name__ == '__main__':
    main()
