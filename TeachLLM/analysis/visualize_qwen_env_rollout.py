from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from craftax_llm_eval import _extract_action_tag, _goal_from_instruction_idx, _is_role_action_allowed, _load_template_text, _normalize_role, _prompt_fast, _setup_import_paths

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Visualize Qwen actions in Craftax env.')
    p.add_argument('--model', type=str, default='Qwen/Qwen2.5-3B-Instruct')
    p.add_argument('--adapter_path', type=Path, default=None, help='Path to PEFT LoRA checkpoint dir (e.g. runs/.../checkpoint-900).')
    p.add_argument('--task_idx', type=int, default=None, help='Unique task index from meta task_names; uses first matching instruction variant.')
    p.add_argument('--ma_craftax_root', type=Path, default=Path('/home/irazumovskii/project/MultyAgentCrafText/MA-Craftax'))
    p.add_argument('--megaprompt_root', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt'))
    p.add_argument('--meta_json', type=Path, default=Path('/home/irazumovskii/project/MultyAgentCrafText/MA-Craftax/datasets/expert_mappo.meta.json'))
    p.add_argument('--render_config', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt/craftext_prompt/templates/expert_npz/craftext.yaml'))
    p.add_argument('--instruction_idx', type=int, default=0)
    p.add_argument('--max_steps', type=int, default=30)
    p.add_argument('--max_new_tokens', type=int, default=64)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--render_agent_idx', type=int, default=0)
    p.add_argument('--fps', type=int, default=6)
    p.add_argument('--output_dir', type=Path, default=Path('./runs/qwen-env-viz'))
    p.add_argument('--run_name', type=str, default='')
    p.add_argument('--disable_overlay', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--jax_gpu', action='store_true', help='Run Craftax/JAX on CUDA (default: JAX on CPU to avoid GPU conflicts with PyTorch).')
    p.add_argument('--obs_include_legend', action=argparse.BooleanOptionalAction, default=True, help='Whether to include map legend text in rendered observations.')
    return p

def _safe_name(s: str) -> str:
    return ''.join((ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in s)).strip('_') or 'run'

def _make_renderer(num_agents: int, block_pixel_size: int=64):
    from craftax.craftax_together_talens.constants import TEXTURES, load_player_specific_textures
    from craftax.craftax_together_talens.craftax_state import StaticEnvParams
    from craftax.craftax_together_talens.renderer.renderer_pixels import render_craftax_pixels
    static_params = StaticEnvParams(player_count=num_agents)
    textures = TEXTURES[block_pixel_size]
    player_specific_textures = load_player_specific_textures(textures, static_params.player_count)

    def _render(base_state: Any, agent_idx: int) -> np.ndarray:
        try:
            from jax.experimental import disable_jit
        except Exception:
            disable_jit = None
        if disable_jit is not None:
            with disable_jit():
                pixels = render_craftax_pixels(base_state, block_pixel_size, static_params, player_specific_textures)[agent_idx]
        else:
            pixels = render_craftax_pixels(base_state, block_pixel_size, static_params, player_specific_textures)[agent_idx]
        frame = np.asarray(pixels)
        return np.clip(frame, 0, 255).astype(np.uint8)
    return _render

def _overlay_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, mode='RGBA')
    line_h = 18
    box_h = 12 + line_h * len(lines)
    draw.rectangle([(0, 0), (img.width, box_h)], fill=(0, 0, 0, 160))
    y = 6
    for line in lines:
        draw.text((8, y), line[:180], fill=(255, 255, 255, 255))
        y += line_h
    return np.asarray(img)

def main() -> None:
    args = _parser().parse_args()
    if not args.jax_gpu and (not os.environ.get('JAX_PLATFORMS')):
        os.environ['JAX_PLATFORMS'] = 'cpu'
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with args.meta_json.expanduser().resolve().open('r', encoding='utf-8') as f:
        meta = json.load(f)
    texts = meta['instruction_text_by_idx']
    num_agents = int(meta.get('num_agents', 3))
    env_name = str(meta.get('env_name', 'Craftax-Coop-Symbolic'))
    craftext_name = str(meta.get('craftext_config_name', 'conditional_achievements_easy_train'))
    instruction_idx = int(args.instruction_idx)
    if args.task_idx is not None and args.instruction_idx != 0:
        raise SystemExit('Use only one of --instruction_idx or --task_idx.')
    megaprompt_tools = args.megaprompt_root.expanduser().resolve() / 'tools'
    _setup_import_paths(megaprompt_root=args.megaprompt_root.expanduser().resolve(), ma_craftax_root=args.ma_craftax_root.expanduser().resolve(), megaprompt_tools=megaprompt_tools)
    os.environ['MEGAPROMPT_COOP_NUM_AGENTS'] = str(num_agents)
    from coop_symbolic_action_labels import agent_role_labels_from_meta, coop_symbolic_action_dim, label_for_action_id, prompt_allowed_action_strings
    from megaprompt.act.bullet_list import render as render_act
    from megaprompt.dialog.last_five import render as render_dialog
    from craftax.craftax_together_talens.envs.patterns_wrapper import make_craftax_env_with_my_wrapper
    from craftax_together_talens.game_logic.constants import get_role_action_mask
    import jax
    import jax.numpy as jnp
    import imageio.v2 as imageio
    role_labels = agent_role_labels_from_meta(meta, num_agents=num_agents)
    dialog_block = render_dialog([])
    template = _load_template_text(args.render_config.expanduser().resolve())
    label_to_id = {label_for_action_id(i): i for i in range(coop_symbolic_action_dim(num_agents=num_agents))}
    all_allowed_strings = prompt_allowed_action_strings(num_agents=num_agents)
    role_allowed_strings: dict[str, list[str]] = {}
    for role in role_labels:
        rn = _normalize_role(role)
        if rn not in role_allowed_strings:
            role_allowed_strings[rn] = [a for a in all_allowed_strings if a in label_to_id and _is_role_action_allowed(role=rn, action_id=int(label_to_id[a]))]
    noop_id = label_to_id.get('NOOP', 0)
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    dtype = torch.float32 if device == 'cpu' else torch.float16
    adapter_path = args.adapter_path.expanduser().resolve() if args.adapter_path is not None else None
    tok_src = str(adapter_path) if adapter_path is not None and (adapter_path / 'tokenizer.json').is_file() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype, device_map='auto' if device == 'cuda' else None, attn_implementation='eager')
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
    if device == 'cpu':
        model = model.to('cpu')
    model.eval()

    class _StopOnSubseq(StoppingCriteria):

        def __init__(self, stop_ids: list[int]):
            super().__init__()
            self.stop_ids = stop_ids

        def __call__(self, input_ids, scores, **kwargs):
            seq = input_ids[0].tolist()
            s = self.stop_ids
            return len(seq) >= len(s) and seq[-len(s):] == s
    stop_tok = tokenizer.encode('</action>', add_special_tokens=False)
    stopping = StoppingCriteriaList([_StopOnSubseq(stop_tok)]) if stop_tok else None
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    env = make_craftax_env_with_my_wrapper(env_name, craftext_config_name=craftext_name)
    if args.task_idx is not None:
        itt = np.asarray(jax.device_get(env.instruction_to_task_idx), dtype=np.int32).reshape(-1)
        matches = np.where(itt == int(args.task_idx))[0]
        if matches.size == 0:
            raise SystemExit(f'--task_idx={args.task_idx} not found in env instruction map.')
        instruction_idx = int(matches[0])
    if instruction_idx < 0 or instruction_idx >= len(texts):
        raise SystemExit(f'instruction_idx out of range [0, {len(texts) - 1}]')
    if args.render_agent_idx < 0 or args.render_agent_idx >= len(env.agents):
        raise SystemExit(f'--render_agent_idx must be in [0, {len(env.agents) - 1}]')
    render_frame = _make_renderer(num_agents=num_agents)
    rng = jax.random.PRNGKey(int(args.seed))
    rng, k0 = jax.random.split(rng)
    obs, env_state = env.reset(k0, instruction_idx=instruction_idx)
    goal = _goal_from_instruction_idx(instruction_idx, texts)
    frames: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    done_all = False
    final_instruction_done = 0.0
    for step in range(int(args.max_steps)):
        action_dict: dict[str, Any] = {}
        step_agents: list[dict[str, Any]] = []
        action_labels_for_overlay: list[str] = []
        base_state_before_step = getattr(env_state, 'env_state', env_state)
        pos_before_arr = None
        if hasattr(base_state_before_step, 'player_position'):
            try:
                pos_before_arr = np.asarray(jax.device_get(getattr(base_state_before_step, 'player_position')))
            except Exception:
                pos_before_arr = None
        specializations = np.asarray(jax.device_get(getattr(base_state_before_step, 'player_specialization'))).reshape(-1)
        role_masks = [np.asarray(jax.device_get(get_role_action_mask(jnp.int32(int(spec)), int(coop_symbolic_action_dim(num_agents=num_agents)))), dtype=bool) for spec in specializations]
        for ag_idx, agent in enumerate(env.agents):
            obs_row = np.asarray(obs[agent], dtype=np.float32)
            prompt = _prompt_fast(template, goal, obs_row, render_act(role_allowed_strings.get(_normalize_role(role_labels[ag_idx]), all_allowed_strings)), dialog_block, role_labels[ag_idx], bool(args.obs_include_legend))
            messages = [{'role': 'user', 'content': prompt}]
            chat_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt', add_generation_prompt=True)
            if isinstance(chat_ids, torch.Tensor):
                inputs = {'input_ids': chat_ids.to(next(model.parameters()).device)}
            else:
                inputs = {k: v.to(next(model.parameters()).device) for k, v in chat_ids.items()}
            gen_kw: dict[str, Any] = {**inputs, 'max_new_tokens': int(args.max_new_tokens), 'pad_token_id': pad_id, 'eos_token_id': eos_id, 'do_sample': float(args.temperature) > 0.0}
            if stopping is not None:
                gen_kw['stopping_criteria'] = stopping
            if float(args.temperature) > 0.0:
                gen_kw['temperature'] = float(args.temperature)
            with torch.inference_mode():
                out = model.generate(**gen_kw)
            gen_ids = out[0][inputs['input_ids'].shape[-1]:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
            end_tag = '</action>'
            cut = raw.find(end_tag)
            if cut != -1:
                raw = raw[:cut + len(end_tag)]
            parsed = _extract_action_tag(raw, allowed_actions=role_allowed_strings.get(_normalize_role(role_labels[ag_idx]), all_allowed_strings))
            if parsed is None or parsed not in label_to_id:
                aid = noop_id
                final_label = 'NOOP'
                parse_failed = True
            else:
                aid = int(label_to_id[parsed])
                final_label = str(parsed)
                parse_failed = False
            role_allowed = True
            if ag_idx < len(role_masks):
                mask = role_masks[ag_idx]
                role_allowed = int(aid) < len(mask) and bool(mask[int(aid)])
            effective_aid = int(aid) if role_allowed else int(noop_id)
            effective_label = final_label if role_allowed else 'NOOP'
            action_dict[agent] = jnp.int32(effective_aid)
            action_labels_for_overlay.append(f'{agent}:{effective_label}')
            step_agents.append({'agent': agent, 'role': role_labels[ag_idx], 'parsed_action': parsed, 'generated_action': final_label, 'generated_action_id': int(aid), 'parse_failed': parse_failed, 'role_allowed': bool(role_allowed), 'applied_action': effective_label, 'applied_action_id': int(effective_aid), 'raw_generation': raw})
        rng, k1 = jax.random.split(rng)
        obs, env_state, reward, done, info = env.step(k1, env_state, action_dict)
        done_all = bool(jax.device_get(done['__all__']))
        final_instruction_done = float(jax.device_get(getattr(env_state, 'instruction_done', 0.0)))
        user_info = info.get('user_info', {})
        task_idx = int(np.asarray(jax.device_get(user_info.get('Instructions/task_idx', np.array([0]))))[0])
        base_state_after_step = getattr(env_state, 'env_state', env_state)
        pos_after_arr = None
        if hasattr(base_state_after_step, 'player_position'):
            try:
                pos_after_arr = np.asarray(jax.device_get(getattr(base_state_after_step, 'player_position')))
            except Exception:
                pos_after_arr = None
        stats = {'health': np.asarray(jax.device_get(getattr(base_state_after_step, 'player_health'))).tolist(), 'food': np.asarray(jax.device_get(getattr(base_state_after_step, 'player_food'))).tolist(), 'drink': np.asarray(jax.device_get(getattr(base_state_after_step, 'player_drink'))).tolist(), 'energy': np.asarray(jax.device_get(getattr(base_state_after_step, 'player_energy'))).tolist(), 'hunger': np.asarray(jax.device_get(getattr(base_state_after_step, 'player_hunger'))).tolist(), 'thirst': np.asarray(jax.device_get(getattr(base_state_after_step, 'player_thirst'))).tolist(), 'fatigue': np.asarray(jax.device_get(getattr(base_state_after_step, 'player_fatigue'))).tolist(), 'alive': np.asarray(jax.device_get(getattr(base_state_after_step, 'player_alive'))).tolist()}
        if pos_after_arr is not None:
            stats['position'] = pos_after_arr.tolist()
        for ag_idx, info_row in enumerate(step_agents):
            pb = None
            pa = None
            if pos_before_arr is not None and ag_idx < len(pos_before_arr):
                pb = np.asarray(pos_before_arr[ag_idx]).tolist()
            if pos_after_arr is not None and ag_idx < len(pos_after_arr):
                pa = np.asarray(pos_after_arr[ag_idx]).tolist()
            moved = bool(pb is not None and pa is not None and (pb != pa))
            info_row['pos_before'] = pb
            info_row['pos_after'] = pa
            info_row['moved'] = moved
        base_state = env_state.env_state
        frame = render_frame(base_state, agent_idx=int(args.render_agent_idx))
        if not args.disable_overlay:
            r = int(args.render_agent_idx)
            hp = float(stats['health'][r]) if r < len(stats['health']) else -1.0
            fd = int(stats['food'][r]) if r < len(stats['food']) else -1
            dr = int(stats['drink'][r]) if r < len(stats['drink']) else -1
            en = int(stats['energy'][r]) if r < len(stats['energy']) else -1
            pos = stats.get('position', [])
            pos_s = str(pos[r]) if isinstance(pos, list) and r < len(pos) else 'n/a'
            overlay = [f'step={step} done={int(done_all)} instr_done={final_instruction_done:.0f} task_idx={task_idx}', f'instruction_idx={instruction_idx}', f'agent{r}: hp={hp:.1f} food={fd} drink={dr} energy={en} pos={pos_s}', *action_labels_for_overlay[:4]]
            frame = _overlay_frame(frame, overlay)
        frames.append(frame)
        trace.append({'step': step, 'done_all': int(done_all), 'instruction_done': final_instruction_done, 'task_idx': task_idx, 'agents': step_agents, 'player_stats': stats, 'reward': {k: float(jax.device_get(v)) for k, v in reward.items()}})
        if done_all:
            break
    success = bool(final_instruction_done > 0.5)
    success_done_and_instruction = bool(done_all and final_instruction_done > 0.5)
    run_name = args.run_name.strip() or _safe_name(f'qwen_instr{instruction_idx}_seed{args.seed}')
    gif_path = out_dir / f'{run_name}.gif'
    trace_path = out_dir / f'{run_name}_trace.jsonl'
    summary_path = out_dir / f'{run_name}_summary.json'
    if not frames:
        raise SystemExit('No frames generated.')
    imageio.mimsave(str(gif_path), frames, fps=max(int(args.fps), 1))
    with trace_path.open('w', encoding='utf-8') as f:
        for row in trace:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    task_names = meta.get('task_names', [])
    task_name = str(task_names[task_idx]) if trace and 0 <= int(trace[-1].get('task_idx', -1)) < len(task_names) else ''
    summary = {'model': args.model, 'adapter_path': str(adapter_path) if adapter_path is not None else '', 'instruction_idx': instruction_idx, 'task_idx': int(trace[-1]['task_idx']) if trace else -1, 'task_name': task_name, 'instruction_text': texts[instruction_idx], 'steps_recorded': len(trace), 'done_all': bool(done_all), 'instruction_done': float(final_instruction_done), 'success': int(success), 'success_done_and_instruction': int(success_done_and_instruction), 'render_agent_idx': int(args.render_agent_idx), 'gif_path': str(gif_path), 'trace_path': str(trace_path)}
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'Saved GIF: {gif_path}')
    print(f'Saved trace: {trace_path}')
    print(f'Saved summary: {summary_path}')
    print(json.dumps(summary, ensure_ascii=False))
if __name__ == '__main__':
    main()
