from __future__ import annotations
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import yaml
_ACTION_RE = re.compile('<action>\\s*(.*?)\\s*</action>', re.DOTALL | re.IGNORECASE)
_ROLE_COMMON_ACTION_IDS: set[int] = {0, 1, 2, 3, 4, 5, 6, 17, 19}
_ACTION_ID_GIVE = 52
_ACTION_ID_REQUEST_PICKAXE = 54

def _goal_from_instruction_idx(instruction_idx: int, instruction_text_by_idx: list[str]) -> str:
    if instruction_idx < 0:
        return 'Follow the current scenario instruction.'
    if instruction_idx >= len(instruction_text_by_idx):
        return f'Instruction index {instruction_idx} (out of range).'
    return instruction_text_by_idx[instruction_idx]

def _load_template_text(render_config: Path) -> str:
    cfg = render_config.resolve()
    with cfg.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f'Invalid craftext yaml: {cfg}')
    protocol_items = data.get('protocol')
    if not isinstance(protocol_items, list) or not protocol_items:
        raise ValueError(f"craftext yaml must contain non-empty 'protocol' list: {cfg}")
    protocol = protocol_items[0]
    if not isinstance(protocol, dict) or 'template' not in protocol:
        raise ValueError(f"Protocol entry must define 'template': {cfg}")
    template_name = protocol['template']
    path = Path(template_name)
    if not path.is_absolute():
        path = cfg.parent / path
    with path.open('r', encoding='utf-8') as f:
        return f.read()

def _prompt_fast(template: str, goal: str, obs_row: Any, action_block: str, dialog_block: str, agent_role: str, obs_include_legend: bool) -> str:
    from megaprompt.goal.just_a_goal import render as render_goal
    from megaprompt.obs.symbolic_coop_flat import render as render_obs
    obs_s = render_obs(obs_row, include_legend=bool(obs_include_legend))
    g = render_goal(goal)
    return template.replace('{{goal}}', g).replace('{{observation}}', obs_s).replace('{{dialog}}', dialog_block).replace('{{action}}', action_block).replace('{{agent_role}}', agent_role)

def _extract_action_tag(text: str, allowed_actions: Optional[list[str] | set[str] | tuple[str, ...]]=None) -> Optional[str]:
    if not text:
        return None
    cleaned = re.sub('(?<![</])action>', '<action>', text, flags=re.IGNORECASE)
    m = _ACTION_RE.search(cleaned)
    if m:
        action = (m.group(1) or '').strip().replace('\n', ' ').strip()
        return action or None
    if not allowed_actions:
        return None
    text_u = cleaned.upper()
    labels = sorted({str(x).upper() for x in allowed_actions if str(x).strip()}, key=len, reverse=True)
    for lab in labels:
        if re.search(f'(?<![A-Z0-9_]){re.escape(lab)}(?![A-Z0-9_])', text_u):
            return lab
    return None

def _setup_import_paths(*, megaprompt_root: Path, ma_craftax_root: Path, megaprompt_tools: Path) -> None:
    mp = str(megaprompt_root.resolve())
    mpt = str(megaprompt_tools.resolve())
    ma = str(ma_craftax_root.resolve())
    for p in (mp, mpt, ma):
        if p not in sys.path:
            sys.path.insert(0, p)

def _normalize_role(role: Any) -> str:
    s = str(role).strip().upper()
    if s == 'CRAFTER':
        return 'WARRIOR'
    return s

def _is_role_action_allowed(*, role: str, action_id: int) -> bool:
    give_allowed = _ACTION_ID_GIVE <= int(action_id) < _ACTION_ID_REQUEST_PICKAXE
    if int(action_id) in _ROLE_COMMON_ACTION_IDS or give_allowed:
        return True
    role_n = _normalize_role(role)
    if role_n == 'MINER':
        return int(action_id) == 54
    if role_n == 'WARRIOR':
        return int(action_id) == 55
    if role_n == 'BUILDER':
        return int(action_id) in {45, 46, 47, 48, 49, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
    return False

def _build_label_to_id(num_agents: int) -> dict[str, int]:
    from coop_symbolic_action_labels import coop_symbolic_action_dim, label_for_action_id
    d: dict[str, int] = {}
    for i in range(coop_symbolic_action_dim(num_agents=num_agents)):
        lab = label_for_action_id(i)
        d.setdefault(lab, i)
    return d

def _jax_scalar(x: Any) -> float:
    import jax
    return float(jax.device_get(x))

@dataclass
class CraftaxEvalConfig:
    ma_craftax_root: Path
    megaprompt_root: Path
    meta_json: Path
    render_config: Path
    env_name: str = ''
    craftext_config_name: str = ''
    max_steps_per_episode: int = 512
    max_new_tokens: int = 64
    eval_temperature: float = 0.0
    max_instructions: int = 0
    eval_per_task: bool = True
    eval_seed: int = 0
    task_names: tuple[str, ...] = ()
    task_names_include: tuple[str, ...] = ()
    obs_include_legend: bool = True

def run_craftax_instruction_suite(model: Any, tokenizer: Any, cfg: CraftaxEvalConfig, *, epoch: int=0) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    megaprompt_tools = cfg.megaprompt_root / 'tools'
    _setup_import_paths(megaprompt_root=cfg.megaprompt_root, ma_craftax_root=cfg.ma_craftax_root, megaprompt_tools=megaprompt_tools)
    from coop_symbolic_action_labels import agent_role_labels_from_meta, coop_symbolic_action_dim, prompt_allowed_action_strings
    from megaprompt.act.bullet_list import render as render_act
    from megaprompt.dialog.last_five import render as render_dialog
    from craftax.craftax_together_talens.envs.patterns_wrapper import make_craftax_env_with_my_wrapper
    from craftax_together_talens.game_logic.constants import get_role_action_mask
    with cfg.meta_json.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    texts = meta.get('instruction_text_by_idx')
    if not isinstance(texts, list) or not texts:
        raise ValueError('meta_json must contain instruction_text_by_idx list')
    num_agents = int(meta.get('num_agents', 3))
    env_name = cfg.env_name or str(meta.get('env_name', 'Craftax-Coop-Symbolic'))
    craftext_name = cfg.craftext_config_name or str(meta.get('craftext_config_name', 'conditional_achievements_easy_train'))
    os.environ['MEGAPROMPT_COOP_NUM_AGENTS'] = str(num_agents)
    role_labels = agent_role_labels_from_meta(meta, num_agents=num_agents)
    allowed_strings = prompt_allowed_action_strings(num_agents=num_agents)
    label_to_id = _build_label_to_id(num_agents)
    action_dim = coop_symbolic_action_dim(num_agents=num_agents)
    noop_id = label_to_id.get('NOOP', 0)
    role_allowed_strings: dict[str, list[str]] = {}
    for role in role_labels:
        rn = _normalize_role(role)
        if rn not in role_allowed_strings:
            role_allowed_strings[rn] = [a for a in allowed_strings if a in label_to_id and _is_role_action_allowed(role=rn, action_id=int(label_to_id[a]))]
    dialog_block = render_dialog([])
    template_fast = _load_template_text(cfg.render_config)
    env = make_craftax_env_with_my_wrapper(env_name, craftext_config_name=craftext_name)
    agents = list(env.agents)
    itt = np.asarray(jax.device_get(env.instruction_to_task_idx), dtype=np.int32).reshape(-1)
    include_names = {str(x).strip().upper() for x in cfg.task_names_include if str(x).strip()}
    n_instr = len(texts)
    if cfg.eval_per_task:
        task_to_indices: dict[int, list[int]] = {}
        for ii in range(n_instr):
            tid = int(itt[ii]) if ii < int(itt.size) else 0
            if include_names:
                tname = cfg.task_names[tid] if 0 <= tid < len(cfg.task_names) else ''
                if str(tname).strip().upper() not in include_names:
                    continue
            task_to_indices.setdefault(tid, []).append(ii)
        if not task_to_indices:
            raise ValueError(f'No instructions matched eval task filter: {sorted(include_names)}')
        unique_tasks = sorted(task_to_indices.keys())
        limit = len(unique_tasks) if cfg.max_instructions <= 0 else min(len(unique_tasks), cfg.max_instructions)
        pick_rng = np.random.default_rng(int(cfg.eval_seed) + int(epoch) * 100003)
        indices = [int(pick_rng.choice(task_to_indices[tid])) for tid in unique_tasks[:limit]]
    elif include_names:
        filtered = []
        for ii in range(n_instr):
            tid = int(itt[ii]) if ii < int(itt.size) else 0
            tname = cfg.task_names[tid] if 0 <= tid < len(cfg.task_names) else ''
            if str(tname).strip().upper() in include_names:
                filtered.append(ii)
        if not filtered:
            raise ValueError(f'No instructions matched eval task filter: {sorted(include_names)}')
        limit = len(filtered) if cfg.max_instructions <= 0 else min(len(filtered), cfg.max_instructions)
        indices = filtered[:limit]
    else:
        limit = n_instr if cfg.max_instructions <= 0 else min(n_instr, cfg.max_instructions)
        indices = list(range(limit))

    class _StopOnSubseq(StoppingCriteria):

        def __init__(self, stop_ids: list[int]):
            super().__init__()
            self.stop_ids = stop_ids

        def __call__(self, input_ids, scores, **kwargs):
            seq = input_ids[0].tolist()
            s = self.stop_ids
            return len(seq) >= len(s) and seq[-len(s):] == s
    successes = 0
    successes_done_and_instruction = 0
    generated_actions = 0
    parse_failures = 0
    role_invalid_actions = 0
    comet_rows: list[list[int]] = []
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    stop_tok = tokenizer.encode('</action>', add_special_tokens=False)
    stopping = StoppingCriteriaList([_StopOnSubseq(stop_tok)]) if stop_tok else None
    for ii in indices:
        rng = jax.random.PRNGKey(cfg.eval_seed + epoch * 100003 + ii * 1001)
        rng, k0 = jax.random.split(rng)
        obs, env_state = env.reset(k0, instruction_idx=int(ii))
        last_done = False
        goal = _goal_from_instruction_idx(ii, texts)
        for _t in range(cfg.max_steps_per_episode):
            action_dict: dict[str, Any] = {}
            base_state = getattr(env_state, 'env_state', env_state)
            specializations = np.asarray(jax.device_get(getattr(base_state, 'player_specialization'))).reshape(-1)
            role_masks = [np.asarray(jax.device_get(get_role_action_mask(jnp.int32(int(spec)), action_dim)), dtype=bool) for spec in specializations]
            chat_texts: list[str] = []
            for ag_idx, agent in enumerate(agents):
                obs_row = np.asarray(obs[agent], dtype=np.float32)
                role_n = _normalize_role(role_labels[ag_idx])
                role_actions = role_allowed_strings.get(role_n, allowed_strings)
                action_block = render_act(role_actions)
                prompt = _prompt_fast(template_fast, goal, obs_row, action_block, dialog_block, role_labels[ag_idx], cfg.obs_include_legend)
                text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
                chat_texts.append(str(text))
            tok = tokenizer(chat_texts, return_tensors='pt', padding=True)
            inputs = {k: v.to(device) for k, v in tok.items()}
            gen_kw: dict[str, Any] = {**inputs, 'max_new_tokens': int(cfg.max_new_tokens), 'pad_token_id': pad_id, 'eos_token_id': eos_id, 'do_sample': float(cfg.eval_temperature) > 0.0}
            if stopping is not None:
                gen_kw['stopping_criteria'] = stopping
            if float(cfg.eval_temperature) > 0.0:
                gen_kw['temperature'] = float(cfg.eval_temperature)
            with torch.inference_mode():
                out = model.generate(**gen_kw)
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask')
            if attention_mask is None:
                input_lens = [int(input_ids.shape[-1])] * int(input_ids.shape[0])
            else:
                input_lens = [int(x) for x in attention_mask.sum(dim=1).tolist()]
            for ag_idx, agent in enumerate(agents):
                row = min(ag_idx, int(out.shape[0]) - 1)
                start = min(max(input_lens[row], 0), int(out.shape[-1]))
                gen_ids = out[row][start:]
                raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
                end_tag = '</action>'
                p = raw.find(end_tag)
                if p != -1:
                    raw = raw[:p + len(end_tag)]
                role_n = _normalize_role(role_labels[ag_idx])
                role_actions = role_allowed_strings.get(role_n, allowed_strings)
                label = _extract_action_tag(raw, allowed_actions=role_actions)
                generated_actions += 1
                parsed_ok = label is not None and label in label_to_id
                if not parsed_ok:
                    parse_failures += 1
                    aid = noop_id
                else:
                    aid = label_to_id[label]
                    if ag_idx < len(role_masks):
                        mask = role_masks[ag_idx]
                        if int(aid) >= len(mask) or not bool(mask[int(aid)]):
                            role_invalid_actions += 1
                            aid = noop_id
                action_dict[agent] = jnp.int32(int(aid))
            rng, k1 = jax.random.split(rng)
            obs, env_state, _rew, done, _info = env.step(k1, env_state, action_dict)
            if bool(_jax_scalar(done['__all__'])):
                last_done = True
                break
        instruction_done = _jax_scalar(getattr(env_state, 'instruction_done', 0))
        ok = instruction_done > 0.5
        ok_done_and_instruction = bool(last_done and ok)
        if ok:
            successes += 1
        if ok_done_and_instruction:
            successes_done_and_instruction += 1
        tid = int(itt[ii]) if ii < int(itt.size) else 0
        comet_rows.append([int(ii), tid, 1 if ok else 0])
    sr = successes / max(1, len(indices))
    return {'env/success_rate': float(sr), 'env/successes': float(successes), 'env/num_instructions': float(len(indices)), 'env/eval_per_task': float(1.0 if cfg.eval_per_task else 0.0), 'env/action_dim': float(action_dim), 'env/generated_actions': float(generated_actions), 'env/parse_failure_rate': float(parse_failures / max(generated_actions, 1)), 'env/role_invalid_action_rate': float(role_invalid_actions / max(generated_actions, 1)), 'env/success_rate_done_and_instruction': float(successes_done_and_instruction / max(1, len(indices))), '_comet_rows': comet_rows}
