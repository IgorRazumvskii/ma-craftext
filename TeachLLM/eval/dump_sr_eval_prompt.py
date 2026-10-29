from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from craftax_llm_eval import _goal_from_instruction_idx, _load_template_text, _prompt_fast, _setup_import_paths

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Dump exact SR-eval prompts for each agent.')
    p.add_argument('--ma_craftax_root', type=Path, default=Path('/home/irazumovskii/project/MultyAgentCrafText/MA-Craftax'))
    p.add_argument('--megaprompt_root', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt'))
    p.add_argument('--meta_json', type=Path, default=Path('/home/irazumovskii/project/MultyAgentCrafText/MA-Craftax/datasets/expert_mappo.meta.json'))
    p.add_argument('--render_config', type=Path, default=Path('/home/irazumovskii/project/MegaPrompt/craftext_prompt/templates/expert_npz/craftext.yaml'))
    p.add_argument('--instruction_idx', type=int, default=0)
    p.add_argument('--task_idx', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output_json', type=Path, required=True)
    p.add_argument('--eval_env_name', type=str, default='')
    p.add_argument('--eval_craftext_config', type=str, default='')
    p.add_argument('--jax_gpu', action='store_true', help='Run JAX on CUDA. Default keeps JAX on CPU, like visualize_qwen_env_rollout.py.')
    p.add_argument('--obs_include_legend', action=argparse.BooleanOptionalAction, default=True, help='Whether to include map legend text in rendered observations.')
    return p

def main() -> None:
    args = _parser().parse_args()
    if not args.jax_gpu and (not os.environ.get('JAX_PLATFORMS')):
        os.environ['JAX_PLATFORMS'] = 'cpu'
    megaprompt_root = args.megaprompt_root.expanduser().resolve()
    ma_craftax_root = args.ma_craftax_root.expanduser().resolve()
    _setup_import_paths(megaprompt_root=megaprompt_root, ma_craftax_root=ma_craftax_root, megaprompt_tools=megaprompt_root / 'tools')
    import jax
    from coop_symbolic_action_labels import agent_role_labels_from_meta, coop_symbolic_action_dim, prompt_allowed_action_strings
    from craftax.craftax_together_talens.envs.patterns_wrapper import make_craftax_env_with_my_wrapper
    from megaprompt.act.bullet_list import render as render_act
    from megaprompt.dialog.last_five import render as render_dialog
    with args.meta_json.expanduser().resolve().open('r', encoding='utf-8') as f:
        meta = json.load(f)
    texts = meta['instruction_text_by_idx']
    num_agents = int(meta.get('num_agents', 3))
    os.environ['MEGAPROMPT_COOP_NUM_AGENTS'] = str(num_agents)
    env_name = args.eval_env_name.strip() or str(meta.get('env_name', 'Craftax-Coop-Symbolic'))
    craftext_name = args.eval_craftext_config.strip() or str(meta.get('craftext_config_name', 'conditional_achievements_easy_train'))
    env = make_craftax_env_with_my_wrapper(env_name, craftext_config_name=craftext_name)
    instruction_idx = int(args.instruction_idx)
    if args.task_idx is not None:
        itt = np.asarray(jax.device_get(env.instruction_to_task_idx), dtype=np.int32).reshape(-1)
        matches = np.where(itt == int(args.task_idx))[0]
        if matches.size == 0:
            raise SystemExit(f'--task_idx={args.task_idx} not found in env instruction map.')
        instruction_idx = int(matches[0])
    if instruction_idx < 0 or instruction_idx >= len(texts):
        raise SystemExit(f'instruction_idx out of range [0, {len(texts) - 1}]')
    role_labels = agent_role_labels_from_meta(meta, num_agents=num_agents)
    action_dim = coop_symbolic_action_dim(num_agents=num_agents)
    action_block = render_act(prompt_allowed_action_strings(num_agents=num_agents))
    dialog_block = render_dialog([])
    template = _load_template_text(args.render_config.expanduser().resolve())
    rng = jax.random.PRNGKey(int(args.seed))
    rng, k0 = jax.random.split(rng)
    obs, _env_state = env.reset(k0, instruction_idx=instruction_idx)
    goal = _goal_from_instruction_idx(instruction_idx, texts)
    prompts: list[dict[str, Any]] = []
    for ag_idx, agent in enumerate(env.agents):
        obs_row = np.asarray(obs[agent], dtype=np.float32)
        prompt = _prompt_fast(template, goal, obs_row, action_block, dialog_block, role_labels[ag_idx], bool(args.obs_include_legend))
        prompts.append({'agent': agent, 'agent_idx': int(ag_idx), 'role': role_labels[ag_idx], 'obs_shape': list(obs_row.shape), 'prompt': prompt})
    task_idx = int(np.asarray(jax.device_get(env.instruction_to_task_idx))[instruction_idx])
    task_names = meta.get('task_names', [])
    task_name = task_names[task_idx] if isinstance(task_names, list) and 0 <= task_idx < len(task_names) else ''
    out = {'env_name': env_name, 'craftext_config_name': craftext_name, 'num_agents': num_agents, 'action_dim': int(action_dim), 'instruction_idx': int(instruction_idx), 'task_idx': int(task_idx), 'task_name': task_name, 'goal': goal, 'seed': int(args.seed), 'prompts': prompts}
    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'Saved SR prompts: {output_json}')
if __name__ == '__main__':
    main()
