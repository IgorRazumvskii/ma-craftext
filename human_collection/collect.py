#!/usr/bin/env python3
"""Manual joint-action collection for easy_train. See README.md."""
import argparse
import hashlib
import json
import secrets
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MA = ROOT / 'MultyAgentCrafText/MA-Craftax'
sys.path[:0] = [str(MA), str(MA / 'craftax'), str(MA / 'baselines'), str(ROOT / 'CrafText')]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true', help='List easy_train instructions without launching the environment')
    p.add_argument('--instruction', type=int, default=0, help='Instruction index from --list')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, default=ROOT / 'human_data')
    return p.parse_args()


def main():
    args = arguments()
    # This parser uses the same scenario source as TeachLLM without importing JAX.
    sys.path.insert(0, str(ROOT / 'TeachLLM/data'))
    from trajectory_io import load_scenarios
    scenario = ROOT / 'CrafText/craftext/dataset/scenarious/jax_conditional_achievements/instructions.py'
    tasks, instructions, mapping = load_scenarios(scenario)
    if args.list:
        for i, instruction in enumerate(instructions):
            print(f'{i:4}  task={mapping[i]:3} {tasks[mapping[i]]}: {instruction}')
        return
    if not 0 <= args.instruction < len(instructions):
        raise SystemExit(f'Instruction must be between 0 and {len(instructions)-1}')

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pygame
    import pyarrow.parquet as pq
    from trajectory_parquet import arrays_to_table
    from craftax_together_talens.constants import Action, OBS_DIM, BLOCK_PIXEL_SIZE_AGENT
    from craftax_together_talens.envs.patterns_wrapper import CraftaxCoopSymbolicEnvWithMyWrapper
    from craftax_together_talens.envs.craftext_adapter import MultiAgentInstructionAdapter
    from craftax_together_talens.renderer.renderer_pixels import render_craftax_pixels
    from craftax_together_talens.constants import TEXTURES, load_player_specific_textures

    class NoEncoder:
        # The collector does not use embeddings: human reads the instruction.
        def __init__(self, form_to_use=None): pass
        def encode(self, items): return np.zeros((len(items), 1), dtype=np.float32)

    base = CraftaxCoopSymbolicEnvWithMyWrapper()
    env = MultiAgentInstructionAdapter(base, config_name='easy_train', encode_model_class=NoEncoder)
    actual = [str(x) for x in env.scenario_handler.scenario_data.instructions_list]
    actual_mapping = np.asarray(env.instruction_to_task_idx).reshape(-1).tolist()
    if actual != instructions or actual_mapping != mapping:
        raise RuntimeError('Runtime easy_train instructions differ from the SFT scenario mapping')
    textures = load_player_specific_textures(TEXTURES[BLOCK_PIXEL_SIZE_AGENT], base.static_env_params.player_count)
    pixel_fn = jax.jit(lambda state: render_craftax_pixels(state, BLOCK_PIXEL_SIZE_AGENT, base.static_env_params, textures))
    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, instruction_idx=args.instruction)
    agent_names = list(env.agents)
    action_labels = {int(a.value): a.name for a in Action}
    action_labels[52] = 'GIVE_TO_TEAMMATE_1'
    action_labels[53] = 'GIVE_TO_TEAMMATE_2'
    action_labels.update({54:'REQUEST_PICKAXE', 55:'REQUEST_SWORD', 56:'REQUEST_ARMOUR', 57:'REQUEST_BOW', 58:'REQUEST_ARROWS', 59:'REQUEST_TORCHES', 60:'REQUEST_SAPLING'})
    pending = [0] * len(agent_names)
    selected = 0
    cursor = [0] * len(agent_names)
    rows = []
    tid = secrets.randbelow(2**31 - 1) + 1
    pygame.init()
    screen = pygame.display.set_mode((1200, 830))
    pygame.display.set_caption('Craftax: manual trajectories')
    font = pygame.font.SysFont('monospace', 19)
    small = pygame.font.SysFont('monospace', 16)
    clock = pygame.time.Clock()
    message = 'Choose actions for each agent, then press Enter.'
    finished = False

    def write(text, x, y, color=(225, 225, 225), f=font):
        screen.blit(f.render(str(text), True, color), (x, y))

    def save():
        args.output.mkdir(parents=True, exist_ok=True)
        parts = args.output / 'parts'
        parts.mkdir(exist_ok=True)
        path = parts / f'human_{tid}.parquet'
        meta_path = args.output / f'human_{tid}.json'
        if path.exists() or meta_path.exists():
            raise FileExistsError('Trajectory ID collision; restart to generate a new ID')
        data = {k: np.asarray([row[k] for row in rows]) for k in rows[0]}
        data['trajectory_id'] = np.full(len(rows), tid, dtype=np.int32)
        data['step_id'] = np.arange(len(rows), dtype=np.int32)
        tmp = path.with_suffix('.tmp')
        pq.write_table(arrays_to_table(data), tmp, compression='zstd')
        tmp.rename(path)
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = None, None
        meta = {'trajectory_id': tid, 'source': 'human', 'seed': args.seed, 'instruction_idx': args.instruction,
                'instruction': instructions[args.instruction], 'task_idx': mapping[args.instruction],
                'task_name': tasks[mapping[args.instruction]], 'scenario': 'easy_train', 'success': True,
                'steps': len(rows), 'git_commit': revision, 'git_dirty': dirty,
                'scenario_sha256': hashlib.sha256(scenario.read_bytes()).hexdigest(),
                'created_utc': datetime.now(timezone.utc).isoformat(), 'observation': 'compact symbolic without instruction embedding',
                'parquet': str(path.relative_to(args.output))}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return path

    while True:
        masks = base.get_avail_actions(state.env_state)
        allowed = [[i for i, valid in enumerate(np.asarray(masks[a])) if valid] for a in agent_names]
        for i in range(len(agent_names)):
            cursor[i] = min(cursor[i], len(allowed[i]) - 1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type != pygame.KEYDOWN or finished:
                continue
            if event.key == pygame.K_TAB:
                selected = (selected + 1) % len(agent_names)
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                selected = event.key - pygame.K_1
            elif event.key in (pygame.K_UP, pygame.K_DOWN):
                cursor[selected] = (cursor[selected] + (-1 if event.key == pygame.K_UP else 1)) % len(allowed[selected])
            elif event.key == pygame.K_SPACE:
                pending[selected] = allowed[selected][cursor[selected]]
            elif event.key == pygame.K_RETURN:
                if any(pending[i] not in allowed[i] for i in range(len(agent_names))):
                    message = 'A queued action is no longer allowed; select another.'
                    continue
                actions = {a: jnp.int32(pending[i]) for i, a in enumerate(agent_names)}
                before = np.stack([np.asarray(obs[a][:-env.instruction_dim], dtype=np.float32) for a in agent_names])
                key, step_key = jax.random.split(key)
                obs, state, rewards, done, info = step_fn(step_key, state, actions)
                success = bool(np.asarray(info['user_info']['Instructions/instruction_done'])[0])
                terminal = bool(done['__all__'])
                rows.append({'obs': before, 'actions': np.asarray(pending, dtype=np.int32),
                             'rewards': np.asarray([rewards[a] for a in agent_names], dtype=np.float32),
                             'dones': np.asarray([done[a] for a in agent_names], dtype=bool),
                             'done_all': terminal, 'instruction_done': np.float32(success),
                             'instruction_idx': np.int32(args.instruction), 'task_idx': np.int32(mapping[args.instruction])})
                pending = [0] * len(agent_names)
                if terminal:
                    finished = True
                    message = f'SUCCESS: saved {save()}' if success else 'Episode failed or timed out; no trajectory saved.'
                else:
                    message = f'Step {len(rows)} complete. Actions reset to NOOP.'

        screen.fill((23, 25, 30))
        write(f'easy_train | task {mapping[args.instruction]}: {tasks[mapping[args.instruction]]}', 20, 15)
        for i, line in enumerate(textwrap.wrap(instructions[args.instruction], width=95)):
            write(line, 20, 45 + i * 22, f=small)
        # Only the selected agent is displayed. Mask tiles hidden by the symbolic light channel.
        if not finished:
            image = np.asarray(pixel_fn(state.env_state)[selected], dtype=np.float32).copy()
            sym = np.asarray(obs[agent_names[selected]][:-env.instruction_dim])
            cells = OBS_DIM[0] * OBS_DIM[1]
            channel_count = base.get_flat_map_obs_shape() // cells
            light = sym[:cells * channel_count].reshape(*OBS_DIM, channel_count)[..., -1] > 0.5
            tile_mask = np.repeat(np.repeat(light, BLOCK_PIXEL_SIZE_AGENT, 0), BLOCK_PIXEL_SIZE_AGENT, 1)
            image[:tile_mask.shape[0], :tile_mask.shape[1]] *= tile_mask[..., None]
            image = np.clip(image, 0, 255).astype(np.uint8)
            surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
            surface = pygame.transform.scale(surface, (440, int(440 * image.shape[0] / image.shape[1])))
            screen.blit(surface, (20, 120))
        write(f'View: agent {selected + 1}', 500, 125, (255, 220, 100))
        for i in range(len(agent_names)):
            write(f'{i+1}: {action_labels.get(pending[i], str(pending[i]))}', 500, 160 + i*26, (255, 220, 100) if i == selected else (210, 210, 210))
        write('Actions (up/down, space to select):', 500, 255, f=small)
        options = allowed[selected]
        start = max(0, min(cursor[selected] - 8, len(options) - 17))
        for line, idx in enumerate(range(start, min(len(options), start + 17))):
            aid = options[idx]
            write(('> ' if idx == cursor[selected] else '  ') + action_labels.get(aid, str(aid)), 500, 285 + line*22, (255, 220, 100) if idx == cursor[selected] else (200, 200, 200), small)
        write('1/2/3 or Tab: agent | Enter: step | Esc: quit', 20, 750, f=small)
        for i, line in enumerate(textwrap.wrap(message, width=115)):
            write(line, 20, 775+i*19, (140, 220, 160), small)
        pygame.display.flip()
        clock.tick(30)

if __name__ == '__main__':
    main()
