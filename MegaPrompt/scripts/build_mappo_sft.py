#!/usr/bin/env python3
"""Convert successful cooperative MAPPO trajectories into per-agent SFT JSONL."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1]
CRAFTAX = WORK / 'ma-craftext/MultyAgentCrafText/MA-Craftax'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CRAFTAX / 'baselines'))
from megaprompt.renders import Renderer
from trajectory_parquet import table_to_arrays
from visualize_each_task import load_scenarios
from visualize_mappo_trajectory import read_layout

DEFAULT_PARTS = WORK / 'HF/datasets/mappo_100k_20261101_210728_parts'
DEFAULT_SCENARIOS = WORK / 'ma-craftext/CrafText/craftext/dataset/scenarious/jax_conditional_achievements/instructions.py'
REQUESTS = ['FOOD', 'DRINK', 'WOOD', 'STONE', 'IRON', 'COAL', 'DIAMOND', 'RUBY', 'SAPPHIRE', 'PICKAXE', 'SWORD', 'ARMOUR', 'BOW', 'ARROWS', 'TORCHES']
ROLES = ['MINER', 'BUILDER', 'CRAFTER']
INVENTORY = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapphire', 'ruby', 'sapling', 'torches', 'arrows', 'books', 'pickaxe_tier', 'sword_tier', 'sword_enchantment', 'bow_enchantment', 'bow']
STATS = ['food', 'drink', 'energy', 'mana', 'xp', 'dexterity', 'strength', 'intelligence']
COMMON = [0, 1, 2, 3, 4, 5, 6, 17, 19, 52, 53]
BUILDER = [45, 46, 47, 48, 49, 50, 51, 7, 8, 9, 10, 28, 11, 12, 13, 20, 14, 15, 16, 21, 22, 23, 25, 38]

# Stable, unique map symbols; the full legend is included in each observation.
BLOCK_SYMBOLS = {
    'INVALID': 'IV', 'OUT_OF_BOUNDS': 'OB', 'GRASS': 'GR', 'WATER': 'WA',
    'STONE': 'ST', 'TREE': 'TR', 'WOOD': 'WO', 'PATH': 'PA', 'COAL': 'CO',
    'IRON': 'IR', 'DIAMOND': 'DI', 'CRAFTING_TABLE': 'CT', 'FURNACE': 'FU',
    'SAND': 'SA', 'LAVA': 'LA', 'PLANT': 'PL', 'RIPE_PLANT': 'RP',
    'WALL': 'WL', 'DARKNESS': 'DK', 'WALL_MOSS': 'WM', 'STALAGMITE': 'SG',
    'SAPPHIRE': 'SP', 'RUBY': 'RU', 'CHEST': 'CH', 'FOUNTAIN': 'FO',
    'FIRE_GRASS': 'FG', 'ICE_GRASS': 'IG', 'GRAVEL': 'GV', 'FIRE_TREE': 'FT',
    'ICE_SHRUB': 'IS', 'ENCHANTMENT_TABLE_FIRE': 'EF',
    'ENCHANTMENT_TABLE_ICE': 'EI', 'NECROMANCER': 'NE', 'GRAVE': 'G1',
    'GRAVE2': 'G2', 'GRAVE3': 'G3', 'NECROMANCER_VULNERABLE': 'NV',
}
OFFSCREEN_DIRECTIONS = ('west', 'east', 'north', 'northwest', 'northeast',
                        'south', 'southwest', 'southeast')
# Indices follow the current renderer's texture lists in constants.py.
MOB_NAMES = (
    ('zombie', 'gnome warrior', 'orc soldier', 'lizard', 'knight', 'troll', 'pigman', 'frost troll'),
    ('cow', 'bat', 'snail'),
    ('skeleton', 'gnome archer', 'orc mage', 'kobold', 'knight archer', 'deep thing', 'fire elemental', 'ice elemental'),
    ('arrow', 'dagger', 'fireball', 'iceball', 'arrow', 'slimeball', 'fireball', 'iceball'),
    ('arrow', 'dagger', 'fireball', 'iceball', 'arrow', 'slimeball', 'fireball', 'iceball'),
)
# Minimum material requirements of actions executed in do_crafting, before
# concurrent trades. Do not infer feasibility from darkness or hidden state.
CRAFT_REQUIREMENTS = {
    11: {'wood': 1}, 12: {'wood': 1, 'stone': 1},
    13: {'wood': 1, 'stone': 1, 'iron': 1, 'coal': 1},
    20: {'wood': 1, 'diamond': 3}, 14: {'wood': 1},
    15: {'wood': 1, 'stone': 1},
    16: {'wood': 1, 'stone': 1, 'iron': 1, 'coal': 1},
    21: {'wood': 1, 'diamond': 2}, 22: {'iron': 3, 'coal': 3},
    23: {'diamond': 3}, 25: {'wood': 1, 'stone': 1},
    38: {'wood': 1, 'coal': 1},
}
PLACEMENT_REQUIREMENTS = {7: {'stone': 1}, 8: {'wood': 2}, 9: {'stone': 1},
                          10: {'sapling': 1}, 28: {'torches': 1}}
TOOL_UPGRADES = {11: (11, 1), 12: (11, 2), 13: (11, 3), 20: (11, 4),
                 14: (12, 1), 15: (12, 2), 16: (12, 3), 21: (12, 4)}


def relative_location(y, x):
    dy, dx = y - 4, x - 5
    parts = []
    if dy: parts.append(f'{abs(dy)} {"south" if dy > 0 else "north"}')
    if dx: parts.append(f'{abs(dx)} {"east" if dx > 0 else "west"}')
    return ', '.join(parts) if parts else 'at your position'


def exclusion_reason(obs, action):
    """Exclude only actions provably ineffective under the current game code."""
    if len(obs) != 8746:
        raise ValueError(f'Expected compact 8746-feature observation, got {len(obs)}')
    tail = np.asarray(obs[8613:], dtype=float)
    dashboard = tail[:60].reshape(3, 20)
    if dashboard[0, 1] <= .5:
        return 'dead'
    if tail[126] > .5 or tail[127] > .5:
        return 'sleeping_or_resting'
    requirements = CRAFT_REQUIREMENTS.get(int(action)) or PLACEMENT_REQUIREMENTS.get(int(action))
    if requirements:
        inventory = np.rint((tail[84:94] * 10) ** 2)
        materials = dict(zip(INVENTORY[:10], inventory))
        if any(materials[name] < amount for name, amount in requirements.items()):
            return ('craft_insufficient_materials' if int(action) in CRAFT_REQUIREMENTS
                    else 'placement_insufficient_materials')
    if int(action) in TOOL_UPGRADES:
        index, tier = TOOL_UPGRADES[int(action)]
        if round(tail[84 + index] * 4) >= tier:
            return 'craft_tool_not_upgrade'
    return None


def action_name(value, actions):
    if value in (52, 53):
        return f'GIVE_TO_TEAMMATE_{value - 51}'
    if value in range(54, 60):
        return 'REQUEST_' + REQUESTS[9 + value - 54]
    if value not in actions:
        raise ValueError(f'Unknown action ID {value}')
    return actions[value]


def allowed(role, actions):
    ids = COMMON + ([54] if role == 'MINER' else [55] if role == 'CRAFTER' else BUILDER)
    return [action_name(i, actions) for i in ids]


def split_for(tid, seed, val, test):
    digest = hashlib.sha256(f'{seed}:{tid}'.encode()).digest()
    number = int.from_bytes(digest[:8], 'big') / 2**64
    return 'validation' if number < val else 'test' if number < val + test else 'train'


def describe_obs(obs, agent, classes):
    """Decode renderer_symbolic.py's 3-agent, 9x11 local observation layout."""
    if len(obs) != 8746:
        raise ValueError(f'Expected compact 8746-feature observation, got {len(obs)}')
    blocks, items = classes['BlockType'], classes['ItemType']
    nb, ni, n = len(blocks), len(items), 3
    channels = nb + ni + 40 + n + 2
    grid_size = 9 * 11 * channels
    grid = np.asarray(obs[:grid_size]).reshape(9, 11, channels)
    light = grid[:, :, -1] > .5
    hot = grid[:, :, :nb]
    if np.any(np.abs(hot.sum(axis=-1)[light] - 1) > 1e-4):
        raise ValueError('Incompatible map: illuminated blocks are not one-hot')
    indices = hot.argmax(axis=-1)
    used = sorted(set(indices[light].tolist()))
    if set(blocks.values()) != set(BLOCK_SYMBOLS) or len(set(BLOCK_SYMBOLS.values())) != len(BLOCK_SYMBOLS):
        raise ValueError('Unsupported block names or duplicate map symbols')
    rows = [' '.join(BLOCK_SYMBOLS[blocks[int(indices[y,x])]] if light[y,x] else '??' for x in range(11)) for y in range(9)]
    lines = ['Local 9x11 map: north is up, east is right; you are at row 4, column 5 (zero-based). ?? is unlit/unknown.',
             *rows, 'Legend: ' + ', '.join(f'{BLOCK_SYMBOLS[blocks[i]]}={blocks[i].lower().replace("_", " ")}' for i in used)]
    nearest = {}
    for y in range(9):
        for x in range(11):
            if not light[y, x]:
                continue
            name = blocks[int(indices[y, x])]
            if name in {'GRASS', 'PATH', 'SAND', 'OUT_OF_BOUNDS', 'DARKNESS', 'INVALID'}:
                continue
            candidate = (abs(y-4) + abs(x-5), y, x)
            if name not in nearest or candidate < nearest[name]:
                nearest[name] = candidate
    lines.append('Nearest visible terrain by type (distance ignores obstacles): ' +
                 ('; '.join(f'{name.lower().replace("_", " ")}: {relative_location(y, x)}'
                            for name, (_, y, x) in sorted(nearest.items())) if nearest else 'none'))
    objects = []
    offsets = [a for a in range(n) if a != agent]
    for y in range(9):
        for x in range(11):
            if not light[y, x]:
                continue
            cell = grid[y, x]
            names = []
            for i in np.flatnonzero(cell[nb:nb+ni] > .5):
                if int(i) != 0:
                    names.append('item ' + items[int(i)])
            for i in np.flatnonzero(cell[nb+ni:nb+ni+40] > .5):
                category, type_id = divmod(int(i), 8)
                kind = ['melee mob', 'passive mob', 'ranged mob', 'mob projectile', 'player projectile'][category]
                label = MOB_NAMES[category][type_id] if type_id < len(MOB_NAMES[category]) else 'unknown'
                names.append(f'{label} ({kind}, type {type_id})')
            mates = cell[nb+ni+40:nb+ni+40+n]
            for i in np.flatnonzero(mates[1:] > .5):
                names.append(f'teammate {int(i)+1} (agent {offsets[int(i)]})')
            if names:
                objects.append(f'{relative_location(y, x)}: ' + ', '.join(names))
    lines.append('Visible overlays: ' + ('; '.join(objects) if objects else 'none'))
    tail = np.asarray(obs[grid_size:], dtype=float)
    # Dashboard is reordered: self first, then other agents in ascending ID.
    dashboard = tail[:60].reshape(3, 20)
    others = [agent] + offsets
    for slot, row in enumerate(dashboard):
        role_bits = row[2:5]
        role = ROLES[int(role_bits.argmax())] if role_bits.max() > .5 else 'UNKNOWN'
        requests = [REQUESTS[i] for i in np.flatnonzero(row[5:20] > .5)]
        label = 'You' if slot == 0 else f'Teammate {slot} (agent {others[slot]})'
        lines.append(f'{label}: health={row[0]*10:.3g}, alive={bool(row[1] > .5)}, role={role}, request={", ".join(requests) if requests else "none"}')
    role = ROLES[int(dashboard[0, 2:5].argmax())] if dashboard[0, 2:5].max() > .5 else 'UNKNOWN'
    directions = tail[60:84].reshape(3, 8)
    for slot in (1, 2):
        ix = np.flatnonzero(directions[slot] > .5)
        if len(ix):
            lines.append(f'Teammate {slot} off-screen direction: {OFFSCREEN_DIRECTIONS[int(ix[0])]}')
    p = 84
    inv = tail[p:p+16].copy(); p += 16
    inv[:10] = np.rint((inv[:10]*10)**2)
    inv[11:13] *= 4
    lines.append('Inventory: ' + ', '.join(f'{k}={v:.3g}' for k, v in zip(INVENTORY, inv)))
    potions = np.rint((tail[p:p+6]*10)**2); p += 6
    lines.append('Potion counts (red, green, blue, pink, cyan, yellow): ' + ', '.join(f'{v:.0f}' for v in potions))
    stats = tail[p:p+8]*10; p += 8
    lines.append('Stats: ' + ', '.join(f'{k}={v:.3g}' for k, v in zip(STATS, stats)))
    facing = tail[p:p+4]; p += 4
    lines.append('Facing: ' + (['west', 'east', 'north', 'south'][int(facing.argmax())] if facing.max() > .5 else 'unknown'))
    armour = tail[p:p+4]*2; p += 4
    enchant = tail[p:p+4]; p += 4
    lines.append('Armour tiers (four slots): ' + ', '.join(f'{v:.3g}' for v in armour))
    lines.append('Armour enchantments (four slots): ' + ', '.join(f'{v:.3g}' for v in enchant))
    special = tail[p:p+3]; p += 3
    level = tail[p:p+4]; p += 4
    lines.append('Sleeping={:.3g}, resting={:.3g}, learned_spells={:.3g}'.format(*special))
    lines.append('Light level={:.3g}, dungeon level={:.3g}, level cleared={:.3g}, boss vulnerable={:.3g}'.format(level[0], level[1]*10, level[2], level[3]))
    if p != len(tail):
        raise ValueError(f'Unexpected observation tail length: {len(tail)}')
    return '\n'.join(lines), role, bool(dashboard[0, 1] > .5)


def scan(parts, db, mapping, skip_bad):
    episodes, bad = {}, []
    columns = ['trajectory_id', 'step_id', 'task_idx', 'instruction_idx', 'instruction_done', 'done_all', 'actions']
    for index, path in enumerate(parts, 1):
        try:
            rows = pq.read_table(path, columns=columns).to_pylist()
        except Exception as exc:
            if not skip_bad:
                raise RuntimeError(f'Cannot read {path}; use --skip-unreadable to report and skip it') from exc
            bad.append({'file': str(path), 'error': str(exc)})
            continue
        inserts = []
        for row in rows:
            tid, step, task, instr = (int(row[x]) for x in ('trajectory_id', 'step_id', 'task_idx', 'instruction_idx'))
            if step < 0 or not 0 <= instr < len(mapping) or mapping[instr] != task:
                raise ValueError(f'Invalid step or instruction/task mapping in {path}: {tid}, {step}, {instr}, {task}')
            ep = episodes.setdefault(tid, {'task': task, 'instruction': instr, 'count': 0, 'min': step, 'max': step, 'terminals': [], 'successes': []})
            if (task, instr) != (ep['task'], ep['instruction']):
                raise ValueError(f'Inconsistent labels in trajectory {tid}')
            ep['count'] += 1; ep['min'] = min(ep['min'], step); ep['max'] = max(ep['max'], step)
            if row['done_all']: ep['terminals'].append(step)
            if row['instruction_done'] > 0: ep['successes'].append(step)
            inserts.append((tid, step, json.dumps(row['actions'])))
        try:
            db.executemany('INSERT INTO steps VALUES (?, ?, ?)', inserts)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f'Duplicate trajectory/step in {path}') from exc
        if index % 250 == 0:
            db.commit()
            print(f'Scanned {index}/{len(parts)} shards, {len(episodes)} trajectories', file=sys.stderr, flush=True)
    db.commit()
    valid = {tid: ep for tid, ep in episodes.items() if ep['min'] == 0 and ep['max'] == ep['count']-1 and ep['terminals'] == [ep['max']] and ep['successes'] == [ep['max']]}
    return valid, bad, len(episodes)


class HubSplitWriter:
    """Upload bounded in-memory JSONL shards; never create output files locally."""
    def __init__(self, api, repo_id, split, limit):
        self.api, self.repo_id, self.split, self.limit = api, repo_id, split, limit
        self.buffer = io.BytesIO()
        self.index = 0

    def write(self, text):
        data = text.encode('utf-8')
        if self.buffer.tell() and self.buffer.tell() + len(data) > self.limit:
            self.flush()
        self.buffer.write(data)

    def flush(self):
        if not self.buffer.tell():
            return
        name = f'data/{self.split}-{self.index:05d}.jsonl'
        self.buffer.seek(0)
        self.api.upload_file(path_or_fileobj=self.buffer, path_in_repo=name,
                             repo_id=self.repo_id, repo_type='dataset',
                             commit_message=f'Upload {name}')
        print(f'Uploaded {name}', file=sys.stderr, flush=True)
        self.index += 1
        self.buffer.close()
        self.buffer = io.BytesIO()

    def close(self):
        self.flush()
        self.buffer.close()

    def abort(self):
        self.buffer.close()


def hub_dataset_card(counts):
    splits = ''.join(f'      - split: {name}\n        path: data/{name}-*.jsonl\n'
                     for name in ('train', 'validation', 'test') if counts[name])
    return '''---
configs:
  - config_name: default
    data_files:
''' + splits + '''---

# MAPPO trajectories converted to per-agent SFT examples

Each record contains `messages` and provenance `metadata`. Observations are local
and precede the target action. Splits are assigned by trajectory ID. The source
contains successful team episodes, not verified optimal actions or reasoning.
This is mappo_sft_v2: corrected off-screen directions, named terrain and entities,
and conservative filtering of sleep/rest and resource-impossible actions.
Policy-selected action history may contain actions ignored by the environment.
See `report.json` for provenance, limitations and conversion statistics.
'''


def build(args):
    parts = sorted(args.parts.expanduser().resolve().glob('*.parquet'))
    if args.max_shards: parts = parts[:args.max_shards]
    if not parts: raise ValueError('No Parquet shards found')
    tasks, instructions, mapping = load_scenarios(args.scenario_file)
    dimensions, classes = read_layout(args.constants_file)
    if dimensions != (9, 11) or len(classes['BlockType']) != 37 or len(classes['ItemType']) != 5:
        raise ValueError('Unsupported observation layout')
    renderer = Renderer(config_name='mappo_sft')
    output = args.output.expanduser().resolve() if args.output else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    counts, action_counts, role_counts, task_counts = Counter(), Counter(), Counter(), Counter()
    skipped, skipped_actions = Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix='mappo_sft_') as tmp, sqlite3.connect(str(Path(tmp)/'index.sqlite')) as db:
        db.execute('CREATE TABLE steps (tid INTEGER, step INTEGER, actions TEXT, PRIMARY KEY (tid, step)) WITHOUT ROWID')
        valid, bad, total = scan(parts, db, mapping, args.skip_unreadable)
        if not valid: raise ValueError('No complete successful episodes found in readable shards')
        unreadable = {x['file'] for x in bad}
        api = None
        if args.hf_repo_id:
            from huggingface_hub import HfApi
            api = HfApi()
            if args.hf_use_empty_repo:
                existing = set(api.list_repo_files(repo_id=args.hf_repo_id, repo_type='dataset'))
                unexpected = existing - {'.gitattributes'}
                if unexpected:
                    raise ValueError(f'Existing repository {args.hf_repo_id} is not empty: {sorted(unexpected)[:20]}; refusing to write')
                print(f'Using existing empty dataset repository {args.hf_repo_id}', file=sys.stderr, flush=True)
            else:
                # Refuse to overwrite or mix with any existing repository.
                api.create_repo(repo_id=args.hf_repo_id, repo_type='dataset', private=not args.hf_public, exist_ok=False)
            files = {name: HubSplitWriter(api, args.hf_repo_id, name, args.hf_shard_mib * 1024 * 1024) for name in ('train', 'validation', 'test')}
        else:
            files = {name: (output / f'{name}.jsonl').open('x', encoding='utf-8') for name in ('train', 'validation', 'test')}
        cache = OrderedDict()
        try:
            for idx, path in enumerate(parts, 1):
                if str(path) in unreadable: continue
                f = pq.ParquetFile(path)
                for batch in f.iter_batches(columns=['trajectory_id', 'step_id', 'obs', 'actions'], batch_size=128):
                    table = pa.Table.from_batches([batch], schema=pa.schema([f.schema_arrow.field(name) for name in batch.schema.names], metadata=f.schema_arrow.metadata))
                    data = table_to_arrays(table)
                    for i in range(len(batch)):
                        tid, step = int(data['trajectory_id'][i]), int(data['step_id'][i])
                        if tid not in valid: continue
                        ep = valid[tid]
                        if tid not in cache:
                            cache[tid] = {int(s): json.loads(a) for s, a in db.execute('SELECT step, actions FROM steps WHERE tid=? ORDER BY step', (tid,))}
                            if len(cache) > 256: cache.popitem(last=False)
                        cache.move_to_end(tid)
                        for agent in range(3):
                            action_id = int(data['actions'][i, agent])
                            reason = exclusion_reason(data['obs'][i, agent], action_id)
                            if reason:
                                skipped[reason] += 1
                                skipped_actions[(reason, action_name(action_id, classes['Action']))] += 1
                                continue
                            obs, role, alive = describe_obs(data['obs'][i, agent], agent, classes)
                            if not alive: raise ValueError(f'Inconsistent alive flag for {tid}/{step}/{agent}')
                            actions = allowed(role, classes['Action'])
                            if role == 'UNKNOWN': raise ValueError(f'Unknown role for {tid}/{step}/{agent}')
                            target = action_name(action_id, classes['Action'])
                            if target not in actions: raise ValueError(f'Action {target} invalid for {role}, trajectory {tid}, step {step}')
                            history = [action_name(int(cache[tid][j][agent]), classes['Action']) for j in range(max(0, step-args.history), step)]
                            prompt = renderer.render({'goal': instructions[ep['instruction']], 'obs': obs, 'act': actions, 'action_history': ', '.join(history) if history else 'None.'})
                            split = split_for(tid, args.seed, args.val_fraction, args.test_fraction)
                            record = {'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': f'<action>{target}</action>'}],
                                      'metadata': {'trajectory_id': tid, 'step_id': step, 'agent_id': agent, 'task_idx': ep['task'], 'task_name': tasks[ep['task']], 'instruction_idx': ep['instruction'], 'action_id': action_id, 'role': role, 'source_shard': path.name, 'format_version': 'mappo_sft_v2'}}
                            files[split].write(json.dumps(record, ensure_ascii=False) + '\n')
                            counts[split] += 1
                            action_counts[target] += 1
                            role_counts[role] += 1
                            task_counts[tasks[ep['task']]] += 1
                if idx % 100 == 0: print(f'Converted {idx}/{len(parts)} shards; {sum(counts.values())} examples', file=sys.stderr, flush=True)
        except BaseException:
            for f in files.values():
                if api is not None:
                    f.abort()
                else:
                    f.close()
            raise
        for f in files.values():
            f.close()
    report = {'format_version': 'mappo_sft_v2', 'source_parts': str(args.parts.resolve()), 'scenario_file': str(args.scenario_file.resolve()), 'scenario_sha256': hashlib.sha256(args.scenario_file.read_bytes()).hexdigest(), 'constants_file': str(args.constants_file.resolve()), 'constants_sha256': hashlib.sha256(args.constants_file.read_bytes()).hexdigest(), 'scenario_note': 'Reconstructed from current source; instruction/task ID consistency was checked, but original metadata was unavailable.', 'shards_scanned': len(parts), 'unreadable_shards': bad, 'trajectories_seen': total, 'complete_successful_trajectories': len(valid), 'rejected_trajectories': total-len(valid), 'examples': {k: counts[k] for k in ('train', 'validation', 'test')}, 'actions': dict(action_counts.most_common()), 'roles': dict(role_counts), 'tasks': dict(task_counts), 'action_ids': {action_name(i, classes['Action']): i for i in sorted(set(COMMON + BUILDER + [54, 55]))}, 'skipped_by_reason': dict(skipped), 'skipped_actions': {f'{reason}:{action}': count for (reason, action), count in skipped_actions.items()}, 'filter_note': 'Sleep/rest/death, insufficient craft or placement materials and non-upgrading tool crafts are filtered using current game logic. Other ineffective actions may remain.', 'seed': args.seed, 'validation_fraction': args.val_fraction, 'test_fraction': args.test_fraction, 'history_length': args.history}
    report['hf_repo_id'] = args.hf_repo_id
    report['source_hf_repo_id'] = args.source_hf_repo_id
    report['source_hf_revision'] = args.source_hf_revision
    report['converter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['prompt_template_sha256'] = hashlib.sha256((ROOT / 'craftext_prompt/templates/mappo_sft/prompt.txt').read_bytes()).hexdigest()
    source_root = args.constants_file.resolve().parent
    report['game_logic_sha256'] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in {
            'actions': source_root / 'game_logic/actions.py',
            'step': source_root / 'game_logic/step.py',
            'renderer_symbolic': source_root / 'renderer/renderer_symbolic.py',
        }.items() if path.is_file()
    }
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    if api is not None:
        api.upload_file(path_or_fileobj=report_bytes, path_in_repo='report.json', repo_id=args.hf_repo_id, repo_type='dataset', commit_message='Upload conversion report')
        # Publish card last. Its presence indicates the conversion reached completion.
        api.upload_file(path_or_fileobj=hub_dataset_card(counts).encode('utf-8'), path_in_repo='README.md', repo_id=args.hf_repo_id, repo_type='dataset', commit_message='Publish completed dataset card')
    else:
        (output / 'report.json').write_bytes(report_bytes)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parts', type=Path, default=DEFAULT_PARTS)
    p.add_argument('--scenario-file', type=Path, default=DEFAULT_SCENARIOS)
    p.add_argument('--constants-file', type=Path, default=CRAFTAX/'craftax/craftax_together_talens/constants.py')
    destination = p.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output', type=Path, help='New local output directory; never overwrite existing output')
    destination.add_argument('--hf-repo-id', help='New Hugging Face dataset repository owner/name')
    p.add_argument('--hf-use-empty-repo', action='store_true', help='Use an existing dataset repo only if it contains no files besides .gitattributes')
    p.add_argument('--hf-public', action='store_true', help='Create the new Hub repository public (default: private)')
    p.add_argument('--hf-shard-mib', type=int, default=64, help='Approximate uncompressed JSONL shard size in MiB')
    p.add_argument('--source-hf-repo-id', help='Source repository ID to record in report (does not download data)')
    p.add_argument('--source-hf-revision', help='Source repository revision to record in report')
    p.add_argument('--history', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=.05)
    p.add_argument('--test-fraction', type=float, default=.05)
    p.add_argument('--max-shards', type=int, default=0, help='Limit input for smoke testing; 0 means all')
    p.add_argument('--skip-unreadable', action='store_true', help='Skip and report unreadable shards instead of failing')
    args = p.parse_args()
    if args.history < 0 or args.max_shards < 0 or args.hf_shard_mib < 1 or not (0 <= args.val_fraction < 1 and 0 <= args.test_fraction < 1 and args.val_fraction + args.test_fraction < 1): p.error('Invalid history, shard limit, shard size, or split fractions')
    if args.hf_public and not args.hf_repo_id: p.error('--hf-public requires --hf-repo-id')
    if args.hf_use_empty_repo and not args.hf_repo_id: p.error('--hf-use-empty-repo requires --hf-repo-id')
    if args.hf_use_empty_repo and args.hf_public: p.error('--hf-public only controls creation of a new repository')
    build(args)


if __name__ == '__main__':
    main()
