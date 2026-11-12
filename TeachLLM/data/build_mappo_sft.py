#!/usr/bin/env python3
"""Convert successful cooperative MAPPO trajectories into structured SFT JSONL."""
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
PROJECT = ROOT.parent
WORK = PROJECT.parent
CRAFTAX = PROJECT / 'MultyAgentCrafText/MA-Craftax'
from trajectory_io import table_to_arrays, load_scenarios, read_layout

DEFAULT_PARTS = WORK / 'HF/datasets/mappo_100k_20261101_210728_parts'
DEFAULT_SCENARIOS = PROJECT / 'CrafText/craftext/dataset/scenarious/jax_conditional_achievements/instructions.py'
REQUESTS = ['FOOD', 'DRINK', 'WOOD', 'STONE', 'IRON', 'COAL', 'DIAMOND', 'RUBY', 'SAPPHIRE', 'PICKAXE', 'SWORD', 'ARMOUR', 'BOW', 'ARROWS', 'TORCHES']
ROLES = ['MINER', 'BUILDER', 'CRAFTER']
INVENTORY = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapphire', 'ruby', 'sapling', 'torches', 'arrows', 'books', 'pickaxe_tier', 'sword_tier', 'sword_enchantment', 'bow_enchantment', 'bow']
STATS = ['food', 'drink', 'energy', 'mana', 'xp', 'dexterity', 'strength', 'intelligence']
COMMON = [0, 1, 2, 3, 4, 5, 6, 17, 19, 52, 53]
BUILDER = [45, 46, 47, 48, 49, 50, 51, 7, 8, 9, 10, 28, 11, 12, 13, 20, 14, 15, 16, 21, 22, 23, 25, 38]

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


def n(v):
 return round(float(v),6)

def describe_obs(obs, agent, history, classes):
 if len(obs) != 8746: raise ValueError(f'Expected compact 8746-feature observation, got {len(obs)}')
 grid=obs[:8613].reshape(9,11,87); tail=obs[8613:]; dash=tail[:60].reshape(3,20); pos=[agent]+[x for x in range(3) if x!=agent]
 terrain=[]; unknown_items=[]; objects=[]; nearest={}
 for y in range(9):
  names=[]
  for x in range(11):
   cell=grid[y,x];rel=[y-4,x-5]
   if cell[-1]<.5:
    names.append('unknown'); continue
   one=cell[:37]
   if abs(float(one.sum())-1)>1e-4: raise ValueError('Illuminated terrain is not one-hot')
   name=classes['BlockType'][int(one.argmax())].lower(); names.append(name)
   if name not in {'grass','sand','path','out_of_bounds','darkness','invalid'}:
    val=(abs(rel[0])+abs(rel[1]),y,x)
    if name not in nearest or val<nearest[name]:nearest[name]=val
   item=cell[37:42]; active=np.flatnonzero(item>.5)
   if len(active)==0: unknown_items.append(rel)
   elif len(active)==1 and active[0]>0:objects.append({'kind':'item','type':classes['ItemType'][int(active[0])].lower(),'relative':rel})
   elif len(active)>1:raise ValueError('Unexpected item encoding')
   for i in np.flatnonzero(cell[42:82]>.5):
    cat,ty=divmod(int(i),8); kind=['melee_mob','passive_mob','ranged_mob','mob_projectile','player_projectile'][cat]; typ=MOB_NAMES[cat][ty] if ty<len(MOB_NAMES[cat]) else 'unknown';objects.append({'kind':kind,'type':typ.replace(' ','_'),'type_id':ty,'relative':rel})
   for j in np.flatnonzero(cell[82:85]>.5):
    j=int(j); objects.append({'kind':'player','id':f'agent_{pos[j]}','relative':rel,
                              'cell_alive_marker':bool(cell[85]>.5)})
   if cell[85]>.5 and not np.any(cell[82:85]>.5):
    objects.append({'kind':'player_alive_marker','relative':rel})
  terrain.append(' '.join(names))
 team=[];directions=tail[60:84].reshape(3,8)
 for j in (1,2):
  d=dash[j]; roleid=np.flatnonzero(d[2:5]>.5);rq=np.flatnonzero(d[5:20]>.5);di=np.flatnonzero(directions[j]>.5)
  team.append({'slot':j,'id':f'agent_{pos[j]}','role':ROLES[int(roleid[0])] if len(roleid)==1 else 'unknown','health':n(d[0]*10),'alive':bool(d[1]>.5),'request':REQUESTS[int(rq[0])] if len(rq)==1 else None,'offscreen_direction':OFFSCREEN_DIRECTIONS[int(di[0])] if len(di)==1 else None})
 d=dash[0];rid=np.flatnonzero(d[2:5]>.5)
 if len(rid)!=1: raise ValueError('Unknown or invalid agent role')
 own_request=np.flatnonzero(d[5:20]>.5)
 invraw=tail[84:100]; inv={name:(int(np.rint((float(v)*10)**2)) if i<10 else int(round(float(v)*4)) if i in (11,12) else n(v)) for i,(name,v) in enumerate(zip(INVENTORY,invraw))}
 potion=[int(round((float(v)*10)**2)) for v in tail[100:106]]
 stats={k:n(v*10) for k,v in zip(STATS,tail[106:114])}
 facing_bits=tail[114:118]; fi=np.flatnonzero(facing_bits>.5);facing=['west','east','north','south'][int(fi[0])] if len(fi)==1 else 'unknown'
 off=tail[130:133]
 return {'id':f'agent_{agent}','role':ROLES[int(rid[0])],'facing':facing,'health':n(d[0]*10),'alive':bool(d[1]>.5),
  'request':REQUESTS[int(own_request[0])] if len(own_request)==1 else None,
  'inventory':inv,'potions':dict(zip(['red','green','blue','pink','cyan','yellow'],potion)),
  'stats':stats,'armor_tiers':[n(v*2) for v in tail[118:122]],'armor_enchantments':[n(v) for v in tail[122:126]],
  'status':{'sleeping':bool(tail[126]>.5),'resting':bool(tail[127]>.5),'learned_spells':n(tail[128])},
  'world':{'light_level':n(tail[129]),'dungeon_level':n(tail[130]*10),'level_cleared':bool(tail[131]>.5),'boss_vulnerable':bool(tail[132]>.5)},
  'observation':{'terrain_rows':terrain,'nearest_visible_terrain':[{'type':name,'relative':[y-4,x-5]} for name,(_,y,x) in sorted(nearest.items())],
                 'visible_entities':objects,'item_state_unknown':unknown_items},
  'teammates':team,'selected_action_history':history,'allowed_actions':allowed(ROLES[int(rid[0])],classes['Action'])}



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
    """Bounded in-memory JSONL chunks; output is uploaded without local shards."""
    def __init__(self, api, repo_id, mode, split, limit):
        self.api, self.repo_id, self.mode, self.split, self.limit = api, repo_id, mode, split, limit
        self.buffer = io.BytesIO()
        self.index = 0

    def write(self, text):
        data = text.encode('utf-8')
        if self.buffer.tell() and self.buffer.tell() + len(data) > self.limit:
            self.flush()
        self.buffer.write(data)

    def flush(self):
        if not self.buffer.tell(): return
        name = f'data/{self.mode}/{self.split}-{self.index:05d}.jsonl'
        self.buffer.seek(0)
        self.api.upload_file(path_or_fileobj=self.buffer, path_in_repo=name,
                             repo_id=self.repo_id, repo_type='dataset',
                             commit_message=f'Upload {name}')
        print(f'Uploaded {name}', file=sys.stderr, flush=True)
        self.index += 1
        self.buffer.close(); self.buffer = io.BytesIO()

    def close(self):
        self.flush(); self.buffer.close()

    def abort(self):
        self.buffer.close()


def hub_dataset_card(counts, modes):
    configs = []
    for mode in modes:
        paths = [(split, counts[mode, split]) for split in ('train', 'validation', 'test')]
        if not any(count for _, count in paths): continue
        configs.append(f'  - config_name: {mode}\n    data_files:\n' + ''.join(
            f'      - split: {split}\n        path: data/{mode}/{split}-*.jsonl\n'
            for split, count in paths if count))
    return ('---\nconfigs:\n' + ''.join(configs) + '---\n\n'
            '# Cooperative MAPPO trajectories for SFT\n\n'
            'Each row contains messages and provenance metadata. The individual config uses '
            'one agent’s local observation; the team config uses all three local observations. '
            'All targets are policy-selected actions from successful team episodes, not proven '
            'optimal actions. See report.json for filters and provenance.\n')


def build(args):
    parts = sorted(args.parts.expanduser().resolve().glob('*.parquet'))
    if args.max_shards: parts = parts[:args.max_shards]
    if not parts: raise ValueError('No Parquet shards found')
    tasks, instructions, mapping = load_scenarios(args.scenario_file)
    dimensions, classes = read_layout(args.constants_file)
    if dimensions != (9, 11) or len(classes['BlockType']) != 37 or len(classes['ItemType']) != 5:
        raise ValueError('Unsupported observation layout')
    modes = ('individual', 'team') if args.mode == 'both' else (args.mode,)
    prompts = {mode: (ROOT / 'data/prompts' / f'{mode}.txt').read_text(encoding='utf-8').rstrip('\n') for mode in modes}
    output = args.output.expanduser().resolve() if args.output else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        for mode in modes: (output / mode).mkdir()
    counts = Counter(); action_counts = Counter(); role_counts = Counter(); task_counts = Counter()
    skipped, skipped_actions, team_reasons = Counter(), Counter(), Counter()
    team_skipped_steps = 0
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
                if unexpected: raise ValueError(f'Existing repository {args.hf_repo_id} is not empty: {sorted(unexpected)[:20]}; refusing to write')
                print(f'Using existing empty dataset repository {args.hf_repo_id}', file=sys.stderr, flush=True)
            else:
                api.create_repo(repo_id=args.hf_repo_id, repo_type='dataset', private=not args.hf_public, exist_ok=False)
            files = {(mode, split): HubSplitWriter(api, args.hf_repo_id, mode, split, args.hf_shard_mib * 1024 * 1024)
                     for mode in modes for split in ('train', 'validation', 'test')}
        else:
            files = {(mode, split): (output / mode / f'{split}.jsonl').open('x', encoding='utf-8')
                     for mode in modes for split in ('train', 'validation', 'test')}
        cache = OrderedDict()
        try:
            for idx, path in enumerate(parts, 1):
                if str(path) in unreadable: continue
                f = pq.ParquetFile(path)
                for batch in f.iter_batches(columns=['trajectory_id', 'step_id', 'obs', 'actions'], batch_size=128):
                    schema = pa.schema([f.schema_arrow.field(name) for name in batch.schema.names], metadata=f.schema_arrow.metadata)
                    data = table_to_arrays(pa.Table.from_batches([batch], schema=schema))
                    for i in range(len(batch)):
                        tid, step = int(data['trajectory_id'][i]), int(data['step_id'][i])
                        if tid not in valid: continue
                        ep = valid[tid]
                        if tid not in cache:
                            cache[tid] = {int(s): json.loads(a) for s, a in db.execute('SELECT step, actions FROM steps WHERE tid=? ORDER BY step', (tid,))}
                            if len(cache) > 256: cache.popitem(last=False)
                        cache.move_to_end(tid)
                        split = split_for(tid, args.seed, args.val_fraction, args.test_fraction)
                        meta = {'trajectory_id': tid, 'step_id': step, 'task_idx': ep['task'],
                                'task_name': tasks[ep['task']], 'instruction_idx': ep['instruction'],
                                'source_shard': path.name, 'format_version': 'mappo_structured_v1'}
                        goal = instructions[ep['instruction']]
                        views = [None, None, None]; targets = [None, None, None]; reasons = []
                        for agent in range(3):
                            action_id = int(data['actions'][i, agent]); target = action_name(action_id, classes['Action'])
                            reason = exclusion_reason(data['obs'][i, agent], action_id)
                            if reason:
                                skipped[reason] += 1; skipped_actions[(reason, target)] += 1; reasons.append(reason)
                                continue
                            history = [action_name(int(cache[tid][j][agent]), classes['Action'])
                                       for j in range(max(0, step-args.history), step)]
                            view = describe_obs(data['obs'][i, agent], agent, history, classes)
                            if target not in view['allowed_actions']:
                                raise ValueError(f'Action {target} invalid for {view["role"]}, trajectory {tid}, step {step}')
                            views[agent], targets[agent] = view, target
                            if 'individual' in modes:
                                user = {'goal': goal, **view}
                                record = {'messages': [{'role': 'system', 'content': prompts['individual']},
                                            {'role': 'user', 'content': json.dumps(user, ensure_ascii=False, separators=(',', ':'))},
                                            {'role': 'assistant', 'content': f'<action>{target}</action>'}],
                                          'metadata': {**meta, 'agent_id': agent, 'action_id': action_id, 'role': view['role']}}
                                files['individual', split].write(json.dumps(record, ensure_ascii=False)+'\n')
                                counts['individual', split] += 1; action_counts[target] += 1
                                role_counts[view['role']] += 1; task_counts[tasks[ep['task']]] += 1
                        if 'team' in modes:
                            if reasons:
                                team_skipped_steps += 1
                                team_reasons.update(set(reasons))
                            else:
                                user = {'goal': goal, 'agents': {v['id']: v for v in views}}
                                answer = '<actions>\n'+'\n'.join(f'<agent_{a}><action>{t}</action></agent_{a}>' for a,t in enumerate(targets))+'\n</actions>'
                                record = {'messages': [{'role': 'system', 'content': prompts['team']},
                                            {'role': 'user', 'content': json.dumps(user, ensure_ascii=False, separators=(',', ':'))},
                                            {'role': 'assistant', 'content': answer}],
                                          'metadata': {**meta, 'action_ids': [int(a) for a in data['actions'][i]]}}
                                files['team', split].write(json.dumps(record, ensure_ascii=False)+'\n')
                                counts['team', split] += 1
                if idx % 100 == 0: print(f'Converted {idx}/{len(parts)} shards; {sum(counts.values())} examples', file=sys.stderr, flush=True)
        except BaseException:
            for f in files.values():
                if api is not None: f.abort()
                else: f.close()
            raise
        for f in files.values(): f.close()
    source_root = args.constants_file.resolve().parent
    report = {'format_version': 'mappo_structured_v1', 'modes': list(modes),
        'source_parts': str(args.parts.resolve()), 'source_hf_repo_id': args.source_hf_repo_id,
        'source_hf_revision': args.source_hf_revision, 'hf_repo_id': args.hf_repo_id,
        'scenario_file': str(args.scenario_file.resolve()),
        'scenario_sha256': hashlib.sha256(args.scenario_file.read_bytes()).hexdigest(),
        'constants_file': str(args.constants_file.resolve()),
        'constants_sha256': hashlib.sha256(args.constants_file.read_bytes()).hexdigest(),
        'converter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prompt_sha256': {mode: hashlib.sha256((ROOT / 'data/prompts' / f'{mode}.txt').read_bytes()).hexdigest() for mode in modes},
        'game_logic_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
            'actions': source_root/'game_logic/actions.py', 'step': source_root/'game_logic/step.py',
            'renderer_symbolic': source_root/'renderer/renderer_symbolic.py'}.items() if path.is_file()},
        'scenario_note': 'Instruction text reconstructed from current scenario source; original collection metadata unavailable.',
        'shards_scanned': len(parts), 'unreadable_shards': bad, 'trajectories_seen': total,
        'complete_successful_trajectories': len(valid), 'rejected_trajectories': total-len(valid),
        'examples': {mode: {split: counts[mode, split] for split in ('train','validation','test')} for mode in modes},
        'individual_action_counts': dict(action_counts), 'individual_role_counts': dict(role_counts),
        'individual_task_counts': dict(task_counts), 'skipped_agent_steps_by_reason': dict(skipped),
        'skipped_agent_actions': {f'{reason}:{action}': count for (reason, action),count in skipped_actions.items()},
        'team_steps_skipped': team_skipped_steps, 'team_skip_reasons_overlap': dict(team_reasons),
        'filter_note': 'Using current game logic, exclude dead/sleeping/resting agents and selected crafting or placement actions that cannot succeed given own resources, plus non-upgrading tool crafts. A team step is excluded if any agent is excluded. Other ineffective actions may remain.',
        'seed': args.seed, 'validation_fraction': args.val_fraction, 'test_fraction': args.test_fraction,
        'history_length': args.history}
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    if api is not None:
        api.upload_file(path_or_fileobj=report_bytes, path_in_repo='report.json', repo_id=args.hf_repo_id, repo_type='dataset', commit_message='Upload conversion report')
        api.upload_file(path_or_fileobj=hub_dataset_card(counts, modes).encode('utf-8'), path_in_repo='README.md', repo_id=args.hf_repo_id, repo_type='dataset', commit_message='Publish completed dataset card')
    else:
        (output / 'report.json').write_bytes(report_bytes)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('individual', 'team', 'both'), default='both')
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
