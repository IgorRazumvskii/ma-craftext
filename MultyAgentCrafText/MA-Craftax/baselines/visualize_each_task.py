#!/usr/bin/env python3
"""Render one successful recorded trajectory per task from MAPPO Parquet shards.

Tasks/instruction text are reconstructed from CrafText's easy scenario source;
the original dataset metadata was not saved. The index mapping is checked
against every readable row before rendering. GIFs show local observations before
recorded actions; the terminal post-action world state is not in the dataset.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from trajectory_parquet import table_to_arrays
from visualize_mappo_trajectory import draw_frame, font, read_layout

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = ROOT.parents[1] / 'CrafText/craftext/dataset/scenarious/jax_conditional_achievements/instructions.py'
COLUMNS = ['trajectory_id', 'step_id', 'task_idx', 'instruction_idx', 'instruction_done', 'done_all']


def load_scenarios(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    easy = next((node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'easy' for t in node.targets)), None)
    if not isinstance(easy, ast.Dict):
        raise ValueError(f'Cannot find easy task dictionary in {path}')
    tasks, instructions, instruction_to_task = [], [], []
    for task_id, (key, value) in enumerate(zip(easy.keys, easy.values)):
        name = ast.literal_eval(key)
        if not isinstance(value, ast.Dict):
            raise ValueError(f'Task {name} is not a dictionary')
        fields = {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(value.keys, value.values)
                  if isinstance(k, ast.Constant) and k.value in ('instruction', 'instruction_paraphrases')}
        variants = [fields['instruction']] + fields.get('instruction_paraphrases', [])
        if not all(isinstance(text, str) for text in variants):
            raise ValueError(f'Unexpected text for task {name}')
        tasks.append(name)
        instructions.extend(variants)
        instruction_to_task.extend([task_id] * len(variants))
    return tasks, instructions, instruction_to_task


def scan(parts, mapping):
    episodes = {}
    unreadable = []
    for index, path in enumerate(parts, 1):
        try:
            f = pq.ParquetFile(path)
        except Exception as exc:
            unreadable.append({'file': str(path), 'error': str(exc)})
            continue
        try:
            for batch in f.iter_batches(columns=COLUMNS, batch_size=32768):
                data = {name: batch.column(i).to_numpy(zero_copy_only=False) for i, name in enumerate(COLUMNS)}
                ids, inv = np.unique(data['trajectory_id'], return_inverse=True)
                for j, tid in enumerate(ids):
                    mask = inv == j
                    instr = np.unique(data['instruction_idx'][mask])
                    task = np.unique(data['task_idx'][mask])
                    if len(instr) != 1 or len(task) != 1:
                        raise ValueError(f'Trajectory {tid}: multiple instruction or task IDs')
                    instr, task, tid = int(instr[0]), int(task[0]), int(tid)
                    if not 0 <= instr < len(mapping) or mapping[instr] != task:
                        raise ValueError(f'Instruction/task mapping mismatch for trajectory {tid}: instruction={instr}, task={task}')
                    step = data['step_id'][mask]
                    successful = (data['instruction_done'][mask] > 0) & data['done_all'][mask]
                    found = episodes.setdefault(tid, {'tid': tid, 'task': task, 'instruction': instr, 'count': 0,
                                                       'min_step': None, 'max_step': -1, 'success_steps': [], 'paths': set()})
                    if found['task'] != task or found['instruction'] != instr:
                        raise ValueError(f'Trajectory {tid}: inconsistent labels across parts')
                    found['count'] += len(step)
                    found['min_step'] = min(found['min_step'], int(step.min())) if found['min_step'] is not None else int(step.min())
                    found['max_step'] = max(found['max_step'], int(step.max()))
                    found['success_steps'].extend(map(int, step[successful]))
                    found['paths'].add(path)
        except Exception as exc:
            raise RuntimeError(f'Unable to read {path}: {exc}') from exc
        if index % 500 == 0:
            print(f'Scanned {index}/{len(parts)} shards', flush=True)
    return episodes, unreadable


def select_episodes(episodes):
    by_task = defaultdict(list)
    for episode in episodes.values():
        n = episode['count']
        if (episode['min_step'] == 0 and episode['max_step'] == n - 1
                and episode['success_steps'] == [n - 1]):
            by_task[episode['task']].append(episode)
    # Median-length example gives a typical visualization; ID breaks ties.
    return {task: sorted(items, key=lambda x: (x['count'], x['tid']))[(len(items)-1)//2]
            for task, items in by_task.items()}


def collect_selected(selected, max_frames):
    by_file = defaultdict(list)
    samples, observed_steps = {}, defaultdict(list)
    for task, episode in selected.items():
        n = episode['count']
        count = min(max_frames, n) if max_frames else n
        samples[task] = {'want': set(np.linspace(0, n - 1, count, dtype=np.int64).tolist()), 'frames': {}}
        for path in episode['paths']:
            by_file[path].append(task)
    for path, tasks in by_file.items():
        f = pq.ParquetFile(path)
        by_id = {selected[t]['tid']: t for t in tasks}
        for batch in f.iter_batches(batch_size=128):
            table = pa.Table.from_batches([batch], schema=f.schema_arrow)
            data = table_to_arrays(table)
            for tid in np.unique(data['trajectory_id']):
                task = by_id.get(int(tid))
                if task is None:
                    continue
                rows = np.flatnonzero(data['trajectory_id'] == tid)
                for row in rows:
                    step = int(data['step_id'][row])
                    observed_steps[task].append(step)
                    if step in samples[task]['want']:
                        if step in samples[task]['frames']:
                            raise ValueError(f'Duplicate step {step} in trajectory {tid}')
                        samples[task]['frames'][step] = {key: np.array(value[row], copy=True) for key, value in data.items()}
    for task, ep in selected.items():
        if sorted(observed_steps[task]) != list(range(ep['count'])):
            raise ValueError(f'Trajectory {ep["tid"]} has missing or duplicate steps')
        if len(samples[task]['frames']) != len(samples[task]['want']):
            raise ValueError(f'Trajectory {ep["tid"]} has missing selected frames')
    return samples


def add_header(frame, task_name, instruction, steps):
    # Wrap by rendered width, so long task names and instructions stay visible.
    width = frame.width - 24
    probe = ImageDraw.Draw(frame)
    def wrap(text, face):
        lines, current = [], ''
        for word in text.split():
            candidate = f'{current} {word}'.strip()
            if current and probe.textbbox((0, 0), candidate, font=face)[2] > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines
    heading_font, body_font = font(20), font(14)
    heading = wrap(f'Task: {task_name}   |   Completed in {steps} environment steps', heading_font)
    detail = wrap(instruction, body_font)
    bar = 14 + len(heading) * 25 + len(detail) * 19
    image = Image.new('RGB', (frame.width, frame.height + bar), '#e6edf2')
    image.paste(frame, (0, bar))
    draw = ImageDraw.Draw(image)
    y = 7
    for line in heading:
        draw.text((12, y), line, fill='#142334', font=heading_font)
        y += 25
    for line in detail:
        draw.text((12, y), line, fill='#33495a', font=body_font)
        y += 19
    return image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True, help='Directory containing *_parts folder')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--scenario-file', type=Path, default=DEFAULT_SCENARIOS)
    p.add_argument('--fps', type=float, default=4)
    p.add_argument('--max-frames', type=int, default=120, help='Sample across whole trajectory; 0 means all')
    p.add_argument('--tile-size', type=int, default=28)
    args = p.parse_args()
    if args.fps <= 0 or args.max_frames < 0 or args.tile_size < 16:
        p.error('fps must be positive; max-frames >= 0; tile-size >= 16')
    folder = args.dataset.expanduser().resolve()
    if folder.name.endswith('_parts'):
        parts_dirs = [folder]
    else:
        parts_dirs = sorted(folder.glob('*_parts'))
    if len(parts_dirs) != 1:
        p.error(f'Expected one *_parts directory, found {len(parts_dirs)}; pass the desired _parts directory')
    parts = sorted(parts_dirs[0].glob('*.parquet'))
    if not parts:
        p.error(f'No Parquet files found in {parts_dirs[0]}')
    tasks, instructions, mapping = load_scenarios(args.scenario_file)
    episodes, unreadable = scan(parts, mapping)
    selected = select_episodes(episodes)
    if not selected:
        raise RuntimeError('No complete successful trajectories found')
    print(f'Found {len(selected)} tasks, {len(episodes)} trajectories, {len(unreadable)} unreadable shards', flush=True)
    samples = collect_selected(selected, args.max_frames)
    layout = read_layout()
    meta = {'task_names': tasks, 'instruction_text_by_idx': instructions,
            'agents': ['agent_0', 'agent_1', 'agent_2']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for task, ep in sorted(selected.items()):
        tid, steps = ep['tid'], ep['count']
        safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', tasks[task]).strip('_')
        output = args.output_dir / f'{task:03d}_{safe_name}_trajectory_{tid}.gif'
        frames = []
        for step, row in sorted(samples[task]['frames'].items()):
            frame = draw_frame(row, tid, meta, layout, args.tile_size)
            frames.append(add_header(frame, tasks[task], instructions[ep['instruction']], steps))
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       duration=max(10, round(1000 / args.fps / 10) * 10), loop=0, optimize=False)
        results.append({'task_idx': task, 'task_name': tasks[task], 'instruction_idx': ep['instruction'],
                        'instruction': instructions[ep['instruction']], 'trajectory_id': tid,
                        'environment_steps': steps, 'frame_count': len(frames), 'gif': output.name})
        print(f'[{len(results)}/{len(selected)}] {tasks[task]}: {steps} steps -> {output}', flush=True)
    report = {'dataset': str(folder), 'scenario_source': str(args.scenario_file),
              'scenario_note': 'Text reconstructed from current scenario source; original dataset metadata unavailable.',
              'selection': 'median-length successful trajectory per task, tie-break trajectory ID',
              'missing_tasks': [{'task_idx': i, 'task_name': tasks[i]} for i in sorted(set(range(len(tasks))) - set(selected))],
              'unreadable_parts': unreadable, 'results': results}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Saved {len(results)} GIFs and summary.json in {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
