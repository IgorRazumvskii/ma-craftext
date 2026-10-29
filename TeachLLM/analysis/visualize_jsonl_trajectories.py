from __future__ import annotations
import argparse
import gzip
import json
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None
_GOAL_RE = re.compile('## Goal\\s*\\n(.*?)(?:\\n\\s*\\n## Observation|\\n\\s*\\n##)', re.DOTALL)

def _open_text(path: Path):
    return gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else open(path, 'rt', encoding='utf-8')

def _safe_name(value: Any) -> str:
    s = str(value)
    return ''.join((ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in s)).strip('_') or 'item'

def _parse_ids(raw: str) -> set[int]:
    out: set[int] = set()
    for part in raw.split(','):
        part = part.strip()
        if part:
            out.add(int(part))
    return out

def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _parse_goal(prompt: str) -> str:
    m = _GOAL_RE.search(prompt)
    if not m:
        return ''
    return ' '.join((m.group(1) or '').strip().split())

def _parse_grid(prompt: str) -> Optional[list[list[str]]]:
    lines = prompt.splitlines()
    start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('r\\c'):
            start = i + 1
            break
    if start < 0:
        return None
    grid: list[list[str]] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            break
        parts = stripped.split()
        if len(parts) < 2:
            break
        try:
            int(parts[0])
        except ValueError:
            break
        grid.append(parts[1:])
    return grid or None

def _collect_trajectories(path: Path, *, trajectory_ids: set[int], num_trajectories: int, max_rows: int) -> dict[int, dict[int, list[dict[str, Any]]]]:
    selected_ids = set(trajectory_ids)
    collecting_first = not selected_ids
    trajectories: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row_idx, row in enumerate(_iter_rows(path), start=1):
        if max_rows > 0 and row_idx > max_rows:
            break
        if 'trajectory_id' not in row or 'step_id' not in row:
            continue
        tid = int(row['trajectory_id'])
        if collecting_first and tid not in selected_ids:
            if len(selected_ids) >= num_trajectories:
                continue
            selected_ids.add(tid)
        if tid not in selected_ids:
            continue
        step = int(row['step_id'])
        trajectories[tid][step].append(row)
    return trajectories

def _font(size: int):
    candidates = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/Library/Fonts/Arial.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf']
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def _tile_color(sym: str) -> tuple[int, int, int]:
    colors = {'.': (88, 151, 93), ':': (190, 172, 118), '~': (72, 137, 191), 'T': (45, 114, 68), '@': (226, 91, 70), '#': (91, 94, 102), 'S': (156, 156, 163), 'C': (142, 101, 70), 'W': (86, 116, 181), 'Z': (108, 90, 144), 'P': (91, 166, 135), 'F': (211, 132, 68), 'L': (120, 120, 120)}
    if sym in colors:
        return colors[sym]
    seed = sum((ord(ch) for ch in sym))
    return (90 + seed * 37 % 120, 90 + seed * 53 % 120, 90 + seed * 71 % 120)

def _draw_grid(draw: ImageDraw.ImageDraw, grid: list[list[str]], *, x: int, y: int, tile: int, font: Any) -> None:
    for r, row in enumerate(grid):
        for c, sym in enumerate(row):
            x0 = x + c * tile
            y0 = y + r * tile
            fill = _tile_color(sym)
            draw.rectangle((x0, y0, x0 + tile - 1, y0 + tile - 1), fill=fill, outline=(36, 42, 45))
            text_fill = (255, 255, 255) if sum(fill) < 410 else (30, 34, 38)
            bbox = draw.textbbox((0, 0), sym, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x0 + (tile - tw) / 2, y0 + (tile - th) / 2 - 1), sym, fill=text_fill, font=font)

def _wrap_text(text: str, width: int) -> list[str]:
    if not text:
        return []
    return textwrap.wrap(text, width=width, max_lines=2, placeholder='...')

def _make_frame(*, trajectory_id: int, step_id: int, rows: list[dict[str, Any]], task_name: str, tile: int, show_legend: bool) -> Image.Image:
    rows = sorted(rows, key=lambda r: int(r.get('agent_id', 0)))
    title_font = _font(20)
    body_font = _font(15)
    small_font = _font(13)
    grid_font = _font(max(12, int(tile * 0.42)))
    parsed = []
    for row in rows:
        prompt = str(row.get('prompt', ''))
        parsed.append({'row': row, 'goal': _parse_goal(prompt), 'grid': _parse_grid(prompt)})
    grids = [p['grid'] for p in parsed if p['grid']]
    grid_rows = max((len(g) for g in grids), default=9)
    grid_cols = max((len(g[0]) for g in grids if g), default=11)
    panel_w = max(360, grid_cols * tile + 34)
    panel_h = 176 + grid_rows * tile
    legend_h = 38 if show_legend else 0
    margin = 18
    gutter = 14
    width = margin * 2 + len(parsed) * panel_w + max(0, len(parsed) - 1) * gutter
    height = margin * 2 + panel_h + legend_h
    img = Image.new('RGB', (width, height), (244, 246, 241))
    draw = ImageDraw.Draw(img)
    draw.text((margin, 10), f'trajectory={trajectory_id}  step={step_id}  task={task_name}', fill=(31, 36, 42), font=title_font)
    y0 = 48
    for i, item in enumerate(parsed):
        row = item['row']
        x0 = margin + i * (panel_w + gutter)
        draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=6, fill=(255, 255, 255), outline=(198, 205, 198))
        role = str(row.get('agent_role', 'agent'))
        agent_id = row.get('agent_id', '?')
        action = str(row.get('expert_action_label', '?'))
        draw.text((x0 + 14, y0 + 12), f'agent_{agent_id} / {role}', fill=(35, 40, 46), font=body_font)
        draw.text((x0 + 14, y0 + 36), f'expert_action: {action}', fill=(140, 50, 42), font=body_font)
        goal_lines = _wrap_text(item['goal'], width=40)
        gy = y0 + 62
        for line in goal_lines:
            draw.text((x0 + 14, gy), line, fill=(72, 78, 86), font=small_font)
            gy += 17
        grid = item['grid']
        if grid:
            _draw_grid(draw, grid, x=x0 + 14, y=y0 + 108, tile=tile, font=grid_font)
        else:
            draw.text((x0 + 14, y0 + 112), 'No symbolic grid found in prompt.', fill=(140, 50, 42), font=body_font)
    if show_legend:
        legend = '. grass   : sand   ~ water   T tree   @ current agent'
        draw.text((margin, height - legend_h + 10), legend, fill=(70, 76, 82), font=small_font)
    return img

def _load_task_names(path: Path | None) -> list[str]:
    if path is None:
        return []
    path = path.expanduser().resolve()
    if not path.is_file():
        return []
    with path.open('r', encoding='utf-8') as f:
        meta = json.load(f)
    names = meta.get('task_names')
    return [str(x) for x in names] if isinstance(names, list) else []

def _write_gif(frames: list[Image.Image], path: Path, *, fps: float) -> None:
    if not frames:
        raise ValueError('No frames to save.')
    duration = int(1000 / max(float(fps), 0.1))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=False)

def main() -> None:
    p = argparse.ArgumentParser(description='Visualize local observations from JSONL trajectory shard as GIFs.')
    p.add_argument('--jsonl', type=Path, required=True, help='Input *.jsonl or *.jsonl.gz shard.')
    p.add_argument('--output_dir', type=Path, default=Path('runs/jsonl-trajectory-gifs'))
    p.add_argument('--trajectory_ids', type=str, default='', help='Comma-separated trajectory ids. Empty = first N.')
    p.add_argument('--num_trajectories', type=int, default=3, help='Used only when --trajectory_ids is empty.')
    p.add_argument('--max_steps', type=int, default=80, help='Max frames per trajectory. 0 = all steps.')
    p.add_argument('--max_rows', type=int, default=0, help='Max input rows to scan. 0 = all rows.')
    p.add_argument('--fps', type=float, default=3.0)
    p.add_argument('--tile_size', type=int, default=34)
    p.add_argument('--meta_json', type=Path, default=None, help='Optional expert_mappo.meta.json for task names.')
    p.add_argument('--no_legend', action='store_true')
    args = p.parse_args()
    if Image is None or ImageDraw is None or ImageFont is None:
        raise SystemExit('This script needs Pillow to render GIFs. Install it in the active env: pip install pillow')
    jsonl_path = args.jsonl.expanduser().resolve()
    if not jsonl_path.is_file():
        raise SystemExit(f'Input file not found: {jsonl_path}')
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_ids = _parse_ids(args.trajectory_ids)
    trajectories = _collect_trajectories(jsonl_path, trajectory_ids=trajectory_ids, num_trajectories=max(int(args.num_trajectories), 1), max_rows=int(args.max_rows))
    if not trajectories:
        raise SystemExit('No trajectories collected. Check --trajectory_ids/--max_rows.')
    task_names = _load_task_names(args.meta_json)
    summaries = []
    for tid in sorted(trajectories):
        step_map = trajectories[tid]
        steps = sorted(step_map)
        if int(args.max_steps) > 0:
            steps = steps[:int(args.max_steps)]
        frames: list[Image.Image] = []
        first_rows = step_map[steps[0]] if steps else []
        task_idx = int(first_rows[0].get('task_idx', -1)) if first_rows else -1
        task_name = task_names[task_idx] if 0 <= task_idx < len(task_names) else f'task_{task_idx}'
        for step in steps:
            frames.append(_make_frame(trajectory_id=tid, step_id=step, rows=step_map[step], task_name=task_name, tile=max(int(args.tile_size), 12), show_legend=not bool(args.no_legend)))
        gif_path = out_dir / f'trajectory_{_safe_name(tid)}_{_safe_name(task_name)}.gif'
        _write_gif(frames, gif_path, fps=float(args.fps))
        summary = {'trajectory_id': int(tid), 'task_idx': int(task_idx), 'task_name': task_name, 'num_frames': len(frames), 'first_step': int(steps[0]) if steps else None, 'last_step': int(steps[-1]) if steps else None, 'gif_path': str(gif_path)}
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    summary_path = out_dir / 'summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f'Saved summary: {summary_path}')
if __name__ == '__main__':
    main()
