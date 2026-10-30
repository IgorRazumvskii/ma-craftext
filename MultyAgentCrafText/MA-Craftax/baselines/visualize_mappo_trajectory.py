#!/usr/bin/env python3
"""Render saved Craftax-Coop symbolic MAPPO observations to an animated GIF.

The frames show observations *before* each recorded action. The dataset does
not contain complete world states or the observation after the final action.
"""
from __future__ import annotations

import argparse
import ast
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CONSTANTS = ROOT / "craftax/craftax_together_talens/constants.py"


def read_layout(path=CONSTANTS):
    """Read literal enum values without importing JAX or loading game textures."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = {}
    dimensions = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OBS_DIM" for t in node.targets):
            dimensions = ast.literal_eval(node.value)
        if isinstance(node, ast.ClassDef) and node.name in {"BlockType", "ItemType", "Action"}:
            members = {}
            for assignment in node.body:
                if isinstance(assignment, ast.Assign) and len(assignment.targets) == 1 and isinstance(assignment.targets[0], ast.Name):
                    try:
                        value = ast.literal_eval(assignment.value)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(value, int):
                        members[value] = assignment.targets[0].id
            classes[node.name] = members
    if dimensions != (9, 11) or set(classes) != {"BlockType", "ItemType", "Action"}:
        raise ValueError("Unsupported Craftax symbolic observation layout")
    for name in ("BlockType", "ItemType"):
        if set(classes[name]) != set(range(len(classes[name]))):
            raise ValueError(f"Non-contiguous {name}; cannot decode observation safely")
    return dimensions, classes


def load_meta(path):
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("env_name") != "Craftax-Coop-Symbolic":
        raise ValueError("Only Craftax-Coop-Symbolic datasets are supported")
    if not meta.get("parts"):
        raise ValueError("No parts listed in metadata")
    return meta


def part_paths(meta_path, meta):
    parts_dir = meta_path.parent / Path(meta["parts_dir"]).name
    for entry in meta["parts"]:
        path = parts_dir / Path(entry["path"]).name
        if not path.is_file():
            raise FileNotFoundError(f"Dataset part not found: {path}")
        yield path


def list_trajectories(meta_path, meta):
    result = {}
    for path in part_paths(meta_path, meta):
        with np.load(path, allow_pickle=False) as data:
            for tid in np.unique(data["trajectory_id"]):
                tid = int(tid)
                mask = data["trajectory_id"] == tid
                entry = result.setdefault(tid, {"steps": 0, "task_idx": int(data["task_idx"][mask][0]), "success": False})
                entry["steps"] += int(mask.sum())
                entry["success"] |= bool(np.any(data["instruction_done"][mask] > 0))
    return result


def read_frames(meta_path, meta, tid, every, max_frames):
    frames = []
    seen = set()
    for path in part_paths(meta_path, meta):
        with np.load(path, allow_pickle=False) as data:
            rows = np.flatnonzero(data["trajectory_id"] == tid)
            for i in rows:
                step = int(data["step_id"][i])
                if step in seen:
                    raise ValueError(f"Duplicate step {step} for trajectory {tid}")
                seen.add(step)
                if step % every:
                    continue
                if max_frames and len(frames) >= max_frames:
                    largest = max(range(len(frames)), key=lambda j: int(frames[j]["step_id"]))
                    if step >= int(frames[largest]["step_id"]):
                        continue
                    frames.pop(largest)
                frames.append({key: np.array(data[key][i], copy=True) for key in
                               ("obs", "actions", "rewards", "done_all", "instruction_done", "instruction_idx", "task_idx", "step_id")})
    if not seen:
        raise ValueError(f"Trajectory {tid} not found")
    frames.sort(key=lambda row: int(row["step_id"]))
    if max_frames:
        frames = frames[:max_frames]
    return frames, len(seen)


COLORS = {
    "GRASS": "#79aa68", "WATER": "#4389b5", "STONE": "#92979b",
    "TREE": "#347c46", "WOOD": "#a36b3e", "PATH": "#c7ae7c",
    "COAL": "#61666e", "IRON": "#bb927d", "DIAMOND": "#6dc8c8",
    "CRAFTING_TABLE": "#a57540", "FURNACE": "#737982", "SAND": "#dec994",
    "LAVA": "#e87935", "PLANT": "#91b553", "RIPE_PLANT": "#dad15b",
    "OUT_OF_BOUNDS": "#30343d", "DARKNESS": "#30343d", "WALL": "#717580",
}
SYMBOLS = {"GRASS": ".", "WATER": "~", "STONE": "S", "TREE": "T", "WOOD": "W",
           "PATH": ":", "COAL": "C", "IRON": "I", "DIAMOND": "D", "SAND": ":",
           "LAVA": "L", "CRAFTING_TABLE": "CT", "FURNACE": "F", "PLANT": "p", "RIPE_PLANT": "P"}


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def action_name(action, actions):
    value = int(action)
    if value in actions:
        return actions[value]
    give = next((v for v, name in actions.items() if name == "GIVE"), None)
    if give is not None and value > give:
        return f"GIVE (choice {value - give + 1})"
    return f"ACTION_{value}"


def draw_frame(row, tid, meta, layout, tile):
    (height, width), classes = layout
    obs = row["obs"]
    if obs.ndim != 2:
        raise ValueError(f"Expected obs shape [agents, features], got {obs.shape}")
    n_agents = obs.shape[0]
    n_blocks, n_items = len(classes["BlockType"]), len(classes["ItemType"])
    channels = n_blocks + n_items + 40 + n_agents + 1 + 1
    map_size = height * width * channels
    if obs.shape[1] < map_size:
        raise ValueError(f"Observation has {obs.shape[1]} values, expected at least {map_size}; incompatible layout")
    grid = obs[:, :map_size].reshape(n_agents, height, width, channels)
    light = grid[..., -1] > 0.5
    blocks = grid[..., :n_blocks]
    # Illuminated map cells are exactly one-hot in the symbolic renderer.
    if np.any(np.abs(blocks.sum(axis=-1)[light] - 1) > 1e-4):
        raise ValueError("Observation map is not one-hot; incompatible symbolic layout")
    block_ids = blocks.argmax(axis=-1)
    items = grid[..., n_blocks:n_blocks + n_items]
    mobs = grid[..., n_blocks + n_items:n_blocks + n_items + 40]
    teammates = grid[..., n_blocks + n_items + 40:n_blocks + n_items + 40 + n_agents]
    title, body, small, tile_font = font(19), font(15), font(12), font(max(11, tile // 3))
    panel_w = max(width * tile + 24, 260)
    top = 125
    canvas = Image.new("RGB", (24 + n_agents * (panel_w + 12), top + height * tile + 90), "#f4f6f1")
    draw = ImageDraw.Draw(canvas)
    step = int(row["step_id"])
    instr_idx = int(row["instruction_idx"])
    task_idx = int(row["task_idx"])
    instruction_list = meta.get("instruction_text_by_idx", [])
    instruction = str(instruction_list[instr_idx]) if 0 <= instr_idx < len(instruction_list) else f"Instruction {instr_idx}"
    task_names = meta.get("task_names", [])
    task = str(task_names[task_idx]) if 0 <= task_idx < len(task_names) else f"task_{task_idx}"
    status = "SUCCESS (after action)" if bool(row["instruction_done"]) else ("DONE (after action)" if bool(row["done_all"]) else "")
    draw.text((12, 8), f"Trajectory {tid} | step {step} | {task} | {status}", fill="#202830", font=title)
    for i, line in enumerate(textwrap.wrap(' '.join(instruction.split()), width=max(35, canvas.width // 9))[:3]):
        draw.text((12, 34 + i * 16), line, fill="#344052", font=small)
    for agent in range(n_agents):
        x0 = 12 + agent * (panel_w + 12)
        name = meta.get("agents", [])[agent] if agent < len(meta.get("agents", [])) else f"agent_{agent}"
        draw.text((x0, 90), f"{name}: {action_name(row['actions'][agent], classes['Action'])}", fill="#202830", font=body)
        draw.text((x0, 108), f"reward: {float(row['rewards'][agent]):.3f}", fill="#556050", font=small)
        for y in range(height):
            for x in range(width):
                xx, yy = x0 + x * tile, top + y * tile
                name_block = classes["BlockType"].get(int(block_ids[agent,y,x]), "INVALID") if light[agent,y,x] else "DARKNESS"
                color = COLORS.get(name_block, "#8c9285")
                draw.rectangle((xx, yy, xx + tile - 1, yy + tile - 1), fill=color, outline="#555b5b")
                label = SYMBOLS.get(name_block, name_block[:2] if name_block not in ("DARKNESS", "OUT_OF_BOUNDS") else "")
                if light[agent,y,x]:
                    item = int(np.argmax(items[agent,y,x]))
                    if item and items[agent,y,x,item] > 0.5:
                        label = {"TORCH":"t", "LADDER_DOWN":"v", "LADDER_UP":"^", "LADDER_DOWN_BLOCKED":"x"}.get(classes['ItemType'].get(item), "i")
                    mob = np.flatnonzero(mobs[agent,y,x] > 0.5)
                    if mob.size:
                        label = "MPRqp"[int(mob[0]) // 8] + str(int(mob[0]) % 8)
                    if teammates[agent,y,x,1:].any():
                        label = "A"
                if (y, x) == (height // 2, width // 2):
                    label = "@"
                if label:
                    bbox = draw.textbbox((0,0), label, font=tile_font)
                    draw.text((xx + (tile - (bbox[2]-bbox[0]))/2, yy + (tile - (bbox[3]-bbox[1]))/2 - 2), label, fill="#ffffff" if name_block in {"DARKNESS", "OUT_OF_BOUNDS", "TREE", "WATER"} else "#1b2024", font=tile_font)
    foot = top + height * tile + 12
    draw.text((12, foot), "@ agent   A teammate   T tree   S stone   ~ water   M/P/R mob   t torch", fill="#344052", font=small)
    draw.text((12, foot + 18), "Observation BEFORE action. Mob labels: class + type ID. Dark cells are unknown.", fill="#344052", font=small)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, required=True, help="Dataset .meta.json")
    parser.add_argument("--trajectory-id", type=int, help="ID to visualize (default: smallest ID)")
    parser.add_argument("--list", action="store_true", help="List trajectories without rendering")
    parser.add_argument("--output", type=Path, help="Output GIF path")
    parser.add_argument("--fps", type=float, default=3)
    parser.add_argument("--every", type=int, default=1, help="Render every Nth step")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum frames; 0 means unlimited")
    parser.add_argument("--tile-size", type=int, default=28)
    args = parser.parse_args()
    if args.fps <= 0 or args.every <= 0 or args.max_frames < 0 or args.tile_size < 16:
        parser.error("fps and every must be positive; max-frames >= 0; tile-size >= 16")
    meta_path = args.meta.expanduser().resolve()
    meta = load_meta(meta_path)
    trajectories = list_trajectories(meta_path, meta)
    if not trajectories:
        raise ValueError("Dataset contains no trajectories")
    if args.list:
        for tid, info in sorted(trajectories.items()):
            print(f"trajectory={tid} steps={info['steps']} task_idx={info['task_idx']} success={info['success']}")
        return
    tid = args.trajectory_id if args.trajectory_id is not None else min(trajectories)
    frames, total = read_frames(meta_path, meta, tid, args.every, args.max_frames)
    layout = read_layout()
    images = [draw_frame(row, tid, meta, layout, args.tile_size) for row in frames]
    output = args.output or Path(f"trajectory_{tid}.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=max(10, round(1000 / args.fps / 10) * 10), loop=0, optimize=False)
    print(f"Saved {output}: {len(images)} frames, {total} recorded steps; frames show observations before actions")
    if args.max_frames and len(images) == args.max_frames and len(images) < (total + args.every - 1) // args.every:
        print("Frame limit reached; use --max-frames 0 or --every N to view more steps.")


if __name__ == "__main__":
    main()
