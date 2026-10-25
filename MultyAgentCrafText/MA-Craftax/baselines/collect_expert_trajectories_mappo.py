"""
Collect expert trajectories from the latest MAPPO checkpoint.

Dataset contains per-step transitions with:
- observations of all agents,
- actions of all agents,
- rewards of all agents,
- instruction_done, instruction_idx, task_idx,
- trajectory_id and step_id.

Usage examples:
    python baselines/collect_expert_trajectories_mappo.py \
      --checkpoint-dir runs/.../checkpoints \
      --num-trajectories 200 \
      --output datasets/expert_mappo.npz

    python baselines/collect_expert_trajectories_mappo.py \
      --checkpoint runs/.../checkpoints/mappo_model_update_1200.pkl \
      --num-trajectories 50 --greedy
"""

import argparse
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from my_mappo import ActorRNN, ScannedRNN, build_actor_action_mask
from craftax.craftax_together_talens.envs.patterns_wrapper import (
    make_craftax_env_with_my_wrapper,
)


def _tree_map():
    try:
        return jax.tree.map
    except AttributeError:
        return jax.tree_util.tree_map


def find_latest_checkpoint(checkpoint_dir: str) -> Path:
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    candidates = list(ckpt_dir.glob("mappo_model_update_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No MAPPO checkpoints found in {ckpt_dir} "
            "(expected files like mappo_model_update_100.pkl)"
        )

    def extract_step(p: Path) -> int:
        m = re.search(r"mappo_model_update_(\d+)\.pkl$", p.name)
        return int(m.group(1)) if m else -1

    candidates.sort(key=extract_step)
    return candidates[-1]


def load_checkpoint(path: str | Path) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    if not isinstance(checkpoint, dict) or "actor_train_state" not in checkpoint:
        raise ValueError(
            "Invalid MAPPO checkpoint format: expected key 'actor_train_state'"
        )
    return checkpoint


def restore_actor_params(checkpoint: dict):
    tree_map_fn = _tree_map()
    return tree_map_fn(lambda x: jnp.array(x), checkpoint["actor_train_state"]["params"])


def select_fixed_instruction_idx(env, instruction_idx: int, task_idx: int | None) -> int:
    if instruction_idx >= 0 and task_idx is not None:
        raise ValueError("Use only one of --instruction-idx or --task-idx")

    if task_idx is not None:
        instruction_to_task = np.asarray(env.instruction_to_task_idx).astype(int).reshape(-1)
        matches = np.where(instruction_to_task == int(task_idx))[0]
        if matches.size == 0:
            available = sorted(set(instruction_to_task.tolist()))
            raise ValueError(f"Task idx {task_idx} not found. Available: {available}")
        return int(matches[0])

    if instruction_idx >= 0:
        n_instr = int(len(env.scenario_handler.scenario_data.instructions_list))
        if instruction_idx >= n_instr:
            raise ValueError(
                f"--instruction-idx={instruction_idx} out of range [0, {n_instr - 1}]"
            )
        return int(instruction_idx)

    return -1


def batchify_obs(obs: dict, agent_list, num_actors):
    x = jnp.stack([obs[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify_actions(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((num_agents, num_envs))
    return {a: x[i] for i, a in enumerate(agent_list)}


def build_rollout_fn(
    env,
    actor_network,
    actor_params,
    num_envs: int,
    max_steps: int,
    greedy: bool,
):
    agents = env.agents
    num_agents = env.num_agents
    num_actors = num_agents * num_envs
    actor_action_mask = build_actor_action_mask(env, num_envs)

    def _env_step(carry, _):
        env_state, last_obs, last_done, hstate, rng, alive = carry

        obs_actor = batchify_obs(last_obs, agents, num_actors)
        ac_in = (
            obs_actor[jnp.newaxis, :],
            last_done[jnp.newaxis, :],
            actor_action_mask[jnp.newaxis, :],
        )

        rng, action_rng = jax.random.split(rng)
        hstate, pi = actor_network.apply(actor_params, hstate, ac_in)
        if greedy:
            action = jnp.argmax(pi.probs, axis=-1)
        else:
            action = pi.sample(seed=action_rng)
        action = action.squeeze()

        env_act = unbatchify_actions(action, agents, num_envs, num_agents)
        env_act = {k: v.squeeze() for k, v in env_act.items()}

        rng, step_rng = jax.random.split(rng)
        rng_step = jax.random.split(step_rng, num_envs)
        next_obs, next_env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(rng_step, env_state, env_act)

        done_batch = batchify_obs(done, agents, num_actors).squeeze()
        done_all = done["__all__"]
        next_alive = jnp.logical_and(alive, jnp.logical_not(done_all))

        obs_ead = jnp.swapaxes(jnp.stack([last_obs[a] for a in agents]), 0, 1)
        actions_ea = jnp.swapaxes(jnp.stack([env_act[a] for a in agents]), 0, 1)
        rewards_ea = jnp.swapaxes(jnp.stack([reward[a] for a in agents]), 0, 1)
        dones_ea = jnp.swapaxes(jnp.stack([done[a] for a in agents]), 0, 1)
        user_info = info["user_info"]
        # These fields are duplicated per-agent; take channel 0 for per-episode scalar labels.
        instruction_done = user_info["Instructions/instruction_done"][:, 0]
        instruction_idx = user_info["Instructions/instruction_idx"][:, 0]
        task_idx = user_info["Instructions/task_idx"][:, 0]

        transition = {
            "obs": obs_ead,
            "actions": actions_ea,
            "rewards": rewards_ea,
            "dones": dones_ea,
            "done_all": done_all,
            "valid": alive,
            "instruction_done": instruction_done,
            "instruction_idx": instruction_idx,
            "task_idx": task_idx,
        }
        new_carry = (next_env_state, next_obs, done_batch, hstate, rng, next_alive)
        return new_carry, transition

    def rollout(env_state, last_obs, last_done, hstate, rng):
        alive = jnp.ones((num_envs,), dtype=bool)
        carry = (env_state, last_obs, last_done, hstate, rng, alive)
        carry, traj = jax.lax.scan(_env_step, carry, None, max_steps)
        return carry, traj

    return jax.jit(rollout)


def collect_rows_from_batched_rollout(
    traj_np: dict,
    trajectory_id_base: int,
    num_trajectories_to_take: int,
):
    rows = []
    lengths = []
    valid = traj_np["valid"]  # [T, E]
    t_max = valid.shape[0]

    for env_idx in range(num_trajectories_to_take):
        traj_id = trajectory_id_base + env_idx
        valid_steps = np.nonzero(valid[:, env_idx])[0]
        if valid_steps.size == 0:
            lengths.append(0)
            continue

        local_step = 0
        for t in valid_steps:
            if t >= t_max:
                break
            rows.append(
                {
                    "trajectory_id": int(traj_id),
                    "step_id": int(local_step),
                    "obs": traj_np["obs"][t, env_idx].astype(np.float32),
                    "actions": traj_np["actions"][t, env_idx].astype(np.int32),
                    "rewards": traj_np["rewards"][t, env_idx].astype(np.float32),
                    "dones": traj_np["dones"][t, env_idx].astype(np.bool_),
                    "done_all": bool(traj_np["done_all"][t, env_idx]),
                    "instruction_done": float(traj_np["instruction_done"][t, env_idx]),
                    "instruction_idx": int(traj_np["instruction_idx"][t, env_idx]),
                    "task_idx": int(traj_np["task_idx"][t, env_idx]),
                }
            )
            local_step += 1
            if rows[-1]["done_all"]:
                break
        lengths.append(local_step)
    return rows, lengths


def run_single_trajectory(
    env,
    actor_network,
    actor_params,
    rng,
    trajectory_id: int,
    max_steps: int,
    instruction_idx: int,
    greedy: bool,
):
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, instruction_idx=instruction_idx)

    agents = env.agents
    num_agents = len(agents)
    hstate = ScannedRNN.initialize_carry(num_agents, actor_network.config["GRU_HIDDEN_DIM"])
    last_done = jnp.zeros((num_agents,), dtype=bool)
    actor_action_mask = build_actor_action_mask(env, 1)

    rows = []
    step_id = 0
    done_all = False

    while (not done_all) and (step_id < max_steps):
        obs_arr = np.stack([np.asarray(obs[a]) for a in agents], axis=0)
        ac_in = (
            jnp.asarray(obs_arr)[None, :],
            last_done[None, :],
            actor_action_mask[None, :],
        )

        rng, action_rng = jax.random.split(rng)
        hstate, pi = actor_network.apply(actor_params, hstate, ac_in)
        if greedy:
            action = jnp.argmax(pi.probs, axis=-1).squeeze()
        else:
            action = pi.sample(seed=action_rng).squeeze()

        if action.ndim > 1:
            action = action.flatten()
        action_arr = np.asarray(action, dtype=np.int32).reshape(num_agents)
        action_dict = {agent: jnp.int32(action_arr[i]) for i, agent in enumerate(agents)}

        rng, step_rng = jax.random.split(rng)
        next_obs, env_state, reward, done, _ = env.step(step_rng, env_state, action_dict)

        reward_arr = np.asarray([reward[a] for a in agents], dtype=np.float32)
        done_arr = np.asarray([done[a] for a in agents], dtype=np.bool_)

        rows.append(
            {
                "trajectory_id": int(trajectory_id),
                "step_id": int(step_id),
                "obs": obs_arr.astype(np.float32),
                "actions": action_arr.astype(np.int32),
                "rewards": reward_arr,
                "dones": done_arr,
                "done_all": bool(done["__all__"]),
            }
        )

        obs = next_obs
        last_done = jnp.asarray(done_arr)
        done_all = bool(done["__all__"])
        step_id += 1

    return rows, rng


def save_dataset(rows: list[dict], output_path: str, meta: dict):
    if not rows:
        raise RuntimeError("Attempted to save empty rows.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    trajectory_id = np.asarray([r["trajectory_id"] for r in rows], dtype=np.int32)
    step_id = np.asarray([r["step_id"] for r in rows], dtype=np.int32)
    obs = np.stack([r["obs"] for r in rows], axis=0).astype(np.float32)
    actions = np.stack([r["actions"] for r in rows], axis=0).astype(np.int32)
    rewards = np.stack([r["rewards"] for r in rows], axis=0).astype(np.float32)
    dones = np.stack([r["dones"] for r in rows], axis=0).astype(np.bool_)
    done_all = np.asarray([r["done_all"] for r in rows], dtype=np.bool_)
    instruction_done = np.asarray([r["instruction_done"] for r in rows], dtype=np.float32)
    instruction_idx = np.asarray([r["instruction_idx"] for r in rows], dtype=np.int32)
    task_idx = np.asarray([r["task_idx"] for r in rows], dtype=np.int32)

    np.savez_compressed(
        output,
        trajectory_id=trajectory_id,
        step_id=step_id,
        obs=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        done_all=done_all,
        instruction_done=instruction_done,
        instruction_idx=instruction_idx,
        task_idx=task_idx,
    )

    meta_path = output.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved dataset: {output}")
    print(f"Saved metadata: {meta_path}")
    print(
        f"Shapes -> obs: {obs.shape}, actions: {actions.shape}, "
        f"rewards: {rewards.shape}, steps: {trajectory_id.shape[0]}, "
        f"instruction_idx: {instruction_idx.shape}"
    )


def save_dataset_part(rows: list[dict], parts_dir: Path, part_idx: int) -> dict:
    if not rows:
        return {"path": None, "num_rows": 0}

    parts_dir.mkdir(parents=True, exist_ok=True)
    part_path = parts_dir / f"part_{part_idx:06d}.npz"
    trajectory_id = np.asarray([r["trajectory_id"] for r in rows], dtype=np.int32)
    step_id = np.asarray([r["step_id"] for r in rows], dtype=np.int32)
    obs = np.stack([r["obs"] for r in rows], axis=0).astype(np.float32)
    actions = np.stack([r["actions"] for r in rows], axis=0).astype(np.int32)
    rewards = np.stack([r["rewards"] for r in rows], axis=0).astype(np.float32)
    dones = np.stack([r["dones"] for r in rows], axis=0).astype(np.bool_)
    done_all = np.asarray([r["done_all"] for r in rows], dtype=np.bool_)
    instruction_done = np.asarray([r["instruction_done"] for r in rows], dtype=np.float32)
    instruction_idx = np.asarray([r["instruction_idx"] for r in rows], dtype=np.int32)
    task_idx = np.asarray([r["task_idx"] for r in rows], dtype=np.int32)

    np.savez_compressed(
        part_path,
        trajectory_id=trajectory_id,
        step_id=step_id,
        obs=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        done_all=done_all,
        instruction_done=instruction_done,
        instruction_idx=instruction_idx,
        task_idx=task_idx,
    )
    print(
        f"Saved part {part_idx:06d}: {part_path} "
        f"(rows={trajectory_id.shape[0]}, traj_id=[{trajectory_id.min()}..{trajectory_id.max()}])"
    )
    return {"path": str(part_path), "num_rows": int(trajectory_id.shape[0])}


def main():
    parser = argparse.ArgumentParser(
        description="Collect expert trajectories from MAPPO checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to MAPPO checkpoint .pkl",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory with mappo_model_update_*.pkl; latest will be used",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz file path (default: datasets/expert_mappo_<timestamp>.npz)",
    )
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--num-envs-collect",
        type=int,
        default=32,
        help="Parallel envs for inference rollout (higher is faster, but uses more memory)",
    )
    parser.add_argument(
        "--save-every-trajectories",
        type=int,
        default=2000,
        help="Flush rows to disk every N collected trajectories",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use argmax actions instead of sampling",
    )
    parser.add_argument(
        "--instruction-idx",
        type=int,
        default=-1,
        help="Fix exact instruction variant index",
    )
    parser.add_argument(
        "--task-idx",
        type=int,
        default=None,
        help="Fix unique task index; uses first matching instruction variant",
    )
    parser.add_argument(
        "--hf-upload",
        action="store_true",
        help="Upload collected parts + meta.json to Hugging Face Hub after collection",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Hub dataset repo id (or set HF_HUB_REPO_ID). Requires HF_TOKEN.",
    )
    parser.add_argument(
        "--hf-public",
        action="store_true",
        help="Create/use a public Hub dataset repo (default: private)",
    )
    args = parser.parse_args()

    if args.checkpoint is None and args.checkpoint_dir is None:
        raise ValueError("Provide either --checkpoint or --checkpoint-dir")
    if args.checkpoint is not None and args.checkpoint_dir is not None:
        raise ValueError("Use only one of --checkpoint or --checkpoint-dir")

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        print(f"Using latest checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.get("config", {})
    env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    craftext_config_name = config.get("CRAFTEXT_CONFIG_NAME", "easy_train")

    env = make_craftax_env_with_my_wrapper(
        env_name,
        craftext_config_name=craftext_config_name,
    )

    fixed_instruction_idx = select_fixed_instruction_idx(
        env, args.instruction_idx, args.task_idx
    )
    if fixed_instruction_idx >= 0:
        task_idx = int(np.asarray(env.instruction_to_task_idx)[fixed_instruction_idx])
        task_name = (
            env.task_names[task_idx]
            if 0 <= task_idx < len(env.task_names)
            else f"task_{task_idx}"
        )
        print(
            f"Fixed instruction_idx={fixed_instruction_idx}, "
            f"task_idx={task_idx}, task_name={task_name}"
        )

    task_names = [str(x) for x in getattr(env, "task_names", [])]
    try:
        instruction_text_by_idx = [
            str(x) for x in env.scenario_handler.scenario_data.instructions_list
        ]
    except Exception:
        instruction_text_by_idx = []

    actor_params = restore_actor_params(checkpoint)
    actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"datasets/expert_mappo_{ts}.npz"
    else:
        output_path = args.output

    num_envs_collect = int(args.num_envs_collect)
    if num_envs_collect <= 0:
        raise ValueError("--num-envs-collect must be > 0")
    save_every_trajectories = int(args.save_every_trajectories)
    if save_every_trajectories <= 0:
        raise ValueError("--save-every-trajectories must be > 0")

    rollout_fn = build_rollout_fn(
        env=env,
        actor_network=actor_network,
        actor_params=actor_params,
        num_envs=num_envs_collect,
        max_steps=int(args.max_steps),
        greedy=bool(args.greedy),
    )

    rng = jax.random.PRNGKey(args.seed)
    chunk_rows = []
    trajectories_done = 0
    trajectories_since_last_flush = 0
    total_steps_collected = 0
    total_len_sum = 0
    min_len = None
    max_len = 0
    part_idx = 0

    output_base = Path(output_path)
    parts_dir = output_base.with_suffix("").parent / f"{output_base.with_suffix('').name}_parts"
    parts_manifest = []

    num_agents = env.num_agents
    num_actors_collect = num_agents * num_envs_collect

    while trajectories_done < args.num_trajectories:
        remaining = args.num_trajectories - trajectories_done
        take_in_batch = min(num_envs_collect, remaining)

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, num_envs_collect)
        reset_fn = lambda k: env.reset(k, instruction_idx=fixed_instruction_idx)
        obs_batch, env_state_batch = jax.vmap(reset_fn)(reset_keys)

        hstate = ScannedRNN.initialize_carry(
            num_actors_collect, actor_network.config["GRU_HIDDEN_DIM"]
        )
        last_done = jnp.zeros((num_actors_collect,), dtype=bool)

        rng, rollout_rng = jax.random.split(rng)
        _, traj = rollout_fn(env_state_batch, obs_batch, last_done, hstate, rollout_rng)
        traj_np = jax.device_get(traj)

        batch_rows, batch_lens = collect_rows_from_batched_rollout(
            traj_np=traj_np,
            trajectory_id_base=trajectories_done,
            num_trajectories_to_take=take_in_batch,
        )
        chunk_rows.extend(batch_rows)
        total_steps_collected += len(batch_rows)
        total_len_sum += int(sum(batch_lens))
        if batch_lens:
            batch_min = int(min(batch_lens))
            batch_max = int(max(batch_lens))
            min_len = batch_min if min_len is None else min(min_len, batch_min)
            max_len = max(max_len, batch_max)
        trajectories_done += take_in_batch
        trajectories_since_last_flush += take_in_batch

        # Flush chunk to disk to avoid RAM growth.
        if (
            trajectories_since_last_flush >= save_every_trajectories
            or trajectories_done == args.num_trajectories
        ):
            part_idx += 1
            part_info = save_dataset_part(chunk_rows, parts_dir=parts_dir, part_idx=part_idx)
            if part_info["path"] is not None:
                parts_manifest.append(part_info)
            chunk_rows = []
            trajectories_since_last_flush = 0

        if trajectories_done % 10 == 0 or trajectories_done == args.num_trajectories:
            mean_len = float(total_len_sum / max(trajectories_done, 1))
            print(
                f"[{trajectories_done}/{args.num_trajectories}] "
                f"steps_collected={total_steps_collected}, mean_len={mean_len:.1f}"
            )

    if total_steps_collected == 0:
        raise RuntimeError("No transitions were collected (empty dataset).")

    avg_len = float(total_len_sum / max(args.num_trajectories, 1))
    meta = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_update_steps": checkpoint.get("update_steps", None),
        "env_name": env_name,
        "craftext_config_name": craftext_config_name,
        "num_trajectories": int(args.num_trajectories),
        "num_envs_collect": int(num_envs_collect),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "greedy": bool(args.greedy),
        "instruction_idx": int(args.instruction_idx),
        "task_idx": None if args.task_idx is None else int(args.task_idx),
        "num_agents": int(env.num_agents),
        "agents": list(env.agents),
        "task_names": task_names,
        "instruction_text_by_idx": instruction_text_by_idx,
        "num_transitions": int(total_steps_collected),
        "avg_trajectory_len": avg_len,
        "min_trajectory_len": int(min_len if min_len is not None else 0),
        "max_trajectory_len": int(max_len),
        "output_mode": "chunked_npz_parts",
        "parts_dir": str(parts_dir),
        "parts": parts_manifest,
        "save_every_trajectories": int(save_every_trajectories),
    }

    meta_path = output_base.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {meta_path}")
    print(
        f"Done. Parts saved: {len(parts_manifest)}, total transitions: {total_steps_collected}, "
        f"parts dir: {parts_dir}"
    )

    # Optional Hugging Face Hub upload of parts + metadata.
    hf_private = not bool(args.hf_public)
    try:
        from hf_hub_utils import upload_files_to_hub, upload_folder_to_hub

        upload_folder_to_hub(
            parts_dir,
            repo_id=args.hf_repo_id,
            repo_type="dataset",
            path_in_repo=parts_dir.name,
            private=hf_private,
            enabled=bool(args.hf_upload),
        )
        upload_files_to_hub(
            [meta_path],
            repo_id=args.hf_repo_id,
            repo_type="dataset",
            path_in_repo_prefix="",
            private=hf_private,
            enabled=bool(args.hf_upload),
        )
    except Exception as hf_err:
        print(f"[hf_hub] Dataset upload failed (local files kept): {hf_err}")


if __name__ == "__main__":
    main()
