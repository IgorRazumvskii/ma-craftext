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

    latest = ckpt_dir / "mappo_model_latest.pkl"
    if latest.is_file():
        return latest
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
    compact_obs: bool,
    instruction_dim: int,
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

        rng, step_rng = jax.random.split(rng)
        rng_step = jax.random.split(step_rng, num_envs)
        next_obs, next_env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(rng_step, env_state, env_act)

        done_batch = batchify_obs(done, agents, num_actors).squeeze()
        done_all = done["__all__"]
        next_alive = jnp.logical_and(alive, jnp.logical_not(done_all))

        obs_ead = jnp.swapaxes(jnp.stack([last_obs[a] for a in agents]), 0, 1)
        if compact_obs:
            obs_ead = obs_ead[..., :-instruction_dim]
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

    def rollout(carry):
        carry, traj = jax.lax.scan(_env_step, carry, None, max_steps)
        return carry, traj

    return jax.jit(rollout)


def save_rollout_part(traj, parts_dir, part_idx, trajectory_id_base, take, step_base):
    """Save only valid steps, in trajectory-major order, without per-step objects."""
    valid = traj["valid"][:, :take].T
    env_idx, time_idx = np.nonzero(valid)
    if not len(env_idx):
        return None, np.zeros(take, dtype=np.int64)
    counts = valid.sum(axis=1, dtype=np.int64)
    data = {
        key: np.asarray(value)[time_idx, env_idx]
        for key, value in traj.items() if key != "valid"
    }
    data["trajectory_id"] = (trajectory_id_base + env_idx).astype(np.int32)
    data["step_id"] = (step_base[env_idx] + time_idx).astype(np.int32)
    for key in ("instruction_idx", "task_idx"):
        data[key] = data[key].astype(np.int32)
    data["actions"] = data["actions"].astype(np.int32)
    data["instruction_done"] = data["instruction_done"].astype(np.float32)
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_path = parts_dir / f"part_{part_idx:06d}.npz"
    np.savez_compressed(part_path, **data)
    print(f"Saved part {part_idx:06d}: {part_path} (rows={len(env_idx)})", flush=True)
    return {"path": str(part_path), "num_rows": int(len(env_idx))}, counts


def prepare_stream_upload(repo_id, private, parts_dir, output_base):
    from hf_hub_utils import resolve_hf_repo_id, resolve_hf_token
    from huggingface_hub import HfApi

    repo_id = resolve_hf_repo_id(repo_id)
    token = resolve_hf_token()
    if not repo_id or not token:
        raise ValueError("Streaming upload requires --hf-repo-id and HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    existing = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    reserved = {output_base.with_suffix(".meta.json").name,
                output_base.with_suffix(".embeddings.npz").name}
    if any(path.startswith(parts_dir.name + "/") or path in reserved for path in existing):
        raise FileExistsError("Dataset name already exists on Hugging Face; choose a new --output")
    return api, repo_id


def stream_upload_file(api, repo_id, path, remote_path):
    # Keep the local file on any upload error so it can be recovered manually.
    path = Path(path)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=remote_path,
                    repo_id=repo_id, repo_type="dataset")
    path.unlink()
    print(f"[hf_hub] Uploaded and removed local file: {remote_path}", flush=True)


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
    parser.add_argument("--rollout-steps", type=int, default=128,
                        help="Maximum JAX rollout block length (limits device memory)")
    parser.add_argument("--compact-observations", action="store_true",
                        help="Store instruction embeddings once in a separate NPZ table")
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
        "--hf-stream-upload", action="store_true",
        help="Upload each part immediately and delete it locally after successful upload; requires --hf-upload",
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
    if args.hf_stream_upload and not args.hf_upload:
        parser.error("--hf-stream-upload requires --hf-upload")

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
    print("Note: parts are now written every rollout block; --save-every-trajectories is ignored.")

    if args.num_trajectories <= 0 or args.max_steps <= 0 or args.rollout_steps <= 0:
        raise ValueError("--num-trajectories, --max-steps and --rollout-steps must be > 0")
    rollout_steps = min(args.rollout_steps, args.max_steps)
    rollout_fn = build_rollout_fn(
        env=env,
        actor_network=actor_network,
        actor_params=actor_params,
        num_envs=num_envs_collect,
        max_steps=rollout_steps,
        greedy=bool(args.greedy),
        compact_obs=bool(args.compact_observations),
        instruction_dim=int(env.instruction_dim),
    )

    rng = jax.random.PRNGKey(args.seed)
    trajectories_done = 0
    total_steps_collected = 0
    total_len_sum = 0
    min_len = None
    max_len = 0
    part_idx = 0

    output_base = Path(output_path)
    parts_dir = output_base.with_suffix("").parent / f"{output_base.with_suffix('').name}_parts"
    parts_manifest = []
    if parts_dir.exists() or output_base.with_suffix(".meta.json").exists() or output_base.with_suffix(".embeddings.npz").exists():
        raise FileExistsError(f"Dataset output already exists for {output_base}; choose a new --output")

    num_agents = env.num_agents
    num_actors_collect = num_agents * num_envs_collect

    stream_api = None
    stream_repo = None
    if args.hf_stream_upload:
        stream_api, stream_repo = prepare_stream_upload(
            args.hf_repo_id, not args.hf_public, parts_dir, output_base
        )

    # Store the exact embeddings used in this process; no model re-encoding is needed.
    embedding_path = None
    if args.compact_observations:
        embeddings = np.asarray(env.scenario_handler.scenario_data_jax.embeddings_list,
                                dtype=np.float32).reshape((-1, env.instruction_dim))
        embedding_path = output_base.with_suffix(".embeddings.npz")
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(embedding_path, embeddings=embeddings)
        if stream_api is not None:
            stream_upload_file(stream_api, stream_repo, embedding_path, embedding_path.name)

    # Files contain rollout blocks. An episode can span multiple parts.
    while trajectories_done < args.num_trajectories:
        take_in_batch = min(num_envs_collect, args.num_trajectories - trajectories_done)
        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, num_envs_collect)
        obs_batch, env_state_batch = jax.vmap(
            lambda k: env.reset(k, instruction_idx=fixed_instruction_idx)
        )(reset_keys)
        hstate = ScannedRNN.initialize_carry(
            num_actors_collect, actor_network.config["GRU_HIDDEN_DIM"]
        )
        last_done = jnp.zeros((num_actors_collect,), dtype=bool)
        rng, rollout_rng = jax.random.split(rng)
        carry = (env_state_batch, obs_batch, last_done, hstate, rollout_rng,
                 jnp.arange(num_envs_collect) < take_in_batch)
        lengths = np.zeros(take_in_batch, dtype=np.int64)
        steps_run = 0
        while steps_run < args.max_steps:
            # The final block may be padded; rows beyond max_steps are discarded.
            carry, traj = rollout_fn(carry)
            traj_np = jax.device_get(traj)
            block_len = min(rollout_steps, args.max_steps - steps_run)
            if block_len != rollout_steps:
                traj_np = {key: value[:block_len] for key, value in traj_np.items()}
            part_idx += 1
            part, counts = save_rollout_part(
                traj_np, parts_dir, part_idx, trajectories_done, take_in_batch, lengths
            )
            if part:
                parts_manifest.append(part)
                if stream_api is not None:
                    part_path = Path(part["path"])
                    stream_upload_file(stream_api, stream_repo, part_path,
                                       f"{parts_dir.name}/{part_path.name}")
                total_steps_collected += part["num_rows"]
            lengths += counts
            steps_run += block_len
            if not np.asarray(jax.device_get(carry[-1]))[:take_in_batch].any():
                break
        total_len_sum += int(lengths.sum())
        batch_min, batch_max = int(lengths.min()), int(lengths.max())
        min_len = batch_min if min_len is None else min(min_len, batch_min)
        max_len = max(max_len, batch_max)
        trajectories_done += take_in_batch
        print(f"[{trajectories_done}/{args.num_trajectories}] "
              f"steps_collected={total_steps_collected}, "
              f"mean_len={total_len_sum / trajectories_done:.1f}", flush=True)

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
        "output_mode": "rollout_block_npz_parts",
        "format_version": 2,
        "rollout_steps": rollout_steps,
        "compact_observations": bool(args.compact_observations),
        "instruction_dim": int(env.instruction_dim),
        "embeddings_path": str(embedding_path) if embedding_path else None,
        "parts_may_split_trajectories": True,
        "parts_dir": str(parts_dir),
        "parts": parts_manifest,
        "save_every_trajectories": None,
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

    if stream_api is not None:
        # Metadata is the completion marker: publish only after every part succeeded.
        stream_upload_file(stream_api, stream_repo, meta_path, meta_path.name)
        if parts_dir.exists():
            parts_dir.rmdir()
        print(f"[hf_hub] Dataset complete: {stream_repo}")
        return

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
            [meta_path] + ([embedding_path] if embedding_path else []),
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
