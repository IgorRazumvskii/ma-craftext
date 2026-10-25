"""
Evaluate honest instruction SR from a trained checkpoint.

This script runs inference episodes without visualization and reports:
- overall SR (solved / posed),
- optional per-task SR.
"""

import argparse
import csv
import datetime as dt
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from my_ippo import ActorCriticRNN, ScannedRNN
from craftax.craftax_together_talens.envs.patterns_wrapper import (
    make_craftax_env_with_my_wrapper,
)


def load_checkpoint(checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    if not isinstance(checkpoint, dict) or "train_state" not in checkpoint:
        raise ValueError("Invalid checkpoint format: expected key 'train_state'")
    return checkpoint


def restore_model_params(checkpoint):
    try:
        tree_map_fn = jax.tree.map
    except AttributeError:
        tree_map_fn = jax.tree_util.tree_map
    return tree_map_fn(lambda x: jnp.array(x), checkpoint["train_state"]["params"])


def select_fixed_instruction_idx(env, instruction_idx: int, task_idx: int | None):
    if instruction_idx >= 0 and task_idx is not None:
        raise ValueError("Use only one of --instruction-idx or --task-idx")

    if task_idx is not None:
        instruction_to_task = np.asarray(env.instruction_to_task_idx).astype(int).reshape(-1)
        matches = np.where(instruction_to_task == int(task_idx))[0]
        if matches.size == 0:
            available = sorted(set(instruction_to_task.tolist()))
            raise ValueError(
                f"Task idx {task_idx} not found. Available task idx: {available}"
            )
        return int(matches[0])

    if instruction_idx >= 0:
        n_instr = int(len(env.scenario_handler.scenario_data.instructions_list))
        if instruction_idx >= n_instr:
            raise ValueError(
                f"--instruction-idx={instruction_idx} out of range [0, {n_instr - 1}]"
            )
        return int(instruction_idx)

    return -1


def run_episode(env, network, params, rng, max_steps, instruction_idx, greedy):
    rng, reset_rng = jax.random.split(rng)
    obs, text_state = env.reset(reset_rng, instruction_idx=instruction_idx)

    num_agents = env.num_agents
    hstate = ScannedRNN.initialize_carry(num_agents, network.config["GRU_HIDDEN_DIM"])

    done_all = False
    step = 0
    instruction_done = 0.0
    task_idx = -1
    instruction_variant_idx = -1

    while (not done_all) and (step < max_steps):
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape((num_agents, -1))
        done_array = jnp.full((num_agents,), done_all, dtype=bool)
        ac_in = (obs_batch[jnp.newaxis, :], done_array[jnp.newaxis, :])

        rng, action_rng = jax.random.split(rng)
        hstate, pi, _ = network.apply(params, hstate, ac_in)
        if greedy:
            action = jnp.argmax(pi.probs, axis=-1).squeeze()
        else:
            action = pi.sample(seed=action_rng).squeeze()
        if action.ndim > 1:
            action = action.flatten()
        action_dict = {agent: jnp.int32(action[i]) for i, agent in enumerate(env.agents)}

        rng, step_rng = jax.random.split(rng)
        obs, text_state, _, done_dict, info = env.step(step_rng, text_state, action_dict)
        done_all = bool(done_dict["__all__"])

        user_info = info.get("user_info", {})
        instruction_done = float(
            user_info.get("Instructions/instruction_done", jnp.array([0.0]))[0]
        )
        task_idx = int(user_info.get("Instructions/task_idx", jnp.array([-1.0]))[0])
        instruction_variant_idx = int(
            user_info.get("Instructions/instruction_idx", jnp.array([-1.0]))[0]
        )
        step += 1

    return {
        "solved": int(instruction_done > 0.5),
        "task_idx": task_idx,
        "instruction_idx": instruction_variant_idx,
        "steps": step,
        "timed_out": int((not done_all) and (step >= max_steps)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate honest SR on inference")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pkl")
    parser.add_argument("--num-episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--instruction-idx",
        type=int,
        default=-1,
        help="Fix exact instruction variant index (default: random each episode)",
    )
    parser.add_argument(
        "--task-idx",
        type=int,
        default=None,
        help="Fix unique task index; uses first matching instruction variant",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use argmax actions instead of policy sampling",
    )
    parser.add_argument(
        "--print-per-task",
        action="store_true",
        help="Print per-task SR table",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="eval_sr_logs",
        help="Directory to save CSV/JSON logs for plotting",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name for output files",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Build and save plots after evaluation",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Disable plot generation",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N episodes (0 disables intermediate logs)",
    )
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint.get("config", {})
    env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    craftext_config_name = config.get("CRAFTEXT_CONFIG_NAME", "easy_train")

    env = make_craftax_env_with_my_wrapper(
        env_name, craftext_config_name=craftext_config_name
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

    params = restore_model_params(checkpoint)
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

    rng = jax.random.PRNGKey(args.seed)
    posed = 0
    solved = 0
    timeouts = 0

    per_task_count = {}
    per_task_solved = {}
    episode_rows = []

    for episode_idx in range(args.num_episodes):
        rng, ep_rng = jax.random.split(rng)
        result = run_episode(
            env=env,
            network=network,
            params=params,
            rng=ep_rng,
            max_steps=args.max_steps,
            instruction_idx=fixed_instruction_idx,
            greedy=args.greedy,
        )

        posed += 1
        solved += result["solved"]
        timeouts += result["timed_out"]
        cumulative_sr = solved / max(posed, 1)

        t_idx = result["task_idx"]
        per_task_count[t_idx] = per_task_count.get(t_idx, 0) + 1
        per_task_solved[t_idx] = per_task_solved.get(t_idx, 0) + result["solved"]
        episode_rows.append(
            {
                "episode": int(episode_idx),
                "solved": int(result["solved"]),
                "task_idx": int(result["task_idx"]),
                "instruction_idx": int(result["instruction_idx"]),
                "steps": int(result["steps"]),
                "timed_out": int(result["timed_out"]),
                "cumulative_solved": int(solved),
                "cumulative_posed": int(posed),
                "cumulative_sr": float(cumulative_sr),
            }
        )

        if args.progress_every > 0:
            current_episode = episode_idx + 1
            if (current_episode % args.progress_every == 0) or (
                current_episode == args.num_episodes
            ):
                print(
                    f"[{current_episode}/{args.num_episodes}] "
                    f"solved={solved} posed={posed} sr={cumulative_sr:.4f}"
                )

    overall_sr = solved / max(posed, 1)
    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"eval_{now}"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    episodes_csv_path = save_dir / f"{run_name}_episodes.csv"
    summary_json_path = save_dir / f"{run_name}_summary.json"
    cumulative_sr_plot_path = save_dir / f"{run_name}_cumulative_sr.png"
    per_task_plot_path = save_dir / f"{run_name}_per_task_sr.png"

    print("\n=== Inference SR ===")
    print(f"episodes: {posed}")
    print(f"solved: {solved}")
    print(f"timeouts: {timeouts}")
    print(f"sr_honest_micro: {overall_sr:.6f}")

    if args.print_per_task:
        print("\n=== Per-task SR ===")
        for t_idx in sorted(per_task_count.keys()):
            cnt = per_task_count[t_idx]
            ok = per_task_solved.get(t_idx, 0)
            sr = ok / max(cnt, 1)
            task_name = (
                env.task_names[t_idx]
                if isinstance(t_idx, (int, np.integer))
                and 0 <= int(t_idx) < len(env.task_names)
                else f"task_{t_idx}"
            )
            print(f"{int(t_idx):>3} | {task_name:<50} | sr={sr:.4f} | count={cnt}")

    with episodes_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "solved",
                "task_idx",
                "instruction_idx",
                "steps",
                "timed_out",
                "cumulative_solved",
                "cumulative_posed",
                "cumulative_sr",
            ],
        )
        writer.writeheader()
        writer.writerows(episode_rows)

    per_task_summary = []
    for t_idx in sorted(per_task_count.keys()):
        cnt = int(per_task_count[t_idx])
        ok = int(per_task_solved.get(t_idx, 0))
        sr = float(ok / max(cnt, 1))
        task_name = (
            env.task_names[t_idx]
            if isinstance(t_idx, (int, np.integer))
            and 0 <= int(t_idx) < len(env.task_names)
            else f"task_{t_idx}"
        )
        per_task_summary.append(
            {
                "task_idx": int(t_idx),
                "task_name": str(task_name),
                "solved": ok,
                "posed": cnt,
                "sr": sr,
            }
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "env_name": env_name,
        "craftext_config_name": craftext_config_name,
        "num_episodes": int(args.num_episodes),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "greedy": bool(args.greedy),
        "instruction_idx": int(args.instruction_idx),
        "task_idx": None if args.task_idx is None else int(args.task_idx),
        "episodes": int(posed),
        "solved": int(solved),
        "timeouts": int(timeouts),
        "sr_honest_micro": float(overall_sr),
        "per_task": per_task_summary,
        "episodes_csv": str(episodes_csv_path),
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved episode log: {episodes_csv_path}")
    print(f"Saved summary: {summary_json_path}")

    if args.plot:
        # Plot cumulative honest SR over episodes.
        episodes_x = [row["episode"] + 1 for row in episode_rows]
        cumulative_sr_y = [row["cumulative_sr"] for row in episode_rows]
        plt.figure(figsize=(10, 5))
        plt.plot(episodes_x, cumulative_sr_y, linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Cumulative SR")
        plt.title("Inference Honest SR (Solved / Posed)")
        plt.grid(True, alpha=0.3)
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(cumulative_sr_plot_path, dpi=160)
        plt.close()

        # Plot per-task SR (sorted by SR ascending).
        if per_task_summary:
            sorted_tasks = sorted(per_task_summary, key=lambda x: x["sr"])
            labels = [
                f'{item["task_idx"]}:{item["task_name"][:28]}'
                for item in sorted_tasks
            ]
            values = [item["sr"] for item in sorted_tasks]

            plt.figure(figsize=(12, max(4, len(values) * 0.28)))
            y_pos = np.arange(len(values))
            plt.barh(y_pos, values)
            plt.yticks(y_pos, labels)
            plt.xlabel("SR")
            plt.title("Per-task SR")
            plt.xlim(0.0, 1.0)
            plt.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(per_task_plot_path, dpi=160)
            plt.close()

        print(f"Saved cumulative SR plot: {cumulative_sr_plot_path}")
        if per_task_summary:
            print(f"Saved per-task SR plot: {per_task_plot_path}")


if __name__ == "__main__":
    main()
