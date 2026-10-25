"""
Visualize episodes from an IPPO checkpoint (current CrafText integration).

Usage:
    python baselines/my_visualize.py checkpoints/model_update_100.pkl
    python baselines/my_visualize.py checkpoints/model_update_100.pkl --all-agents --num-episodes 2
    python baselines/my_visualize.py checkpoints/model_update_100.pkl --instruction-idx 40
    python baselines/my_visualize.py checkpoints/model_update_100.pkl --task-idx 6
"""

import argparse
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from my_ippo import ActorCriticRNN, ScannedRNN
from craftax.craftax_together_talens.envs.patterns_wrapper import (
    make_craftax_env_with_my_wrapper,
)

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    try:
        import cv2

        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False


def load_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
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


def render_episode(
    env,
    network,
    params,
    rng,
    max_steps=10000,
    log_file=None,
    instruction_idx=-1,
):
    """
    Run one episode and collect base environment states for pixel rendering.
    """
    rng, reset_rng = jax.random.split(rng)
    obs, text_state = env.reset(reset_rng, instruction_idx=instruction_idx)

    num_agents = env.num_agents
    hstate = ScannedRNN.initialize_carry(num_agents, network.config["GRU_HIDDEN_DIM"])

    # We render base states (inside adapter state).
    base_states = [text_state.env_state]
    done_all = False
    step = 0

    log_fp = None
    if log_file:
        log_fp = open(log_file, "w")
        log_fp.write(
            "step,agent,action,reward,done_all,instruction_idx,task_idx,instruction_done,"
            "pos_x,pos_y,health,food,drink,alive\n"
        )

    while (not done_all) and (step < max_steps):
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape((num_agents, -1))
        done_array = jnp.full((num_agents,), done_all, dtype=bool)
        ac_in = (obs_batch[jnp.newaxis, :], done_array[jnp.newaxis, :])

        rng, action_rng = jax.random.split(rng)
        hstate, pi, _ = network.apply(params, hstate, ac_in)
        action = pi.sample(seed=action_rng).squeeze()
        if action.ndim > 1:
            action = action.flatten()
        action_dict = {agent: jnp.int32(action[i]) for i, agent in enumerate(env.agents)}

        rng, step_rng = jax.random.split(rng)
        obs, text_state, reward, done_dict, info = env.step(step_rng, text_state, action_dict)
        done_all = bool(done_dict["__all__"])
        base_states.append(text_state.env_state)

        if log_fp:
            user_info = info.get("user_info", {})
            instruction_done = float(user_info.get("Instructions/instruction_done", jnp.array([0.0]))[0])
            instruction_idx = float(user_info.get("Instructions/instruction_idx", jnp.array([-1.0]))[0])
            task_idx = float(user_info.get("Instructions/task_idx", jnp.array([-1.0]))[0])
            base_state = text_state.env_state
            for i, agent in enumerate(env.agents):
                pos = base_state.player_position[i]
                log_fp.write(
                    f"{step},{agent},{int(action[i])},{float(reward.get(agent, 0.0)):.4f},"
                    f"{int(done_all)},{instruction_idx:.0f},{task_idx:.0f},{instruction_done:.0f},"
                    f"{int(pos[0])},{int(pos[1])},{float(base_state.player_health[i]):.2f},"
                    f"{int(base_state.player_food[i])},{int(base_state.player_drink[i])},"
                    f"{int(base_state.player_alive[i])}\n"
                )
        step += 1

    if log_fp:
        log_fp.close()
        print(f"Saved CSV log: {log_file}")

    print(f"Episode finished. Steps: {step}, Frames: {len(base_states)}")
    return base_states


def render_video_pixels(
    states, agent_idx, filename, fps=20, block_pixel_size=64, use_night_noise=True
):
    if not IMAGEIO_AVAILABLE and not CV2_AVAILABLE:
        raise ImportError("Install imageio[ffmpeg] or opencv-python")

    from craftax.craftax_together_talens.renderer.renderer_pixels import render_craftax_pixels
    from craftax.craftax_together_talens.craftax_state import StaticEnvParams
    from craftax.craftax_together_talens.constants import TEXTURES, load_player_specific_textures

    print(f"Rendering {len(states)} frames -> {filename}")
    if len(states) == 0:
        raise ValueError("No states to render")

    num_agents = int(states[0].player_position.shape[0])
    static_params = StaticEnvParams(player_count=num_agents)
    textures = TEXTURES[block_pixel_size]
    player_specific_textures = load_player_specific_textures(textures, static_params.player_count)

    try:
        from jax.experimental import disable_jit
    except ImportError:
        try:
            from jax import disable_jit
        except ImportError:
            disable_jit = None

    frames = []
    for state in states:
        if disable_jit is not None:
            with disable_jit():
                pixels = render_craftax_pixels(
                    state,
                    block_pixel_size,
                    static_params,
                    player_specific_textures,
                    do_night_noise=use_night_noise,
                )[agent_idx]
        else:
            pixels = render_craftax_pixels(
                state,
                block_pixel_size,
                static_params,
                player_specific_textures,
                do_night_noise=use_night_noise,
            )[agent_idx]
        frame = np.asarray(pixels)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)

    if IMAGEIO_AVAILABLE:
        imageio.mimsave(
            filename,
            frames,
            fps=fps,
            codec="libx264",
            quality=6,
            pixelformat="yuv420p",
            ffmpeg_params=["-preset", "ultrafast", "-crf", "23"],
        )
    else:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, float(fps), (width, height))
        if not out.isOpened():
            raise RuntimeError(f"Failed to open video writer for {filename}")
        for frame in frames:
            out.write(frame[..., ::-1])  # RGB->BGR
        out.release()
    print(f"Saved video: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate videos from checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pkl")
    parser.add_argument("--agent", type=str, default=None, help="Single agent name")
    parser.add_argument("--all-agents", action="store_true", help="Render all agents")
    parser.add_argument("--output", type=str, default=None, help="Output video filename")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-night-noise", action="store_true", help="Disable night noise")
    parser.add_argument("--log-file", type=str, default=None, help="CSV log file path")
    parser.add_argument("--no-log", action="store_true", help="Disable CSV logging")
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
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint.get("config", {})
    update_steps = checkpoint.get("update_steps", "unknown")
    env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    craftext_config_name = config.get("CRAFTEXT_CONFIG_NAME", "easy_train")

    print(f"Checkpoint update: {update_steps}")
    print(f"Environment: {env_name}, CrafText config: {craftext_config_name}")

    env = make_craftax_env_with_my_wrapper(
        env_name,
        craftext_config_name=craftext_config_name,
    )

    if args.instruction_idx >= 0 and args.task_idx is not None:
        raise ValueError("Use only one of --instruction-idx or --task-idx")

    instruction_idx_for_reset = int(args.instruction_idx)
    if args.task_idx is not None:
        task_idx = int(args.task_idx)
        instruction_to_task = np.asarray(env.instruction_to_task_idx).astype(int).reshape(-1)
        matches = np.where(instruction_to_task == task_idx)[0]
        if matches.size == 0:
            raise ValueError(
                f"Task idx {task_idx} not found. Available task idx: "
                f"{sorted(set(instruction_to_task.tolist()))}"
            )
        instruction_idx_for_reset = int(matches[0])

    if instruction_idx_for_reset >= 0:
        n_instr = int(len(env.scenario_handler.scenario_data.instructions_list))
        if instruction_idx_for_reset >= n_instr:
            raise ValueError(
                f"--instruction-idx={instruction_idx_for_reset} out of range [0, {n_instr - 1}]"
            )
        task_idx_value = int(np.asarray(env.instruction_to_task_idx)[instruction_idx_for_reset])
        task_name = env.task_names[task_idx_value] if 0 <= task_idx_value < len(env.task_names) else "unknown"
        instruction_text = env.scenario_handler.scenario_data.instructions_list[instruction_idx_for_reset]
        print(
            "Fixed instruction:"
            f" instruction_idx={instruction_idx_for_reset},"
            f" task_idx={task_idx_value},"
            f" task_name={task_name}\n"
            f"Text: {instruction_text}"
        )

    params = restore_model_params(checkpoint)
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

    if args.all_agents:
        agents_to_visualize = env.agents
    elif args.agent is not None:
        if args.agent not in env.agents:
            raise ValueError(f"Unknown agent '{args.agent}', available: {env.agents}")
        agents_to_visualize = [args.agent]
    else:
        agents_to_visualize = env.agents

    output_dir = "videos_from_checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    rng = jax.random.PRNGKey(args.seed)
    for episode_idx in range(args.num_episodes):
        print(f"\n=== Episode {episode_idx + 1}/{args.num_episodes} ===")
        rng, episode_rng = jax.random.split(rng)

        log_file = None
        if not args.no_log:
            if args.log_file:
                if args.num_episodes > 1:
                    p = Path(args.log_file)
                    log_file = f"{p.stem}_episode_{episode_idx}{p.suffix or '.csv'}"
                else:
                    log_file = args.log_file
            else:
                base = Path(args.checkpoint).stem
                suffix = f"_episode_{episode_idx}" if args.num_episodes > 1 else ""
                log_file = os.path.join(output_dir, f"{base}{suffix}_log.csv")

        states = render_episode(
            env,
            network,
            params,
            episode_rng,
            max_steps=args.max_steps,
            log_file=log_file,
            instruction_idx=instruction_idx_for_reset,
        )

        for agent_idx, agent_name in enumerate(agents_to_visualize):
            if args.output and args.num_episodes == 1 and len(agents_to_visualize) == 1:
                video_filename = args.output
            else:
                base = Path(args.checkpoint).stem
                suffix = f"_episode_{episode_idx}" if args.num_episodes > 1 else ""
                video_filename = os.path.join(
                    output_dir, f"{base}{suffix}_agent_{agent_name}.mp4"
                )

            render_video_pixels(
                states=states,
                agent_idx=agent_idx,
                filename=video_filename,
                fps=args.fps,
                use_night_noise=not args.no_night_noise,
            )

    print(f"\nDone. Videos saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
