"""
Code is adapted from the IPPO RNN implementation of JaxMARL (https://github.com/FLAIROx/JaxMARL/tree/main) 
Credit goes to the original authors: Rutherford et al.
IPPO под my_wrapper

"""

# ===========================
# Imports and Configuration
# ===========================
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import functools
import re
import yaml
from typing import Sequence, NamedTuple, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import chex

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import optax
import distrax
import pickle

try:
    from comet_ml import Experiment
except ImportError:
    raise ImportError(
        "Please install comet_ml for Comet logging: "
        "pip install comet_ml"
    )
from pathlib import Path

from jaxmarl.wrappers.baselines import LogWrapper
from craftax.craftax_together_talens.envs.patterns_wrapper import make_craftax_env_with_my_wrapper
import datetime

# ===========================
# Model Definitions
# ===========================
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

# ===========================
# Data Structures and Utilities
# ===========================
class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

# ===========================
# Training Function
# ===========================
def make_train(config, env):
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    def linear_schedule(count):
        num_minibatches = jnp.array(config["NUM_MINIBATCHES"], dtype=jnp.float32)
        update_epochs = jnp.array(config["UPDATE_EPOCHS"], dtype=jnp.float32)
        num_updates = jnp.array(config["NUM_UPDATES"], dtype=jnp.float32)
        lr = jnp.array(config["LR"], dtype=jnp.float32)
        
        frac = (
            1.0
            - jnp.floor(count / (num_minibatches * update_epochs))
            / num_updates
        )
        return lr * frac

    def train(rng):
        # Python-side cumulative metrics state for logging.
        # "Issued" tasks include in-flight episodes to avoid optimistic early SR.
        cumulative_total_issued = int(config["NUM_ENVS"])
        cumulative_total_finished = 0
        cumulative_total_solved = 0
        cumulative_task_counts = {}
        cumulative_task_solved = {}
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # adding an additional "fake" dimensionality to perform minibatching correctly
                init_hstate = jnp.reshape(
                    init_hstate, (1, config["NUM_ACTORS"], -1)
                )
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info

            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            
            # Calculate reward metrics
            mean_reward = traj_batch.reward.mean()
            total_reward = traj_batch.reward.sum()
            
            metric["update_steps"] = update_steps
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            metric["reward"] = {
                "mean_reward": mean_reward,
                "total_reward": total_reward,
            }

            rng = update_state[-1]

            def callback(metrics, step):
                nonlocal cumulative_total_issued
                nonlocal cumulative_total_finished
                nonlocal cumulative_total_solved
                nonlocal cumulative_task_counts
                nonlocal cumulative_task_solved
                experiment = config["experiment"]
                env_step = (
                    metrics["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                )
                
                # Log loss metrics
                for key, value in metrics["loss"].items():
                    experiment.log_metric(f"loss/{key}", float(value), step=step)
                
                # Log reward metrics
                for key, value in metrics["reward"].items():
                    experiment.log_metric(f"reward/{key}", float(value), step=step)
                
                experiment.log_metric("train/env_step", env_step, step=step)
                
                # Log episode metrics if available
                if metrics["returned_episode"].any():
                    # Use only the first agent channel to avoid per-agent duplication
                    # for episode-level metrics (one done event per environment episode).
                    returned_episode_mask = np.asarray(
                        metrics["returned_episode"][:, :, 0], dtype=bool
                    )
                    episode_info = jax.tree.map(
                        lambda x: x[:, :, 0][returned_episode_mask].mean(),
                        metrics["user_info"]
                    )
                    # If available, replace episode-level SR with cumulative honest SR.
                    instruction_done_key = "Instructions/instruction_done"
                    instruction_idx_key = "Instructions/task_idx"
                    if (
                        instruction_done_key in metrics["user_info"]
                        and instruction_idx_key in metrics["user_info"]
                    ):
                        instruction_idx = np.asarray(
                            metrics["user_info"][instruction_idx_key][:, :, 0][
                                returned_episode_mask
                            ],
                            dtype=np.int32,
                        )
                        instruction_done = np.asarray(
                            metrics["user_info"][instruction_done_key][:, :, 0][
                                returned_episode_mask
                            ],
                            dtype=np.float32,
                        )
                    else:
                        instruction_idx = np.array([], dtype=np.int32)
                        instruction_done = np.array([], dtype=np.float32)

                    if instruction_done.size > 0:
                        finished_now = int(instruction_done.size)
                        solved_now = int(instruction_done.sum())
                        cumulative_total_finished += finished_now
                        cumulative_total_issued += finished_now
                        cumulative_total_solved += solved_now
                        # Primary training SR: solved / issued (issued = finished + in-flight from start).
                        cumulative_honest_micro = cumulative_total_solved / max(
                            cumulative_total_issued, 1
                        )
                        episode_info["Instructions/SR"] = float(cumulative_honest_micro)

                    for key, value in episode_info.items():
                        experiment.log_metric(f"episode/{key}", float(value), step=step)

                    # Per-instruction cumulative SR logging.
                    if instruction_idx.size > 0:
                        task_names = config.get("TASK_NAMES", [])
                        for idx in np.unique(instruction_idx):
                            idx_mask = instruction_idx == idx
                            idx_count_window = int(idx_mask.sum())
                            idx_solved_window = int(instruction_done[idx_mask].sum())
                            task_idx = int(idx)

                            cumulative_task_counts[task_idx] = (
                                cumulative_task_counts.get(task_idx, 0) + idx_count_window
                            )
                            cumulative_task_solved[task_idx] = (
                                cumulative_task_solved.get(task_idx, 0) + idx_solved_window
                            )

                            idx_count_cum = cumulative_task_counts[task_idx]
                            idx_solved_cum = cumulative_task_solved[task_idx]
                            idx_sr_cum = idx_solved_cum / max(idx_count_cum, 1)

                            if 0 <= task_idx < len(task_names):
                                raw_name = str(task_names[task_idx])
                            else:
                                raw_name = f"task_{task_idx}"
                            task_name = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name).strip("_")
                            if not task_name:
                                task_name = f"task_{task_idx}"
                            experiment.log_metric(
                                f"instruction/sr_{task_name}",
                                float(idx_sr_cum),
                                step=step,
                            )
                            experiment.log_metric(
                                f"instruction/count_{task_name}",
                                float(idx_count_cum),
                                step=step,
                            )

                        # Global cumulative honest SR across all solved/posed tasks.
                        honest_micro_sr = cumulative_total_solved / max(
                            cumulative_total_issued, 1
                        )
                        experiment.log_metric(
                            "instruction/sr_honest_micro",
                            float(honest_micro_sr),
                            step=step,
                        )
                        experiment.log_metric(
                            "instruction/count_honest_micro",
                            float(cumulative_total_issued),
                            step=step,
                        )
                        # Diagnostics: finished-only SR for debugging selection bias.
                        finished_only_sr = cumulative_total_solved / max(
                            cumulative_total_finished, 1
                        )
                        experiment.log_metric(
                            "instruction/sr_honest_micro_finished_only",
                            float(finished_only_sr),
                            step=step,
                        )
                        experiment.log_metric(
                            "instruction/count_honest_micro_finished_only",
                            float(cumulative_total_finished),
                            step=step,
                        )
                        # Macro SR: mean of per-task SRs (each task weighted equally).
                        task_names_macro = config.get("TASK_NAMES", [])
                        if task_names_macro:
                            task_srs = [
                                cumulative_task_solved.get(t, 0) / max(cumulative_task_counts.get(t, 0), 1)
                                for t in range(len(task_names_macro))
                            ]
                            honest_macro_sr = float(np.mean(task_srs))
                            num_tasks = len(task_names_macro)
                        elif cumulative_task_counts:
                            task_srs = [
                                cumulative_task_solved[t] / max(cumulative_task_counts[t], 1)
                                for t in cumulative_task_counts
                            ]
                            honest_macro_sr = float(np.mean(task_srs))
                            num_tasks = len(cumulative_task_counts)
                        else:
                            honest_macro_sr = 0.0
                            num_tasks = 0
                        experiment.log_metric(
                            "instruction/sr_honest_macro",
                            honest_macro_sr,
                            step=int(step),
                        )
                        experiment.log_metric(
                            "instruction/count_honest_macro",
                            float(num_tasks),
                            step=int(step),
                        )
                        # Also log at top level for Comet visibility
                        experiment.log_metric(
                            "sr_honest_macro",
                            honest_macro_sr,
                            step=int(step),
                        )
                    else:
                        # Log macro even when no instruction data this batch (use cumulative state).
                        task_names_macro = config.get("TASK_NAMES", [])
                        if task_names_macro:
                            task_srs = [
                                cumulative_task_solved.get(t, 0) / max(cumulative_task_counts.get(t, 0), 1)
                                for t in range(len(task_names_macro))
                            ]
                            honest_macro_sr = float(np.mean(task_srs))
                            num_tasks = len(task_names_macro)
                        elif cumulative_task_counts:
                            task_srs = [
                                cumulative_task_solved[t] / max(cumulative_task_counts[t], 1)
                                for t in cumulative_task_counts
                            ]
                            honest_macro_sr = float(np.mean(task_srs))
                            num_tasks = len(cumulative_task_counts)
                        else:
                            honest_macro_sr = 0.0
                            num_tasks = 0
                        experiment.log_metric(
                            "instruction/sr_honest_macro",
                            honest_macro_sr,
                            step=int(step),
                        )
                        experiment.log_metric(
                            "instruction/count_honest_macro",
                            float(num_tasks),
                            step=int(step),
                        )
                        experiment.log_metric(
                            "sr_honest_macro",
                            honest_macro_sr,
                            step=int(step),
                        )
                    
                    episode_lengths = metrics["returned_episode_lengths"][:, :, 0][
                        metrics["returned_episode"][:, :, 0]
                    ].mean()
                    episode_returns = metrics["returned_episode_returns"][:, :, 0][
                        metrics["returned_episode"][:, :, 0]
                    ].mean()
                    
                    experiment.log_metric("episode/lengths", float(episode_lengths), step=step)
                    experiment.log_metric("episode/returns", float(episode_returns), step=step)
                    
                    # Log episode returns as reward metric
                    experiment.log_metric("reward/episode_returns", float(episode_returns), step=step)
                
                # Print for console output
                to_log = {
                    "env_step": env_step,
                    **metrics["loss"],
                    **metrics["reward"],
                }
                if metrics["returned_episode"].any():
                    returned_episode_mask = np.asarray(
                        metrics["returned_episode"][:, :, 0], dtype=bool
                    )
                    to_log.update(jax.tree.map(
                        lambda x: x[:, :, 0][returned_episode_mask].mean(),
                        metrics["user_info"]
                    ))
                    to_log["episode_lengths"] = metrics["returned_episode_lengths"][:, :, 0][
                        metrics["returned_episode"][:, :, 0]
                    ].mean()
                    to_log["episode_returns"] = metrics["returned_episode_returns"][:, :, 0][
                        metrics["returned_episode"][:, :, 0]
                    ].mean()
                    # Add macro to console output for verification
                    task_names_tl = config.get("TASK_NAMES", [])
                    if task_names_tl:
                        task_srs_tl = [
                            cumulative_task_solved.get(t, 0) / max(cumulative_task_counts.get(t, 0), 1)
                            for t in range(len(task_names_tl))
                        ]
                        to_log["sr_honest_macro"] = float(np.mean(task_srs_tl))
                    elif cumulative_task_counts:
                        task_srs_tl = [
                            cumulative_task_solved[t] / max(cumulative_task_counts[t], 1)
                            for t in cumulative_task_counts
                        ]
                        to_log["sr_honest_macro"] = float(np.mean(task_srs_tl))
                    else:
                        to_log["sr_honest_macro"] = 0.0
                print(to_log)

            jax.experimental.io_callback(callback, None, metric, update_steps)
            
            # Save checkpoint using io_callback to avoid JIT tracing issues
            # Note: io_callback only accepts JAX types, so we use closure to access config
            checkpoint_dir_str = config.get("CHECKPOINT_DIR", config.get("log_path", "checkpoints"))
            checkpoint_interval = config.get("CHECKPOINT_INTERVAL", 100)
            config_dict = {k: v for k, v in config.items() if k not in ['experiment', 'log_path']}
            
            def save_checkpoint_callback(train_state_params, train_state_opt_state, update_steps_val):
                # This runs on CPU, outside JIT
                # Access config from closure
                if update_steps_val % checkpoint_interval == 0:
                    checkpoint_dir = Path(checkpoint_dir_str)
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Convert JAX arrays to numpy for saving
                    try:
                        tree_map_fn = jax.tree.map
                    except AttributeError:
                        tree_map_fn = jax.tree_util.tree_map
                    
                    checkpoint_data = {
                        'train_state': {
                            'params': tree_map_fn(lambda x: np.array(jax.device_get(x)), train_state_params),
                            'opt_state': tree_map_fn(
                                lambda x: np.array(jax.device_get(x)) if hasattr(x, 'shape') else x,
                                train_state_opt_state
                            ),
                        },
                        'update_steps': int(update_steps_val),
                        'config': config_dict,
                    }
                    
                    checkpoint_file = checkpoint_dir / f'model_update_{update_steps_val}.pkl'
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    print(f"Saved checkpoint to {checkpoint_file}")
            
            # Only pass JAX types to io_callback
            jax.experimental.io_callback(
                save_checkpoint_callback,
                None,
                train_state.params,
                train_state.opt_state,
                update_steps
            )
            
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

# ===========================
# Main Run Function
# ===========================
def single_run(config):
    alg_name = config.get("ALG_NAME", "ippo-rnn")
    env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    craftext_config_name = config.get("CRAFTEXT_CONFIG_NAME", "easy_train")
    # Используем instruction-only окружение с CrafText.
    env = make_craftax_env_with_my_wrapper(
        env_name,
        craftext_config_name=craftext_config_name,
    )
    if hasattr(env, "task_names"):
        config["TASK_NAMES"] = list(env.task_names)

    # Setup Comet ML logging
    run_name = config.get("RUN_NAME", "ippo-rnn-run")
    log_dir = config.get("LOG_DIR", "runs")

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%d_%m_%H_%M")

    log_path = Path(log_dir) / f"{alg_name}_{env_name}_{run_name}" / f"log_{formatted_datetime}"
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize Comet experiment
    experiment = Experiment(
        project_name=config.get("COMET_PROJECT_NAME", "craftax-multiagent"),
        workspace=config.get("COMET_WORKSPACE", None),
        experiment_name=f"{alg_name}_{env_name}_{run_name}_{formatted_datetime}",
        auto_param_logging=False,  # We'll log parameters manually
        auto_metric_logging=False,  # We'll log metrics manually
    )
    
    # Log hyperparameters
    params_to_log = {
        "ALG_NAME": alg_name,
        "ENV_NAME": env_name,
        "CRAFTEXT_CONFIG_NAME": craftext_config_name,
        "NUM_ENVS": config.get("NUM_ENVS"),
        "NUM_STEPS": config.get("NUM_STEPS"),
        "NUM_MINIBATCHES": config.get("NUM_MINIBATCHES"),
        "UPDATE_EPOCHS": config.get("UPDATE_EPOCHS"),
        "LR": config.get("LR"),
        "GAMMA": config.get("GAMMA"),
        "GAE_LAMBDA": config.get("GAE_LAMBDA"),
        "CLIP_EPS": config.get("CLIP_EPS"),
        "ENT_COEF": config.get("ENT_COEF"),
        "VF_COEF": config.get("VF_COEF"),
        "MAX_GRAD_NORM": config.get("MAX_GRAD_NORM"),
        "SEED": config.get("SEED"),
        "NUM_SEEDS": config.get("NUM_SEEDS"),
        "TOTAL_TIMESTEPS": config.get("TOTAL_TIMESTEPS"),
    }
    experiment.log_parameters(params_to_log)
    
    config["experiment"] = experiment
    config["log_path"] = str(log_path)  # Store for checkpoint saving
    
    # Setup checkpoint directory
    checkpoint_dir = config.get("CHECKPOINT_DIR", None)
    if checkpoint_dir is None:
        checkpoint_dir = log_path / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config["CHECKPOINT_DIR"] = str(checkpoint_dir)
    
    # Set checkpoint interval if not specified
    if "CHECKPOINT_INTERVAL" not in config:
        config["CHECKPOINT_INTERVAL"] = 100  # Save every 100 update steps
    
    experiment_key = experiment.get_key() if hasattr(experiment, 'get_key') else experiment.id
    workspace = experiment.workspace if experiment.workspace else "default"
    project_name = experiment.project_name if experiment.project_name else "craftax-multiagent"
    print(f"Comet ML experiment: {experiment_key}")
    if experiment_key:
        print(f"View logs at: https://www.comet.ml/{workspace}/{project_name}/{experiment_key}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Checkpoint interval: every {config['CHECKPOINT_INTERVAL']} update steps")

    rng = jax.random.PRNGKey(config["SEED"])

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))
    
    experiment.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Name of the config YAML file (in baselines/config/)")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "config", args.config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    single_run(config)


if __name__ == "__main__":
    main()
