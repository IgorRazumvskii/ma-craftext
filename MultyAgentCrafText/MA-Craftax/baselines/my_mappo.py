"""
Code is adapted from the MAPPO RNN implementation of JaxMARL (https://github.com/FLAIROx/JaxMARL/tree/main)
Credit goes to the original authors: Rutherford et al.
MAPPO под my_wrapper (CrafText / Comet / чекпоинты), по аналогии с my_ippo.py.
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
import pickle
import yaml
import datetime
from functools import partial
from pathlib import Path
from typing import Sequence, NamedTuple, Dict

import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import optax
import distrax

try:
    from comet_ml import Experiment
except ImportError:
    raise ImportError(
        "Please install comet_ml for Comet logging: pip install comet_ml"
    )

from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.wrappers.baselines import JaxMARLWrapper

from craftax.craftax_together_talens.envs.patterns_wrapper import make_craftax_env_with_my_wrapper
from craftax.craftax_together_talens.game_logic.constants import (
    Specialization,
    get_team_action_masks,
)

# Finite stand-in for a masked-out logit. -inf breaks Categorical entropy.
_ILLEGAL_ACTION_LOGIT = jnp.float32(-1e8)


# ===========================
# Environment Wrappers
# ===========================
class WorldStateWrapper(JaxMARLWrapper):
    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """
        For each agent: [agent obs, all other agent obs]
        """

        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs

        all_obs = jnp.array([obs[agent] for agent in self._env.agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs

    def world_state_size(self):
        spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        return sum([space.shape[-1] for space in spaces])


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
        """Applies the module."""
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


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, action_mask = x
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
        # Same mask at sampling and at the PPO log_prob recomputation.
        action_logits = jnp.where(action_mask, action_logits, _ILLEGAL_ACTION_LOGIT)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, jnp.squeeze(critic, axis=-1)


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
    world_state: jnp.ndarray
    action_mask: jnp.ndarray
    info: jnp.ndarray


def build_actor_action_mask(env, num_envs: int) -> jnp.ndarray:
    """Role mask laid out like batchify: agent 0's envs, then agent 1, then agent 2.

    Spawn order is miner, builder, warrior (CRAFTER), repeated if there are more agents.
    """
    action_dim = int(env.action_space(env.agents[0]).n)
    role_order = jnp.array(
        [
            Specialization.MINER.value,
            Specialization.BUILDER.value,
            Specialization.CRAFTER.value,
        ],
        dtype=jnp.int32,
    )
    specializations = role_order[jnp.arange(env.num_agents) % 3]
    role_masks = get_team_action_masks(specializations, action_dim)
    return jnp.repeat(role_masks, num_envs, axis=0)


def _dump_checkpoint_atomic(path: Path, checkpoint_data) -> None:
    """Write a pickle via a temp file in the same directory, then replace the previous one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def unbatchify_actions(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs))
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
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = WorldStateWrapper(env)
    env = LogWrapper(env)
    actor_action_mask = build_actor_action_mask(env, config["NUM_ENVS"])

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
        # Python-side cumulative metrics for logging (same idea as my_ippo.py).
        cumulative_total_issued = int(config["NUM_ENVS"])
        cumulative_total_finished = 0
        cumulative_total_solved = 0
        cumulative_task_counts = {}
        cumulative_task_solved = {}

        action_dim = env.action_space(env.agents[0]).n
        actor_network = ActorRNN(action_dim, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.ones((1, config["NUM_ENVS"], action_dim), dtype=bool),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.world_state_size())),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state

                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    actor_action_mask[np.newaxis, :],
                )
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify_actions(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                world_state = last_obs["world_state"].swapaxes(0, 1)
                world_state = world_state.reshape((config["NUM_ACTORS"], -1))
                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                # Как в my_ippo.py: не трогать info — у LogWrapper часть полей (NUM_ENVS,), не (NUM_ACTORS,).
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    actor_action_mask,
                    info,
                )
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            train_states, env_state, last_obs, last_done, hstates, rng = runner_state

            last_world_state = last_obs["world_state"].swapaxes(0, 1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"], -1))
            cr_in = (
                last_world_state[None, :],
                last_done[np.newaxis, :],
            )
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
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

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.action_mask),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        _, value = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.world_state, traj_batch.done),
                        )

                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss,)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstates)

                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            # Как в my_ippo.py: ratio_0 до усреднения по эпохам обновления.
            ratio_0 = loss_info["ratio"].at[0, 0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            loss_info["ratio_0"] = ratio_0

            train_states = update_state[0]
            metric = traj_batch.info
            mean_reward = traj_batch.reward.mean()
            total_reward = traj_batch.reward.sum()

            metric["update_steps"] = update_steps
            metric["loss"] = loss_info
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

                for key, value in metrics["loss"].items():
                    experiment.log_metric(f"loss/{key}", float(value), step=step)

                for key, value in metrics["reward"].items():
                    experiment.log_metric(f"reward/{key}", float(value), step=step)

                experiment.log_metric("train/env_step", env_step, step=step)

                def _honest_macro():
                    task_names_macro = config.get("TASK_NAMES", [])
                    candidate_ids = (
                        range(len(task_names_macro))
                        if task_names_macro
                        else list(cumulative_task_counts)
                    )
                    ratios = [
                        cumulative_task_solved[task_id] / cumulative_task_counts[task_id]
                        for task_id in candidate_ids
                        if cumulative_task_counts.get(task_id, 0) > 0
                    ]
                    if not ratios:
                        return 0.0, 0
                    return float(np.mean(ratios)), len(ratios)

                if metrics["returned_episode"].any():
                    # Только первый агент — как в my_ippo.py (один эпизод на env).
                    returned_episode_mask = np.asarray(
                        metrics["returned_episode"][:, :, 0], dtype=bool
                    )
                    episode_info = jax.tree.map(
                        lambda x: x[:, :, 0][returned_episode_mask].mean(),
                        metrics["user_info"],
                    )
                    instruction_done_key = "Instructions/instruction_done"
                    instruction_idx_key = "Instructions/task_idx"
                    if (
                        instruction_done_key in metrics["user_info"]
                        and instruction_idx_key in metrics["user_info"]
                    ):
                        instruction_idx = np.asarray(
                            metrics["user_info"][instruction_idx_key][:, :, 0][returned_episode_mask],
                            dtype=np.int32,
                        )
                        instruction_done = np.asarray(
                            metrics["user_info"][instruction_done_key][:, :, 0][returned_episode_mask],
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
                        cumulative_honest_micro = cumulative_total_solved / max(
                            cumulative_total_issued, 1
                        )
                        episode_info["Instructions/SR"] = float(cumulative_honest_micro)

                    for key, value in episode_info.items():
                        experiment.log_metric(f"episode/{key}", float(value), step=step)

                    # Lifetime SR of completed episodes for this task:
                    # successes / finished episodes. Each episode is counted once.
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
                            experiment.log_metric(f"instruction/sr_{task_name}", float(idx_sr_cum), step=step)
                            experiment.log_metric(f"instruction/count_{task_name}", float(idx_count_cum), step=step)

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

                        finished_only_sr = cumulative_total_solved / max(cumulative_total_finished, 1)
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

                        honest_macro_sr, num_tasks = _honest_macro()
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
                    else:
                        honest_macro_sr, num_tasks = _honest_macro()
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
                    experiment.log_metric(
                        "reward/episode_returns", float(episode_returns), step=step
                    )

                # Print for console output
                to_log = {
                    "env_step": env_step,
                    **metrics["loss"],
                    **metrics["reward"],
                }
                if metrics["returned_episode"].any():
                    returned_episode_mask = np.asarray(metrics["returned_episode"][:, :, 0], dtype=bool)
                    to_log.update(
                        jax.tree.map(
                            lambda x: x[:, :, 0][returned_episode_mask].mean(),
                            metrics["user_info"],
                        )
                    )
                    to_log["episode_lengths"] = metrics["returned_episode_lengths"][:, :, 0][
                        metrics["returned_episode"][:, :, 0]
                    ].mean()
                    to_log["episode_returns"] = metrics["returned_episode_returns"][:, :, 0][
                        metrics["returned_episode"][:, :, 0]
                    ].mean()
                    # Add macro to console output for verification
                    to_log["sr_honest_macro"] = _honest_macro()[0]
                print(to_log)

            jax.experimental.io_callback(callback, None, metric, update_steps)

            # Save checkpoint (как my_ippo.py: только JAX-аргументы в io_callback).
            checkpoint_dir_str = config.get("CHECKPOINT_DIR", config.get("log_path", "checkpoints"))
            checkpoint_interval = config.get("CHECKPOINT_INTERVAL", 100)
            config_dict = {k: v for k, v in config.items() if k not in ["experiment", "log_path"]}

            def save_checkpoint_callback(
                actor_params,
                actor_opt_state,
                critic_params,
                critic_opt_state,
                update_steps_val,
            ):
                if update_steps_val % checkpoint_interval != 0:
                    return
                checkpoint_dir = Path(checkpoint_dir_str)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                try:
                    tree_map_fn = jax.tree.map
                except AttributeError:
                    tree_map_fn = jax.tree_util.tree_map

                def params_to_numpy(tree):
                    return tree_map_fn(lambda x: np.array(jax.device_get(x)), tree)

                def opt_to_numpy(tree):
                    return tree_map_fn(
                        lambda x: np.array(jax.device_get(x)) if hasattr(x, "shape") else x,
                        tree,
                    )

                checkpoint_data = {
                    "actor_train_state": {
                        "params": params_to_numpy(actor_params),
                        "opt_state": opt_to_numpy(actor_opt_state),
                    },
                    "critic_train_state": {
                        "params": params_to_numpy(critic_params),
                        "opt_state": opt_to_numpy(critic_opt_state),
                    },
                    "update_steps": int(update_steps_val),
                    "config": config_dict,
                }
                checkpoint_file = checkpoint_dir / "mappo_model_latest.pkl"
                _dump_checkpoint_atomic(checkpoint_file, checkpoint_data)
                print(
                    f"Saved MAPPO checkpoint step {int(update_steps_val)} "
                    f"to {checkpoint_file}"
                )
                persistent_dir = config.get("PERSISTENT_CHECKPOINT_DIR")
                if persistent_dir:
                    persistent_file = Path(persistent_dir) / "mappo_model_latest.pkl"
                    if persistent_file.resolve() != checkpoint_file.resolve():
                        try:
                            _dump_checkpoint_atomic(persistent_file, checkpoint_data)
                            print(
                                f"Saved persistent checkpoint step {int(update_steps_val)} "
                                f"to {persistent_file}"
                            )
                        except OSError as persist_err:
                            print(
                                f"Persistent checkpoint was not saved to {persistent_file}: "
                                f"{persist_err}"
                            )

                # One Hub file is replaced on every checkpoint. Older names are deleted.
                try:
                    from hf_hub_utils import upload_checkpoint_replacing_previous

                    uploaded_paths = upload_checkpoint_replacing_previous(
                        checkpoint_file,
                        int(update_steps_val),
                        repo_id=config.get("HF_HUB_REPO_ID"),
                        token=config.get("HF_TOKEN"),
                        repo_type=config.get("HF_HUB_REPO_TYPE", "model"),
                        path_in_repo_prefix=config.get("HF_HUB_PATH_PREFIX", ""),
                        private=bool(config.get("HF_HUB_PRIVATE", True)),
                        enabled=bool(config.get("HF_HUB_UPLOAD", False)),
                        squash_history=bool(config.get("HF_HUB_SQUASH_HISTORY", False)),
                    )
                    if (
                        uploaded_paths
                        and config.get("DELETE_LOCAL_CHECKPOINT_AFTER_UPLOAD", False)
                    ):
                        checkpoint_file.unlink(missing_ok=True)
                        checkpoint_file.with_name("latest_step.txt").unlink(missing_ok=True)
                        print(
                            f"[hf_hub] Removed local checkpoint after upload: "
                            f"{checkpoint_file}"
                        )
                except Exception as hf_err:
                    print(f"[hf_hub] Checkpoint upload failed (local file kept): {hf_err}")

            jax.experimental.io_callback(
                save_checkpoint_callback,
                None,
                train_states[0].params,
                train_states[0].opt_state,
                train_states[1].params,
                train_states[1].opt_state,
                update_steps,
            )

            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(_update_step, (runner_state, 0), None, config["NUM_UPDATES"])
        return {"runner_state": runner_state}

    return train


# ===========================
# Main Run Function
# ===========================
def single_run(config):
    alg_name = config.get("ALG_NAME", "mappo-rnn")
    env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    craftext_config_name = config.get("CRAFTEXT_CONFIG_NAME", "easy_train")

    # Используем instruction-only окружение с CrafText (как my_ippo.py).
    env = make_craftax_env_with_my_wrapper(
        env_name,
        craftext_config_name=craftext_config_name,
    )
    if hasattr(env, "task_names"):
        config["TASK_NAMES"] = list(env.task_names)

    run_name = config.get("RUN_NAME", "mappo-rnn-run")
    log_dir = config.get("LOG_DIR", "runs")
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%d_%m_%H_%M")
    log_path = Path(log_dir) / f"{alg_name}_{env_name}_{run_name}" / f"log_{formatted_datetime}"
    log_path.mkdir(parents=True, exist_ok=True)

    experiment_name = f"{alg_name}_{env_name}_{run_name}_{formatted_datetime}"
    experiment = Experiment(
        project_name=config.get("COMET_PROJECT_NAME", "craftax-multiagent"),
        workspace=config.get("COMET_WORKSPACE", None),
        auto_param_logging=False,
        auto_metric_logging=False,
        disabled=bool(config.get("COMET_DISABLED", False)),
    )
    experiment.set_name(experiment_name)

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
        "SCALE_CLIP_EPS": config.get("SCALE_CLIP_EPS"),
        "COMET_PROJECT_NAME": config.get("COMET_PROJECT_NAME"),
        "COMET_WORKSPACE": config.get("COMET_WORKSPACE"),
    }
    experiment.log_parameters(params_to_log)

    config["experiment"] = experiment
    config["log_path"] = str(log_path)

    checkpoint_dir = config.get("CHECKPOINT_DIR", None)
    if checkpoint_dir is None:
        checkpoint_dir = log_path / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config["CHECKPOINT_DIR"] = str(checkpoint_dir)

    if "CHECKPOINT_INTERVAL" not in config:
        config["CHECKPOINT_INTERVAL"] = 100

    experiment_key = experiment.get_key() if hasattr(experiment, "get_key") else experiment.id
    workspace = experiment.workspace if experiment.workspace else "default"
    project_name = experiment.project_name if experiment.project_name else "craftax-multiagent"
    print(f"Comet ML experiment: {experiment_key}")
    if experiment_key:
        print(f"View logs at: https://www.comet.ml/{workspace}/{project_name}/{experiment_key}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Checkpoint interval: every {config['CHECKPOINT_INTERVAL']} update steps")
    if config.get("HF_HUB_UPLOAD"):
        print(
            f"HF Hub checkpoint upload: ON "
            f"(repo={config.get('HF_HUB_REPO_ID') or os.environ.get('HF_HUB_REPO_ID')})"
        )
    else:
        print("HF Hub checkpoint upload: OFF (set HF_HUB_UPLOAD=1 + HF_HUB_REPO_ID + HF_TOKEN to enable)")

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    experiment.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Name of the config YAML file (in baselines/config/)")
    parser.add_argument(
        "--hf-upload",
        action="store_true",
        help="Upload each saved checkpoint to Hugging Face Hub (needs HF_HUB_REPO_ID + HF_TOKEN)",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Hugging Face Hub repo id for checkpoints (overrides config / env)",
    )
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "config", args.config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if args.hf_upload:
        config["HF_HUB_UPLOAD"] = True
    if args.hf_repo_id:
        config["HF_HUB_REPO_ID"] = args.hf_repo_id
    # Env can enable upload without CLI flags (see hf_hub_utils / run_mappo_train.sh).
    if os.environ.get("HF_HUB_UPLOAD", "").strip().lower() in {"1", "true", "yes", "on"}:
        config["HF_HUB_UPLOAD"] = True
    if os.environ.get("HF_HUB_REPO_ID") and not config.get("HF_HUB_REPO_ID"):
        config["HF_HUB_REPO_ID"] = os.environ["HF_HUB_REPO_ID"]
    single_run(config)


if __name__ == "__main__":
    main()
