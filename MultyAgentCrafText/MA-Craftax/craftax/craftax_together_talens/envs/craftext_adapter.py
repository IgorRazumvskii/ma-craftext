import sys
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments import spaces

# Allow importing CrafText package from sibling repository directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CRAFTEXT_ROOT = _PROJECT_ROOT / "CrafText"
if str(_CRAFTEXT_ROOT) not in sys.path:
    sys.path.append(str(_CRAFTEXT_ROOT))

from craftext.environment.craftext_constants import Scenarios
from craftext.environment.craftext_constants import Achievement as CrafTextAchievement
from craftext.environment.craftext_wrapper import InstructionWrapper
from craftext.environment.scenarious.checkers.achivments import checker_acvievments
from craftext.environment.scenarious.checkers.conditional import (
    checker_conditional_placement,
)
from craftext.environment.scenarious.checkers.relevant import cheker_localization
from craftext.environment.scenarious.checkers.building_line import checker_line
from craftext.environment.scenarious.checkers.building_square import checker_square
from craftext.environment.scenarious.checkers.building_star import checker_star
from craftext.environment.scenarious.checkers.time_constrained import (
    checker_time_placement,
)
from craftext.environment.states.state import GameData
from craftax_together_talens.constants import Achievement as MAAchievement


def _build_achievement_index_mapping():
    """
    Build mapping from CrafText achievement index space to MA-Craftax achievement index space.

    Returns:
        ma_indices_for_craftext: shape [num_craftext_achievements], int32 (or -1 if missing)
        valid_mask: shape [num_craftext_achievements], bool
    """
    craftext_items = {
        k: int(v)
        for k, v in vars(CrafTextAchievement).items()
        if not k.startswith("_") and isinstance(v, int)
    }
    if not craftext_items:
        raise ValueError("Failed to extract CrafText achievements for index mapping")

    num_craftext = max(craftext_items.values()) + 1
    ma_indices_for_craftext = [-1] * num_craftext

    for name, craftext_idx in craftext_items.items():
        ma_enum = MAAchievement.__members__.get(name)
        if ma_enum is not None:
            ma_indices_for_craftext[craftext_idx] = int(ma_enum.value)

    ma_indices_for_craftext = jnp.asarray(ma_indices_for_craftext, dtype=jnp.int32)
    valid_mask = ma_indices_for_craftext >= 0
    return ma_indices_for_craftext, valid_mask


class MultiAgentInstructionAdapter(InstructionWrapper):
    """
    Adapter that applies CrafText instruction logic to MA-Craftax env API:
    - reset(key) -> obs, state
    - step(key, state, action_dict) -> obs, state, reward_dict, done_dict, info
    """

    def __init__(
        self,
        env,
        config_name=None,
        scenario_handler_class=None,
        encode_model_class=None,
        encode_form=None,
    ) -> None:
        kwargs = {"config_name": config_name}
        if scenario_handler_class is not None:
            kwargs["scenario_handler_class"] = scenario_handler_class
        if encode_model_class is not None:
            kwargs["encode_model_class"] = encode_model_class
        if encode_form is not None:
            kwargs["encode_form"] = encode_form
        super().__init__(env, **kwargs)
        self.default_params = env.default_params
        self.num_agents = env.num_agents
        self.agents = env.agents
        instruction_shape = self.scenario_handler.scenario_data_jax.embeddings_list.shape[1:]
        self.instruction_dim = 1
        for size in instruction_shape:
            self.instruction_dim *= int(size)
        # MA-Craftax inventory does not expose classic fields (wood_pickaxe, etc.),
        # so always use GameData encoder that maps from current inventory schema.
        self.StateStructure = GameData
        # Map instruction variant index -> unique task index (scenario id).
        self.instruction_to_task_idx = jnp.asarray(
            self.scenario_handler.scenario_data.indices_list
        ).reshape(-1)
        # Human-readable task names by unique task index.
        self.task_names = list(self.scenario_handler.all_scenario.keys())
        # CrafText checkers use CrafText achievement indices; MA-Craftax uses shifted ids.
        # Remap achievements by name to avoid false instruction completions.
        (
            self._ma_achievement_indices_for_craftext,
            self._achievement_index_valid_mask,
        ) = _build_achievement_index_mapping()
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.agents
        }

    def observation_space(self, agent):
        """Return the flattened game observation plus the instruction embedding."""
        base_space = self.env.observation_space(agent)
        base_dim = 1
        for size in base_space.shape:
            base_dim *= int(size)
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(base_dim + self.instruction_dim,),
            dtype=jnp.float32,
        )

    def _add_instruction_to_obs(self, obs, instruction):
        instruction = lax.stop_gradient(
            jnp.asarray(instruction, dtype=jnp.float32).reshape(-1)
        )
        return {
            agent: jnp.concatenate(
                (jnp.asarray(obs[agent], dtype=jnp.float32).reshape(-1), instruction)
            )
            for agent in self.agents
        }

    def _normalize_state_for_checkers(self, player_state, achievements_len: int):
        """
        Normalize MA-Craftax state shapes to checker expectations:
        - player_position: (N, 2) -> (2,)
        - achievements: (N, A) -> (A,) and truncate to target mask length
        """
        player_position = player_state.variables.player_position
        if player_position.ndim > 1:
            player_position = player_position[0]

        achievements = player_state.achievements.achievements
        if achievements.ndim > 1:
            achievements = jnp.any(achievements, axis=0).astype(jnp.int32)
        else:
            achievements = achievements.astype(jnp.int32)

        # Remap MA achievement vector -> CrafText achievement index space.
        safe_ma_idx = jnp.clip(
            self._ma_achievement_indices_for_craftext,
            a_min=0,
            a_max=max(int(achievements.shape[0]) - 1, 0),
        )
        mapped_achievements = jnp.take(achievements, safe_ma_idx, mode="clip")
        achievements = jnp.where(self._achievement_index_valid_mask, mapped_achievements, 0)

        if int(achievements.shape[0]) < achievements_len:
            pad = jnp.zeros((achievements_len - int(achievements.shape[0]),), dtype=jnp.int32)
            achievements = jnp.concatenate([achievements, pad], axis=0)
        achievements = achievements[:achievements_len]

        variables = player_state.variables.replace(player_position=player_position)
        player_achievements = player_state.achievements.replace(achievements=achievements)
        return player_state.replace(variables=variables, achievements=player_achievements)

    def _to_scalar_bool(self, value):
        """Normalize checker outputs to scalar bool for JAX switch type compatibility."""
        arr = jnp.asarray(value)
        return jnp.any(arr).astype(jnp.bool_)

    def _generic_check_compatible(self, game_data_vector, target_state, checker_id):
        def ca(ts):
            return self._to_scalar_bool(checker_acvievments(game_data_vector, ts.achievements))

        def cp(ts):
            return self._to_scalar_bool(
                checker_conditional_placement(game_data_vector, ts.conditional_placing)
            )

        def port(ts):
            return self._to_scalar_bool(
                cheker_localization(game_data_vector, ts.Localization_placing)
            )

        def ilf(ts):
            return self._to_scalar_bool(checker_line(game_data_vector, ts.building_line))

        def isf(ts):
            return self._to_scalar_bool(checker_square(game_data_vector, ts.building_square))

        def icf(ts):
            return self._to_scalar_bool(checker_star(game_data_vector, ts.building_star))

        def atp(ts):
            return self._to_scalar_bool(
                checker_time_placement(game_data_vector, ts.time_placement)
            )

        # Keep same checker mapping as original generic_check.
        fns = (ca, cp, port, ilf, isf, icf, atp, ca)
        return lax.switch(checker_id, fns, target_state)

    def reset(self, rng: chex.PRNGKey, instruction_idx: int = -1):
        # Reuse CrafText reset logic entirely; just hide env_params from caller API.
        obs, state = super().reset(
            rng, self.default_params, instruction_idx=instruction_idx
        )
        return self._add_instruction_to_obs(obs, state.instruction), state

    def step(self, rng: chex.PRNGKey, env_state, action):
        rng, rng_reset = jax.random.split(rng)
        # step_env keeps the post-action state. MultiAgentEnv.step would already
        # have replaced it with a fresh world, so the checker would miss successes
        # that happen on a terminal transition.
        obs, next_base_state, reward, done, info = self.env.step_env(
            rng, env_state.env_state, action
        )

        # For checker compatibility, represent action dict as dense array.
        action_array = jnp.array([action[a] for a in self.agents], dtype=jnp.int32)
        game_data_vector = self.StateStructure.from_state(
            env_state.env_state, next_base_state, action_array
        )
        ts = self.batched_ts.select(env_state.idx)
        achievements_len = len(ts.achievements.achievement_mask)
        current_state = self._normalize_state_for_checkers(
            game_data_vector.states[0], achievements_len
        )
        previous_state = self._normalize_state_for_checkers(
            game_data_vector.states[1], achievements_len
        )
        game_data_vector = game_data_vector.replace(states=[current_state, previous_state])
        instruction_done = self._generic_check_compatible(
            game_data_vector, ts, env_state.checker_id
        )

        done_all = done["__all__"]
        is_explore = env_state.checker_id == Scenarios.EXPLORE
        reward = {
            k: lax.cond(is_explore, lambda r: r, lambda r: r / 50.0, v)
            + jnp.float32(instruction_done)
            for k, v in reward.items()
        }

        done_all = jnp.logical_or(done_all, instruction_done)
        new_episode_sr = env_state.success_rate + jnp.float32(instruction_done)
        continued_state = env_state.replace(
            env_state=next_base_state,
            timestep=next_base_state.timestep,
            success_rate=new_episode_sr * (1 - done_all),
            total_success_rate=env_state.total_success_rate * (1 - done_all)
            + new_episode_sr * done_all,
            instruction_done=instruction_done,
        )
        continued_obs = self._add_instruction_to_obs(obs, continued_state.instruction)

        info = dict(info)
        user_info = dict(info.get("user_info", {}))
        per_agent_count = self.num_agents

        # Log the episode that just acted, even when this transition resets.
        user_info["Instructions/SR"] = jnp.full(
            per_agent_count, continued_state.total_success_rate, dtype=jnp.float32
        )
        user_info["Instructions/instruction_done"] = jnp.full(
            per_agent_count, jnp.float32(instruction_done), dtype=jnp.float32
        )
        user_info["Instructions/instruction_idx"] = jnp.full(
            per_agent_count, jnp.float32(env_state.idx), dtype=jnp.float32
        )
        task_idx = jnp.take(
            self.instruction_to_task_idx, jnp.int32(env_state.idx), mode="clip"
        )
        user_info["Instructions/task_idx"] = jnp.full(
            per_agent_count, jnp.float32(task_idx), dtype=jnp.float32
        )

        info["user_info"] = user_info
        info.update(
            {
                "SR": continued_state.total_success_rate,
                "steps": continued_state.timestep,
            }
        )

        reset_obs, reset_state = self.reset(rng_reset)

        def _select_episode(reset_value, continued_value):
            return jnp.where(done_all, reset_value, continued_value)

        state = jax.tree.map(_select_episode, reset_state, continued_state)
        obs = jax.tree.map(_select_episode, reset_obs, continued_obs)
        done = {agent: done_all for agent in self.agents}
        done["__all__"] = done_all
        return obs, state, reward, done, info
