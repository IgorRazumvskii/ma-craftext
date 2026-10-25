import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import chex
from typing import Tuple

from craftax_together_talens.game_logic.actions import *
from craftax_together_talens.game_logic.trade import *
from craftax_together_talens.game_logic.constants import get_allowed_action_flags

from craftax_together_talens.game_logic.core import (
    change_floor,
    shoot_projectile,
    boss_logic,
    level_up_attributes,
    move_player,
    update_mobs,
    spawn_mobs,
    update_plants,
    update_player_intrinsics,
    calculate_inventory_achievements,
    drink_potion,
    read_book,
)
from craftax_together_talens.util.game_logic_utils import clip_inventory_and_intrinsics
from craftax_together_talens.util.game_logic_utils import calculate_light_level
from craftax_together_talens.constants import (
    ACHIEVEMENT_REWARD_MAP,
    Action
)
from craftax_together_talens.craftax_state import EnvState, StaticEnvParams, EnvParams


def craftax_step(
        rng: chex.PRNGKey, state: EnvState, actions: jnp.array, params: EnvParams, static_params: StaticEnvParams
    ) -> Tuple[EnvState, chex.Array]:
    
    init_achievements = state.achievements
    init_health = state.player_health
    
    # Pattern-based tracking/reward is disabled (instruction-only training).

    # Interrupt action if dead, sleeping or resting
    cant_do_action = jnp.logical_or(
        jnp.logical_not(state.player_alive),
        jnp.logical_or(state.is_sleeping, state.is_resting),
    )
    actions = jnp.where(
        cant_do_action,
        Action.NOOP.value,
        actions
    )

    # Enforce role-based action permissions at runtime:
    # disallowed actions are converted to NOOP even if a policy outputs them.
    role_allowed_action = get_allowed_action_flags(actions, state.player_specialization)
    actions = jnp.where(role_allowed_action, actions, Action.NOOP.value)

    # Change floor
    state = change_floor(state, actions, params, static_params)

    # Crafting
    state = do_crafting(state, actions, static_params)

    # Interact (mining, melee attacking, eating plants, drinking water, reviving)
    rng, _rng = jax.random.split(rng)
    state = do_action(_rng, state, actions, params, static_params)
    
    # Placing
    state = place_block(state, actions, static_params)

    # Shooting
    state = shoot_projectile(state, actions, static_params)

    # Casting
    state = cast_spell(state, actions, static_params)

    # Potions
    state = drink_potion(state, actions)

    # Read
    state = read_book(state, actions)

    # Enchant
    rng, _rng = jax.random.split(rng)
    state = enchant(_rng, state, actions, static_params)

    # Boss
    state = boss_logic(state, static_params)

    # Attributes
    state = level_up_attributes(state, actions, params)

    # Request Materials
    state = make_request(state, actions)

    # Trade
    state, successful_trades = trade_materials(state, actions, static_params)

    # Save initial position before movement
    init_position = state.player_position

    # Movement
    state = move_player(state, actions, params, static_params)

    # Mobs
    rng, _rng = jax.random.split(rng)
    state = update_mobs(_rng, state, params, params, static_params)

    rng, _rng = jax.random.split(rng)
    state = spawn_mobs(state, _rng, params, static_params)

    # Plants
    state = update_plants(state, static_params)

    # Intrinsics
    state = update_player_intrinsics(state, actions, static_params)

    # Cap inv
    state = clip_inventory_and_intrinsics(state, params)

    # Inventory achievements
    state = calculate_inventory_achievements(state)

    # Base reward from environment (CrafText-style shaping is applied in adapter).
    achievement_coefficients = ACHIEVEMENT_REWARD_MAP
    achievement_reward = (
        (state.achievements.astype(int) - init_achievements.astype(int))
        * achievement_coefficients
    ).sum(axis=1)
    health_reward = jnp.maximum(0.0, (state.player_health - init_health) * 0.1)
    trade_reward = successful_trades.astype(jnp.float32) * 0.1
    movement_reward = (state.player_position != init_position).any(axis=1).astype(jnp.float32) * 0.01
    individual_reward = achievement_reward + health_reward + trade_reward + movement_reward

    # Суммируем награды для всех игроков
    shared_reward = individual_reward.sum().repeat(static_params.player_count)

    reward = jax.lax.select(
        params.shared_reward,
        shared_reward,
        individual_reward
    )

    player_alive = state.player_health > 0.5

    rng, _rng = jax.random.split(rng)
    
    state = state.replace(
        player_alive=player_alive,
        timestep=state.timestep + 1,
        light_level=calculate_light_level(state.timestep + 1, params),
        state_rng=_rng,
    )

    return state, reward

