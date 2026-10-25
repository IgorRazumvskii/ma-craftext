import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import chex
from typing import Tuple

from actions import *
from trade import *

from craftax_coop.game_logic import (
    change_floor,
    shoot_projectile,
    boss_logic,
    level_up_attributes,
    move_player,
    update_mobs,
    spawn_mobs,
    update_plants,
    update_player_intrinsics,
    clip_inventory_and_intrinsics,
    calculate_inventory_achievements,
    drink_potion,
    read_book,
)
from craftax_coop.util.game_logic_utils import calculate_light_level
from craftax_coop.constants import (
    ACHIEVEMENT_REWARD_MAP,
    Action
)
from craftax_coop.craftax_state import EnvState, StaticEnvParams, EnvParams
from craftax_coop.checker import (
    check_one_table,
    check_two_tables_straight_line,
    check_two_tables_diagonal_line,
    check_two_furnaces_straight_line,
    check_two_furnaces_diagonal_line,
)
from craftax_coop.renderer.renderer_symbolic import render_craftax_symbolic
from craftax_coop.renderer.renderer_pixels import render_craftax_pixels


def craftax_step(
        rng: chex.PRNGKey, state: EnvState, actions: jnp.array, params: EnvParams, static_params: StaticEnvParams
    ) -> Tuple[EnvState, chex.Array]:
    init_achievements = state.achievements
    init_health = state.player_health
    init_inventory_wood = state.inventory.wood
    init_inventory_stone = state.inventory.stone
    init_inventory_pickaxe = state.inventory.pickaxe
    init_inventory_sword = state.inventory.sword
    
    # Save initial map state to track building creation
    init_map = state.map[state.player_level]
    init_has_one_table = check_one_table(state, static_params)
    init_has_two_tables_straight = check_two_tables_straight_line(state, static_params)

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

    # Change floor
    state = change_floor(state, actions, params, static_params)

    # Crafting
    state = do_crafting(state, actions, static_params)

    # Interact (mining, melee attacking, eating plants, drinking water, reviving)
    rng, _rng = jax.random.split(rng)
    state = do_action(_rng, state, actions, params, static_params)
    
    # Save inventory after mining (before trading) to track actual mining
    inventory_after_mining_wood = state.inventory.wood
    inventory_after_mining_stone = state.inventory.stone

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

    # Trade
    state, successful_trades = trade_materials(state, actions, static_params)

    # Request Materials
    state = make_request(state, actions)

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

    # Локальные индексы для паттернов в achievements (используем индексы, которые не конфликтуют с существующими)
    # Эти индексы используются только для отслеживания, не добавляются в constants.py
    PATTERN_TWO_TABLES_STRAIGHT_IDX = 68
    PATTERN_TWO_TABLES_DIAGONAL_IDX = 69
    PATTERN_TWO_FURNACES_STRAIGHT_IDX = 70
    PATTERN_TWO_FURNACES_DIAGONAL_IDX = 71
    
    # Проверяем, какие паттерны существуют сейчас (до обновления achievements)
    has_two_tables_straight_now = check_two_tables_straight_line(state, static_params)
    has_two_tables_diagonal_now = check_two_tables_diagonal_line(state, static_params)
    has_two_furnaces_straight_now = check_two_furnaces_straight_line(state, static_params)
    has_two_furnaces_diagonal_now = check_two_furnaces_diagonal_line(state, static_params)
    
    # Получаем размер массива achievements
    num_achievements = state.achievements.shape[1]
    
    # Безопасная функция для обновления achievement (JAX-совместимо)
    def update_achievement_safe(achievements, idx, pattern_exists):
        idx_safe = jnp.minimum(idx, num_achievements - 1)
        can_update = idx < num_achievements
        return jnp.where(
            can_update,
            achievements.at[:, idx_safe].set(
                jnp.logical_or(achievements[:, idx_safe], pattern_exists)
            ),
            achievements
        )

    # Inventory achievements
    state = calculate_inventory_achievements(state)
    
    # Обновляем achievements для паттернов (после calculate_inventory_achievements, чтобы не потерять изменения)
    # Используем текущие achievements из state и обновляем их для паттернов
    current_achievements = state.achievements
    current_achievements = update_achievement_safe(current_achievements, PATTERN_TWO_TABLES_STRAIGHT_IDX, has_two_tables_straight_now)
    current_achievements = update_achievement_safe(current_achievements, PATTERN_TWO_TABLES_DIAGONAL_IDX, has_two_tables_diagonal_now)
    current_achievements = update_achievement_safe(current_achievements, PATTERN_TWO_FURNACES_STRAIGHT_IDX, has_two_furnaces_straight_now)
    current_achievements = update_achievement_safe(current_achievements, PATTERN_TWO_FURNACES_DIAGONAL_IDX, has_two_furnaces_diagonal_now)
    state = state.replace(achievements=current_achievements)

    # Награды за достижения
    achievement_coefficients = ACHIEVEMENT_REWARD_MAP
    achievement_reward = (
        (state.achievements.astype(int) - init_achievements.astype(int))
        * achievement_coefficients
    ).sum(axis=1)

    # Награда за здоровье
    health_reward = jnp.maximum(0.0, (state.player_health - init_health) * 0.1)

    # Награда за успешный обмен
    trade_reward = successful_trades.astype(jnp.float32) * 0.1
    
    # Безопасная функция для получения achievement (если индекс выходит за границы, возвращаем False)
    def get_achievement_safe(achievements, idx):
        idx_safe = jnp.minimum(idx, num_achievements - 1)
        return jnp.where(idx < num_achievements, achievements[:, idx_safe].any(), False)
    
    # Безопасная функция для получения achievement (если индекс выходит за границы, возвращаем False)
    def get_achievement_safe(achievements, idx):
        idx_safe = jnp.minimum(idx, num_achievements - 1)
        return jnp.where(idx < num_achievements, achievements[:, idx_safe].any(), False)
    
    # Проверяем, были ли паттерны уже получены в этом эпизоде (через achievements)
    # Используем безопасный доступ к achievements (если индекс выходит за границы, считаем что achievement не получен)
    
    init_pattern_two_tables_straight = get_achievement_safe(init_achievements, PATTERN_TWO_TABLES_STRAIGHT_IDX)
    init_pattern_two_tables_diagonal = get_achievement_safe(init_achievements, PATTERN_TWO_TABLES_DIAGONAL_IDX)
    init_pattern_two_furnaces_straight = get_achievement_safe(init_achievements, PATTERN_TWO_FURNACES_STRAIGHT_IDX)
    init_pattern_two_furnaces_diagonal = get_achievement_safe(init_achievements, PATTERN_TWO_FURNACES_DIAGONAL_IDX)
    
    # Проверяем, был ли создан новый паттерн в этом шаге (награда только при первом создании)
    pattern_two_tables_straight_created = jnp.logical_and(
        jnp.logical_not(init_pattern_two_tables_straight),
        has_two_tables_straight_now
    ).astype(jnp.float32)
    pattern_two_tables_diagonal_created = jnp.logical_and(
        jnp.logical_not(init_pattern_two_tables_diagonal),
        has_two_tables_diagonal_now
    ).astype(jnp.float32)
    pattern_two_furnaces_straight_created = jnp.logical_and(
        jnp.logical_not(init_pattern_two_furnaces_straight),
        has_two_furnaces_straight_now
    ).astype(jnp.float32)
    pattern_two_furnaces_diagonal_created = jnp.logical_and(
        jnp.logical_not(init_pattern_two_furnaces_diagonal),
        has_two_furnaces_diagonal_now
    ).astype(jnp.float32)
    
    # Получаем ID выбранного паттерна из state.ff_damage_dealt
    # Нормализуем значение, чтобы получить целое число от 0 до 3
    target_pattern_id = jnp.clip(jnp.int32(jnp.floor(state.ff_damage_dealt)), 0, 3)
    
    # Награда дается ТОЛЬКО за выбранный паттерн
    # Создаем массив всех паттернов и выбираем нужный по индексу
    pattern_created_array = jnp.array([
        pattern_two_tables_straight_created,      # 0
        pattern_two_tables_diagonal_created,      # 1
        pattern_two_furnaces_straight_created,   # 2
        pattern_two_furnaces_diagonal_created     # 3
    ])
    
    # Выбираем награду только за выбранный паттерн
    # Используем jnp.take для безопасного индексирования
    selected_pattern_created = jnp.take(pattern_created_array, target_pattern_id, mode='clip')
    pattern_reward = selected_pattern_created * 1.0
    
    # Награда дается всем игрокам на текущем уровне
    building_reward = jnp.full(static_params.player_count, pattern_reward)
    
    # Награда за добычу ресурсов (только за реальную добычу, не за обмен)
    is_miner = state.player_specialization == Specialization.MINER.value
    # Проверяем изменение инвентаря после добычи, но до обмена
    # wood_mined = jnp.maximum(0, inventory_after_mining_wood - init_inventory_wood)
    # stone_mined = jnp.maximum(0, inventory_after_mining_stone - init_inventory_stone)
    # miner_action_reward = (wood_mined + stone_mined).astype(jnp.float32) * 0.5 * is_miner
    #
    # role_action_reward = miner_action_reward
    
    # Награда за движение
    position_changed = (state.player_position != init_position).any(axis=1)
    movement_reward = position_changed.astype(jnp.float32) * 0.01
    # individual_reward = achievement_reward + health_reward + trade_reward + building_reward + role_action_reward + movement_reward
    individual_reward = achievement_reward + health_reward + trade_reward + building_reward + movement_reward

    # Суммируем награды для всех игроков
    shared_reward = individual_reward.sum().repeat(static_params.player_count)

    reward = jax.lax.select(
        params.shared_reward,
        shared_reward,
        individual_reward
    )

    player_alive = state.player_health > 0.5

    rng, _rng = jax.random.split(rng)
    
    # Восстанавливаем паттерн: нормализуем ff_damage_dealt обратно в диапазон [0, 3]
    # Это защищает от накопления изменений из-за friendly fire damage
    # Паттерн должен быть целым числом от 0 до 3, поэтому используем floor и clip
    pattern_id = jnp.clip(jnp.int32(jnp.floor(state.ff_damage_dealt)), 0, 3)
    normalized_pattern = jnp.float32(pattern_id)

    state = state.replace(
        player_alive=player_alive,
        timestep=state.timestep + 1,
        light_level=calculate_light_level(state.timestep + 1, params),
        state_rng=_rng,
        ff_damage_dealt=normalized_pattern,  # Восстанавливаем нормализованное значение паттерна
    )

    return state, reward

