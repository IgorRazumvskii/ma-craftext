# ВАЖНО: Добавляем путь в sys.path ПЕРЕД всеми импортами!
# Используем тот же подход, что и в wrapper_game_logic.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
from jax import lax
from enum import Enum
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
import functools

# Импортируем все необходимые функции и константы
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
    interplayer_interaction,
    update_plants_with_eat,
    add_new_growing_plant,
)
from craftax_coop.util.game_logic_utils import (
    is_near_block,
    has_beaten_boss,
    attack_mob,
    get_player_damage_vector,
    get_max_drink,
    get_max_food,
    get_max_health,
    is_boss_vulnerable,
    is_fighting_boss,
    spawn_projectile,
    calculate_light_level,
    in_bounds,
    is_in_other_player,
    is_in_mob,
    is_in_solid_block,
)
from craftax_coop.constants import (
    Achievement,
    DIRECTIONS,
    ACHIEVEMENT_REWARD_MAP,
    BOSS_FIGHT_SPAWN_TURNS,
    ProjectileType,
    BlockType,
    Action,
    ItemType,
    CAN_PLACE_ITEM_MAPPING,
    TORCH_LIGHT_MAP,
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

# Локальные константы для запроса инструментов
# Используем значения после GIVE действий, чтобы они входили в action space
# Для 3 агентов: action space = len(Action) + (num_agents - 2) = 53 + 1 = 54 (действия 0-53)
# GIVE = 52, GIVE_TO_PLAYER_1 = 53, поэтому используем 54-59 для запросов инструментов
# Но нужно расширить action space или использовать значения, которые уже есть в Action enum
# ВАЖНО: Эти значения должны быть доступны в action space!
# Используем значения после всех GIVE действий для каждого количества агентов
# Для 3 агентов: действия 52 (GIVE), 53 (GIVE_TO_PLAYER_1) = 2 действия
# Action space = 53 + 1 = 54, значит доступны действия 0-53
# Используем значения 54-59, но нужно расширить action space
# ВРЕМЕННОЕ РЕШЕНИЕ: используем неиспользуемые значения Action enum (27 отсутствует, или переиспользуем REQUEST_*)
# Лучше использовать значения после GIVE, но расширить action space
REQUEST_PICKAXE = 54  # После GIVE действий (52, 53 для 3 агентов)
REQUEST_SWORD = 55
REQUEST_ARMOUR = 56
REQUEST_BOW = 57
REQUEST_ARROWS = 58
REQUEST_TORCHES = 59

# Переопределяем Specialization для этого модуля
class Specialization(Enum):
    UNASSIGNED = 0
    MINER = 1  # Добывает камни и деревья
    BUILDER = 2  # Строит столы и печки
    CRAFTER = 3  # Отвечает за крафт

def add_items_from_chest(rng, state, inventory, is_opening_chest):
    is_miner = state.player_specialization == Specialization.MINER.value
    is_warrior = state.player_specialization == Specialization.BUILDER.value

    # Wood (60%)
    rng, _rng = jax.random.split(rng)
    is_looting_wood = jax.random.uniform(_rng) < 0.6 * is_opening_chest *is_miner
    rng, _rng = jax.random.split(rng)
    wood_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=6) * is_looting_wood
    )

    # Torch (60%)
    rng, _rng = jax.random.split(rng)
    collect_prob = 0.6 * is_miner
    is_looting_torch = jax.random.uniform(_rng) < collect_prob * is_opening_chest
    rng, _rng = jax.random.split(rng)
    torch_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=4, maxval=8) * is_looting_torch
    )

    # Ores (60%)
    rng, _rng = jax.random.split(rng)
    is_looting_ore = jax.random.uniform(_rng) < collect_prob * is_opening_chest
    rng, _rng = jax.random.split(rng)
    ore_loot_id = jax.random.choice(
        _rng,
        jnp.arange(5, dtype=jnp.int32),
        shape=(),
        p=jnp.array([0.3, 0.3, 0.15, 0.125, 0.125]),
    )
    rng, _rng = jax.random.split(rng)

    # Use the same rng as events are mutually exclusive
    coal_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=4)
        * (ore_loot_id == 0)
        * is_looting_ore
    )
    iron_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=3)
        * (ore_loot_id == 1)
        * is_looting_ore
    )
    diamond_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=2)
        * (ore_loot_id == 2)
        * is_looting_ore
    )
    sapphire_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=2)
        * (ore_loot_id == 3)
        * is_looting_ore
    )
    ruby_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=1, maxval=2)
        * (ore_loot_id == 4)
        * is_looting_ore
    )

    # Potion (50%)
    rng, _rng = jax.random.split(rng)
    is_looting_potion = jax.random.uniform(_rng) < 0.5 * is_opening_chest
    rng, _rng = jax.random.split(rng)
    potion_loot_index = jax.random.randint(_rng, shape=(), minval=0, maxval=6)
    rng, _rng = jax.random.split(rng)
    potion_loot_amount = jax.random.randint(_rng, shape=(), minval=1, maxval=3)

    # Arrows (50%)
    rng, _rng = jax.random.split(rng)
    is_looting_arrows = jax.random.uniform(_rng) < 0.5 * is_opening_chest * is_warrior
    rng, _rng = jax.random.split(rng)
    arrows_loot_amount = (
        jax.random.randint(_rng, shape=(), minval=4, maxval=9) * is_looting_arrows
    )

    # Tools (20%)
    rng, _rng = jax.random.split(rng)
    is_looting_tool = jax.random.uniform(_rng) < 0.2
    rng, _rng = jax.random.split(rng)
    tool_id = jax.random.randint(_rng, shape=(), minval=0, maxval=2)

    is_looting_pickaxe = jnp.logical_and(
        jnp.logical_and(
            is_miner,
            jnp.logical_and(is_looting_tool, tool_id == 0)
        ),
        is_opening_chest
    )
    rng, _rng = jax.random.split(rng)
    pickaxe_loot_level = (
        jax.random.choice(
            _rng,
            (jnp.arange(4) + 1).astype(int),
            shape=(),
            p=jnp.array([0.4, 0.3, 0.2, 0.1]),
        )
        * is_looting_pickaxe
    )
    pickaxe_loot_level = jnp.maximum(pickaxe_loot_level, inventory.pickaxe)
    new_pickaxe_level = (
        is_looting_pickaxe * pickaxe_loot_level
        + (1 - is_looting_pickaxe) * inventory.pickaxe
    )

    # Special chests
    is_looting_bow = jnp.logical_and(
        jnp.logical_and(
            is_opening_chest,
            is_warrior,
        ),
        jnp.logical_and(
            state.player_level == 1,
            jnp.logical_not(state.chests_opened[state.player_level]),
        )
    )
    new_bow_level = is_looting_bow * 1 + (1 - is_looting_bow) * inventory.bow

    can_loot_book = jnp.logical_and(
        jnp.logical_not(state.chests_opened[state.player_level]),
        jnp.logical_or(state.player_level == 3, state.player_level == 4),
    )
    is_looting_book = jnp.logical_and(
        can_loot_book,
        is_opening_chest
    )

    # Update inventory
    return inventory.replace(
        wood=inventory.wood + wood_loot_amount*is_miner,
        torches=inventory.torches + torch_loot_amount,
        coal=inventory.coal + coal_loot_amount*is_miner,
        iron=inventory.iron + iron_loot_amount*is_miner,
        diamond=inventory.diamond + diamond_loot_amount*is_miner,
        sapphire=inventory.sapphire + sapphire_loot_amount*is_miner,
        ruby=inventory.ruby + ruby_loot_amount*is_miner,
        arrows=inventory.arrows + arrows_loot_amount,
        pickaxe=new_pickaxe_level,
        # sword=new_sword_level,
        potions=inventory.potions.at[:, potion_loot_index].set(
            inventory.potions[:, potion_loot_index]
            + potion_loot_amount * is_looting_potion * is_opening_chest
        ),
        bow=new_bow_level,
        books=inventory.books + 1 * is_looting_book,
    )


def do_crafting(state, actions, static_params):
    is_at_crafting_table = is_near_block(state, BlockType.CRAFTING_TABLE.value, static_params)
    is_at_furnace = is_near_block(state, BlockType.FURNACE.value, static_params)
    is_crafter = state.player_specialization == Specialization.CRAFTER.value

    new_achievements = state.achievements

    # Wood pickaxe
    can_craft_wood_pickaxe = jnp.logical_and(
        state.inventory.wood >= 1,
        is_crafter
    )

    is_crafting_wood_pickaxe = jnp.logical_and(
        actions == Action.MAKE_WOOD_PICKAXE.value,
        jnp.logical_and(
            can_craft_wood_pickaxe,
            jnp.logical_and(is_at_crafting_table, state.inventory.pickaxe < 1),
        ),
    )

    new_inventory = state.inventory.replace(
        wood=state.inventory.wood - 1 * is_crafting_wood_pickaxe,
        pickaxe=state.inventory.pickaxe * (1 - is_crafting_wood_pickaxe)
        + 1 * is_crafting_wood_pickaxe,
    )

    # Stone pickaxe
    can_craft_stone_pickaxe = jnp.logical_and(
        is_crafter,
        jnp.logical_and(
            new_inventory.wood >= 1, new_inventory.stone >= 1
        )
    )
    is_crafting_stone_pickaxe = jnp.logical_and(
        actions == Action.MAKE_STONE_PICKAXE.value,
        jnp.logical_and(
            can_craft_stone_pickaxe,
            jnp.logical_and(is_at_crafting_table, new_inventory.pickaxe < 2),
        ),
    )

    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_crafting_stone_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_stone_pickaxe,
        pickaxe=new_inventory.pickaxe * (1 - is_crafting_stone_pickaxe)
        + 2 * is_crafting_stone_pickaxe,
    )

    # Iron pickaxe
    can_craft_iron_pickaxe = jnp.logical_and(
        new_inventory.wood >= 1,
        jnp.logical_and(
            new_inventory.stone >= 1,
            jnp.logical_and(
                new_inventory.iron >= 1,
                new_inventory.coal >= 1,
            ),
        ),
    )
    can_craft_iron_pickaxe = jnp.logical_and(
        is_crafter,
        can_craft_iron_pickaxe,
    )
    is_crafting_iron_pickaxe = jnp.logical_and(
        actions == Action.MAKE_IRON_PICKAXE.value,
        jnp.logical_and(
            can_craft_iron_pickaxe,
            jnp.logical_and(
                is_at_furnace,
                jnp.logical_and(is_at_crafting_table, new_inventory.pickaxe < 3),
            ),
        ),
    )

    new_inventory = new_inventory.replace(
        iron=new_inventory.iron - 1 * is_crafting_iron_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_iron_pickaxe,
        stone=new_inventory.stone - 1 * is_crafting_iron_pickaxe,
        coal=new_inventory.coal - 1 * is_crafting_iron_pickaxe,
        pickaxe=new_inventory.pickaxe * (1 - is_crafting_iron_pickaxe)
        + 3 * is_crafting_iron_pickaxe,
    )

    # Diamond pickaxe
    can_craft_diamond_pickaxe = jnp.logical_and(
        new_inventory.wood >= 1, new_inventory.diamond >= 3
    )
    can_craft_diamond_pickaxe = jnp.logical_and(
        is_crafter,
        can_craft_diamond_pickaxe,
    )
    is_crafting_diamond_pickaxe = jnp.logical_and(
        actions == Action.MAKE_DIAMOND_PICKAXE.value,
        jnp.logical_and(
            can_craft_diamond_pickaxe,
            jnp.logical_and(is_at_crafting_table, new_inventory.pickaxe < 4),
        ),
    )

    new_inventory = new_inventory.replace(
        diamond=new_inventory.diamond - 3 * is_crafting_diamond_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_diamond_pickaxe,
        pickaxe=new_inventory.pickaxe * (1 - is_crafting_diamond_pickaxe)
        + 4 * is_crafting_diamond_pickaxe,
    )

    # Wood sword
    can_craft_wood_sword = jnp.logical_and(new_inventory.wood >= 1, is_crafter)
    is_crafting_wood_sword = jnp.logical_and(
        actions == Action.MAKE_WOOD_SWORD.value,
        jnp.logical_and(
            can_craft_wood_sword,
            jnp.logical_and(is_at_crafting_table, new_inventory.sword < 1),
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_wood_sword,
        sword=new_inventory.sword * (1 - is_crafting_wood_sword)
        + 1 * is_crafting_wood_sword,
    )

    # Stone sword
    can_craft_stone_sword = jnp.logical_and(
        new_inventory.stone >= 1, new_inventory.wood >= 1
    )
    can_craft_stone_sword = jnp.logical_and(
        can_craft_stone_sword, 
        is_crafter
    )
    is_crafting_stone_sword = jnp.logical_and(
        actions == Action.MAKE_STONE_SWORD.value,
        jnp.logical_and(
            can_craft_stone_sword,
            jnp.logical_and(is_at_crafting_table, new_inventory.sword < 2),
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_stone_sword,
        stone=new_inventory.stone - 1 * is_crafting_stone_sword,
        sword=new_inventory.sword * (1 - is_crafting_stone_sword)
        + 2 * is_crafting_stone_sword,
    )

    # Iron sword
    can_craft_iron_sword = jnp.logical_and(
        new_inventory.iron >= 1,
        jnp.logical_and(
            new_inventory.wood >= 1,
            jnp.logical_and(new_inventory.stone >= 1, new_inventory.coal >= 1),
        ),
    )
    can_craft_iron_sword = jnp.logical_and(
        is_crafter,
        can_craft_iron_sword,
    )
    is_crafting_iron_sword = jnp.logical_and(
        actions == Action.MAKE_IRON_SWORD.value,
        jnp.logical_and(
            can_craft_iron_sword,
            jnp.logical_and(
                is_at_furnace,
                jnp.logical_and(is_at_crafting_table, new_inventory.sword < 3),
            ),
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_iron_sword,
        iron=new_inventory.iron - 1 * is_crafting_iron_sword,
        stone=new_inventory.stone - 1 * is_crafting_iron_sword,
        coal=new_inventory.coal - 1 * is_crafting_iron_sword,
        sword=new_inventory.sword * (1 - is_crafting_iron_sword)
        + 3 * is_crafting_iron_sword,
    )

    # Diamond sword
    can_craft_diamond_sword = jnp.logical_and(
        new_inventory.diamond >= 2, new_inventory.wood >= 1
    )
    can_craft_diamond_sword = jnp.logical_and(
        is_crafter,
        can_craft_diamond_sword,
    )
    is_crafting_diamond_sword = jnp.logical_and(
        actions == Action.MAKE_DIAMOND_SWORD.value,
        jnp.logical_and(
            can_craft_diamond_sword,
            jnp.logical_and(is_at_crafting_table, new_inventory.sword < 4),
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_diamond_sword,
        diamond=new_inventory.diamond - 2 * is_crafting_diamond_sword,
        sword=new_inventory.sword * (1 - is_crafting_diamond_sword)
        + 4 * is_crafting_diamond_sword,
    )

    # Iron armour
    can_craft_iron_armour = jnp.logical_and((new_inventory.armour < 1).sum(axis=1) > 0, is_crafter)
    can_craft_iron_armour = jnp.logical_and(
        can_craft_iron_armour,
        jnp.logical_and(new_inventory.iron >= 3, new_inventory.coal >= 3),
    )

    iron_armour_index_to_craft = jnp.argmax(new_inventory.armour < 1, axis=1)

    is_crafting_iron_armour = jnp.logical_and(
        actions == Action.MAKE_IRON_ARMOUR.value,
        jnp.logical_and(
            can_craft_iron_armour,
            jnp.logical_and(is_at_crafting_table, is_at_furnace),
        ),
    )

    new_inventory = new_inventory.replace(
        iron=new_inventory.iron - 3 * is_crafting_iron_armour,
        coal=new_inventory.coal - 3 * is_crafting_iron_armour,
        armour=new_inventory.armour.at[
            jnp.arange(0, len(new_inventory.armour)),
            iron_armour_index_to_craft
        ].set(
            is_crafting_iron_armour * 1
            + (1 - is_crafting_iron_armour)
            * new_inventory.armour[
                jnp.arange(0, len(new_inventory.armour)),
                iron_armour_index_to_craft
            ]
        ),
    )
    new_achievements = new_achievements.at[:, Achievement.MAKE_IRON_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.MAKE_IRON_ARMOUR.value],
            is_crafting_iron_armour,
        )
    )

    # Diamond armour
    can_craft_diamond_armour = jnp.logical_and((new_inventory.armour < 2).sum(axis=1) > 0, is_crafter)
    can_craft_diamond_armour = jnp.logical_and(
        can_craft_diamond_armour, new_inventory.diamond >= 3
    )

    diamond_armour_index_to_craft = jnp.argmax(new_inventory.armour < 2, axis=1)

    is_crafting_diamond_armour = jnp.logical_and(
        actions == Action.MAKE_DIAMOND_ARMOUR.value,
        jnp.logical_and(
            can_craft_diamond_armour,
            is_at_crafting_table,
        ),
    )

    new_inventory = new_inventory.replace(
        diamond=new_inventory.diamond - 3 * is_crafting_diamond_armour,
        armour=new_inventory.armour.at[
            jnp.arange(0, len(new_inventory.armour)),
            diamond_armour_index_to_craft
        ].set(
            is_crafting_diamond_armour * 2
            + (1 - is_crafting_diamond_armour)
            * new_inventory.armour[
                jnp.arange(0, len(new_inventory.armour)),
                diamond_armour_index_to_craft
            ]
        ),
    )
    new_achievements = new_achievements.at[:, Achievement.MAKE_DIAMOND_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.MAKE_DIAMOND_ARMOUR.value],
            is_crafting_diamond_armour,
        )
    )

    # Arrow
    can_craft_arrow = jnp.logical_and(new_inventory.stone >= 1, new_inventory.wood >= 1)
    can_craft_arrow = jnp.logical_and(
        can_craft_arrow,
        is_crafter
    )
    is_crafting_arrow = jnp.logical_and(
        actions == Action.MAKE_ARROW.value,
        jnp.logical_and(
            can_craft_arrow,
            jnp.logical_and(is_at_crafting_table, new_inventory.arrows < 99),
        ),
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_arrow,
        stone=new_inventory.stone - 1 * is_crafting_arrow,
        arrows=new_inventory.arrows + 2 * is_crafting_arrow,
    )

    # Torch
    can_craft_torch = jnp.logical_and(new_inventory.coal >= 1, new_inventory.wood >= 1)
    can_craft_torch = jnp.logical_and(
        can_craft_torch,
        is_crafter,
    )
    is_crafting_torch = jnp.logical_and(
        actions == Action.MAKE_TORCH.value,
        jnp.logical_and(
            can_craft_torch,
            jnp.logical_and(is_at_crafting_table, new_inventory.torches < 99),
        ),
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_torch,
        coal=new_inventory.coal - 1 * is_crafting_torch,
        torches=new_inventory.torches + 4 * is_crafting_torch,
    )

    state = state.replace(
        inventory=new_inventory,
        achievements=new_achievements,
    )

    return state


def do_action(rng, state, action, env_params, static_params):
    is_miner = state.player_specialization == Specialization.MINER.value
    is_warrior = state.player_specialization == Specialization.BUILDER.value
    is_crafter = state.player_specialization == Specialization.CRAFTER.value
    is_forager = state.player_specialization == Specialization.MINER.value  # MINER собирает саженцы и пьет воду

    block_position = state.player_position + DIRECTIONS[state.player_direction]
    equal_block_placement = (jnp.expand_dims(block_position, axis=1) == jnp.expand_dims(block_position, axis=0)).all(axis=2)

    doing_action = jnp.logical_and(
        in_bounds(block_position, static_params),
        action == Action.DO.value
    )

    state, did_attack_mob, did_kill_mob = attack_mob(
        state, doing_action, block_position, get_player_damage_vector(state), is_forager
    )
    
    # Interact with other players (Damage/Revive)
    # Сохраняем паттерн перед вызовом interplayer_interaction, так как она изменяет ff_damage_dealt
    saved_pattern = state.ff_damage_dealt
    state = interplayer_interaction(state, block_position, doing_action, env_params, static_params)
    # Восстанавливаем паттерн после взаимодействия
    state = state.replace(ff_damage_dealt=saved_pattern)
    
    # BLOCKS
    # Tree - только MINER может добывать дерево
    can_mine_tree = is_miner
    is_block_tree = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.TREE.value
    )
    is_block_fire_tree = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.FIRE_TREE.value
    )
    is_block_ice_shrub = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.ICE_SHRUB.value
    )

    is_block_tree_type = jnp.logical_or(
        is_block_tree, jnp.logical_or(is_block_fire_tree, is_block_ice_shrub)
    )
    is_mining_tree = jnp.logical_and(
        jnp.logical_and(
            is_block_tree_type,
            can_mine_tree,
        ),
        doing_action
    )
    tree_replacement_block = (
        is_block_tree * BlockType.GRASS.value
        + is_block_fire_tree * BlockType.FIRE_GRASS.value
        + is_block_ice_shrub * BlockType.ICE_GRASS.value
    )
    is_any_player_mining_tree = jnp.logical_and(
        equal_block_placement,
        is_mining_tree[:, None]
    ).any(axis=0)
    mined_tree_block = jnp.where(
        is_any_player_mining_tree,
        tree_replacement_block,
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]],
    )
    new_map = (
        state.map[state.player_level]
        .at[block_position[:, 0], block_position[:, 1]]
        .set(mined_tree_block)
    )
    new_inventory = state.inventory.replace(
        wood=state.inventory.wood + 1 * is_mining_tree
    )

    # Stone - только MINER может добывать камень
    can_mine_stone = jnp.logical_and(
        state.inventory.pickaxe >= 1,
        is_miner
    )
    is_block_stone = (
        state.map[state.player_level][block_position[:, 0], block_position[:, 1]]
        == BlockType.STONE.value
    )
    is_mining_stone = jnp.logical_and(
        jnp.logical_and(
            is_block_stone,
            can_mine_stone
        ),
        doing_action,
    )
    is_any_player_mining_stone = jnp.logical_and(
        equal_block_placement,
        is_mining_stone[:, None]
    ).any(axis=0)
    mined_stone_block = jnp.where(
        is_any_player_mining_stone,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_stone_block)
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone + 1 * is_mining_stone
    )

    # Furnace
    is_block_furnace = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.FURNACE.value
    )
    is_mining_furnace = jnp.logical_and(
        is_block_furnace,
        doing_action,
    )
    is_any_player_mining_furnace = jnp.logical_and(
        equal_block_placement,
        is_mining_furnace[:, None]
    ).any(axis=0)
    mined_furnace_block = jnp.where(
        is_any_player_mining_furnace,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_furnace_block)

    # Crafting Bench
    is_block_crafting_table = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.CRAFTING_TABLE.value
    )
    is_mining_crafting_table = jnp.logical_and(
        is_block_crafting_table,
        doing_action,
    )
    is_any_player_mining_crafting_table = jnp.logical_and(
        equal_block_placement,
        is_mining_crafting_table[:, None]
    ).any(axis=0)
    mined_crafting_table_block = jnp.where(
        is_any_player_mining_crafting_table,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(
        mined_crafting_table_block
    )

    # Coal
    can_mine_coal = jnp.logical_and(state.inventory.pickaxe >= 1, is_miner)
    is_block_coal = (
        state.map[state.player_level][block_position[:, 0], block_position[:, 1]]
        == BlockType.COAL.value
    )
    is_mining_coal = jnp.logical_and(
        jnp.logical_and(
            is_block_coal,
            can_mine_coal
        ),
        doing_action,
    )
    is_any_player_mining_coal = jnp.logical_and(
        equal_block_placement,
        is_mining_coal[:, None]
    ).any(axis=0)
    mined_coal_block = jnp.where(
        is_any_player_mining_coal,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_coal_block)
    new_inventory = new_inventory.replace(
        coal=new_inventory.coal + 1 * is_mining_coal
    )

    # Iron
    can_mine_iron = jnp.logical_and(state.inventory.pickaxe >= 2, is_miner)
    is_block_iron = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.IRON.value
    )
    is_mining_iron = jnp.logical_and(
        jnp.logical_and(
            is_block_iron,
            can_mine_iron
        ),
        doing_action,
    )
    is_any_player_mining_iron = jnp.logical_and(
        equal_block_placement,
        is_mining_iron[:, None]
    ).any(axis=0)
    mined_iron_block = jnp.where(
        is_any_player_mining_iron,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_iron_block)
    new_inventory = new_inventory.replace(
        iron=new_inventory.iron + 1 * is_mining_iron
    )

    # Diamond  
    can_mine_diamond = jnp.logical_and(state.inventory.pickaxe >= 3, is_miner)
    is_block_diamond = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.DIAMOND.value
    )
    is_mining_diamond = jnp.logical_and(
        jnp.logical_and(
            is_block_diamond,
            can_mine_diamond
        ),
        doing_action,
    )
    is_any_player_mining_diamond = jnp.logical_and(
        equal_block_placement,
        is_mining_diamond[:, None]
    ).any(axis=0)
    mined_diamond_block = jnp.where(
        is_any_player_mining_diamond,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_diamond_block)
    new_inventory = new_inventory.replace(
        diamond=new_inventory.diamond + 1 * is_mining_diamond
    )

    # Sapphire
    can_mine_sapphire = jnp.logical_and(state.inventory.pickaxe >= 4, is_miner)
    is_block_sapphire = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.SAPPHIRE.value
    )
    is_mining_sapphire = jnp.logical_and(
        jnp.logical_and(
            is_block_sapphire,
            can_mine_sapphire
        ),
        doing_action,
    )
    is_any_player_mining_sapphire = jnp.logical_and(
        equal_block_placement,
        is_mining_sapphire[:, None]
    ).any(axis=0)
    mined_sapphire_block = jnp.where(
        is_any_player_mining_sapphire,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_sapphire_block)
    new_inventory = new_inventory.replace(
        sapphire=new_inventory.sapphire + 1 * is_mining_sapphire
    )

    # Ruby
    can_mine_ruby = jnp.logical_and(state.inventory.pickaxe >= 4, is_miner)
    is_block_ruby = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.RUBY.value
    )
    is_mining_ruby = jnp.logical_and(
        jnp.logical_and(
            is_block_ruby,
            can_mine_ruby
        ),
        doing_action,
    )
    is_any_player_mining_ruby = jnp.logical_and(
        equal_block_placement,
        is_mining_ruby[:, None]
    ).any(axis=0)
    mined_ruby_block = jnp.where(
        is_any_player_mining_ruby,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_ruby_block)
    new_inventory = new_inventory.replace(
        ruby=new_inventory.ruby + 1 * is_mining_ruby
    )

    # Sapling
    rng, _rng = jax.random.split(rng)
    is_block_grass = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.GRASS.value
    )
    sapling_prob = 0.2 * is_forager # only foragers can collect saplings
    is_mining_sapling = jnp.logical_and(
        jnp.logical_and(
            is_block_grass,
            jax.random.uniform(_rng, (static_params.player_count,)) < sapling_prob,
        ),
        doing_action,
    )

    new_inventory = new_inventory.replace(
        sapling=new_inventory.sapling + 1 * is_mining_sapling
    )

    # Water
    is_block_water = jnp.logical_or(
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.WATER.value,
        state.map[state.player_level][block_position[:, 0], block_position[:, 1]]
        == BlockType.FOUNTAIN.value,
    )
    is_drinking_water = jnp.logical_and(
        is_block_water,
        doing_action,
    )
    new_drink = jnp.where(
        is_drinking_water,
        jnp.minimum(get_max_drink(state), state.player_drink + 4),
        state.player_drink,
    )
    new_thirst = jnp.where(
        is_drinking_water, 
        0.0, 
        state.player_thirst
    )
    new_achievements = state.achievements.at[:, Achievement.COLLECT_DRINK.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.COLLECT_DRINK.value], is_drinking_water
        )
    )

    # Plant
    is_block_plant = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.RIPE_PLANT.value
    )
    is_eating_plant = jnp.logical_and(
        is_block_plant,
        doing_action
    )
    is_any_player_eating_plant = jnp.logical_and(
        equal_block_placement,
        is_eating_plant[:, None]
    ).any(axis=0)
    new_plant = jnp.where(
        is_any_player_eating_plant,
        BlockType.PLANT.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(new_plant)
    new_food = jnp.where(
        is_eating_plant,
        jnp.minimum(get_max_food(state), state.player_food + 4),
        state.player_food,
    )
    new_hunger = jnp.where(is_eating_plant, 0.0, state.player_hunger)
    new_achievements = new_achievements.at[:, Achievement.EAT_PLANT.value].set(
        jnp.logical_or(new_achievements[:, Achievement.EAT_PLANT.value], is_eating_plant)
    )
    
    new_growing_plants_age = update_plants_with_eat(
        state, block_position, is_any_player_eating_plant
    )

    # Stalagmite
    can_mine_stalagmite = jnp.logical_and(state.inventory.pickaxe >= 1, is_miner)
    is_block_stalagmite = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.STALAGMITE.value
    )
    is_mining_stalagmite = jnp.logical_and(
        jnp.logical_and(
            is_block_stalagmite,
            can_mine_stalagmite
        ),
        doing_action,
    )
    is_any_player_mining_stalagmite = jnp.logical_and(
        equal_block_placement,
        is_mining_stalagmite[:, None]
    ).any(axis=0)
    mined_stalagmite_block = jnp.where(
        is_any_player_mining_stalagmite,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_stalagmite_block)
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone + 1 * is_mining_stalagmite
    )

    # Chest
    is_block_chest = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.CHEST.value
    )
    is_opening_chest = jnp.logical_and(
        is_block_chest,
        doing_action,
    )
    is_any_player_opening_chest = jnp.logical_and(
        equal_block_placement,
        is_opening_chest[:, None]
    ).any(axis=0)

    mined_chest_block = jnp.where(
        is_any_player_opening_chest,
        BlockType.PATH.value,
        new_map[block_position[:, 0], block_position[:, 1]],
    )
    new_map = new_map.at[block_position[:, 0], block_position[:, 1]].set(mined_chest_block)
    rng, _rng = jax.random.split(rng)
    new_inventory = add_items_from_chest(_rng, state, new_inventory, is_opening_chest)

    new_chests_opened = state.chests_opened.at[state.player_level].set(
        jnp.logical_or(state.chests_opened[state.player_level], is_opening_chest)
    )

    new_achievements = new_achievements.at[:, Achievement.OPEN_CHEST.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.OPEN_CHEST.value], is_opening_chest
        )
    )

    # Boss
    is_attacking_boss = jnp.logical_and(
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.NECROMANCER.value,
        doing_action,
    )
    can_damage_boss = jnp.logical_and(
        is_boss_vulnerable(state), is_fighting_boss(state, static_params)
    )
    is_damaging_boss = jnp.logical_and(
        is_attacking_boss,
        can_damage_boss,
    )

    # Обновляем прогресс босса (ff_damage_dealt теперь используется для паттерна, поэтому boss_progress можно обновлять свободно)
    new_boss_progress = state.boss_progress + 1 * is_damaging_boss.any()
    new_boss_timesteps_to_spawn_this_round = (
        BOSS_FIGHT_SPAWN_TURNS * is_damaging_boss.any()
        + state.boss_timesteps_to_spawn_this_round * (1 - is_damaging_boss.any())
    )

    new_achievements = new_achievements.at[:, Achievement.DAMAGE_NECROMANCER.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.DAMAGE_NECROMANCER.value], is_damaging_boss
        )
    )

    new_whole_map = state.map.at[state.player_level].set(new_map)

    state = state.replace(
        map=new_whole_map,
        inventory=new_inventory,
        player_drink=new_drink,
        player_thirst=new_thirst,
        player_food=new_food,
        player_hunger=new_hunger,
        growing_plants_age=new_growing_plants_age,
        achievements=new_achievements,
        chests_opened=new_chests_opened,
        boss_progress=new_boss_progress,
        boss_timesteps_to_spawn_this_round=new_boss_timesteps_to_spawn_this_round,
    )

    return state


def cast_spell(state, action, static_params):
    is_miner = state.player_specialization == Specialization.MINER.value
    is_warrior = state.player_specialization == Specialization.BUILDER.value
    is_crafter = state.player_specialization == Specialization.CRAFTER.value

    spell_mana_cost = jnp.array([2,6]) # fireball costs 2, healing costs 5

    def _cast_player_spell(player_info, player_index):
        player_projectiles, player_projectile_directions, player_projectile_owners, player_health = player_info

        is_casting_spell = jnp.logical_and(
            action[player_index] == Action.CAST_SPELL.value,
            state.learned_spells[player_index]
        )

        # Warriors/Miners -> Cast Fireball
        is_casting_fireball = jnp.logical_and(
            is_casting_spell, state.player_mana[player_index] >= spell_mana_cost[0]
        )
        is_casting_fireball = jnp.logical_and(
            is_casting_fireball,
            jnp.logical_and(
                jnp.logical_or(is_miner[player_index], is_warrior[player_index]),
                player_projectiles.mask[state.player_level].sum()
                < (static_params.max_player_projectiles * static_params.player_count),
            )
        )
        new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners = spawn_projectile(
            state,
            static_params,
            player_projectiles,
            player_projectile_directions,
            player_projectile_owners,
            state.player_position[player_index],
            is_casting_fireball,
            player_index,
            DIRECTIONS[state.player_direction[player_index]],
            ProjectileType.FIREBALL.value,
        )


        # Crafters -> Healing
        is_casting_healing = jnp.logical_and(is_casting_spell, is_crafter[player_index])
        is_casting_healing = jnp.logical_and(
            is_casting_healing, state.player_mana[player_index] >= spell_mana_cost[1]
        )
        health_increase = 2
        new_player_health = jnp.minimum(
            player_health + state.player_alive * (health_increase * is_casting_healing), 
            get_max_health(state)
        )

        spell_cast = jnp.array([is_casting_fireball, is_casting_healing])

        return (new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners, new_player_health), spell_cast
    
    (
        new_player_projectiles, 
        new_player_projectile_directions, 
        new_player_projectile_owners,
        new_player_health
    ), spell_cast = jax.lax.scan(
        _cast_player_spell, 
        (
            state.player_projectiles, 
            state.player_projectile_directions, 
            state.player_projectile_owners,
            state.player_health,
        ), 
        jnp.arange(static_params.player_count)
    ) 
    did_cast_spell = spell_cast.any(axis=-1)
    new_achievements = state.achievements.at[:, Achievement.CAST_SPELL.value].set(
        jnp.logical_or(state.achievements[:, Achievement.CAST_SPELL.value], did_cast_spell)
    )

    return state.replace(
        player_projectiles=new_player_projectiles,
        player_projectile_directions=new_player_projectile_directions,
        player_projectile_owners=new_player_projectile_owners,
        player_health=new_player_health,
        player_mana=state.player_mana - jnp.dot(spell_cast, spell_mana_cost),
        achievements=new_achievements,
    )


def enchant(rng, state: EnvState, action, static_params: StaticEnvParams):
    target_block_position = state.player_position + DIRECTIONS[state.player_direction]
    target_block = state.map[
        state.player_level, target_block_position[:, 0], target_block_position[:, 1]
    ]
    target_block_is_enchantment_table = jnp.logical_or(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value,
        target_block == BlockType.ENCHANTMENT_TABLE_ICE.value,
    )

    enchantment_type = jnp.where(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value, 
        1, 
        2
    )

    num_gems = jnp.where(
        target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value,
        state.inventory.ruby,
        state.inventory.sapphire,
    )

    could_enchant = jnp.logical_and(
        state.player_mana >= 9,
        jnp.logical_and(target_block_is_enchantment_table, num_gems >= 1),
    )
    could_enchant_crafter = jnp.logical_and(
        state.player_specialization == Specialization.CRAFTER.value,
        could_enchant
    )

    is_enchanting_bow = jnp.logical_and(
        could_enchant_crafter,
        jnp.logical_and(action == Action.ENCHANT_BOW.value, state.inventory.bow > 0),
    )

    is_enchanting_sword = jnp.logical_and(
        could_enchant_crafter,
        jnp.logical_and(
            action == Action.ENCHANT_SWORD.value, state.inventory.sword > 0
        ),
    )

    is_enchanting_armour = jnp.logical_and(
        could_enchant,
        jnp.logical_and(
            action == Action.ENCHANT_ARMOUR.value, state.inventory.armour.sum(axis=1) > 0
        ),
    )

    rng, _rng = jax.random.split(rng)
    unenchanted_armour = state.armour_enchantments == 0
    opposite_enchanted_armour = jnp.logical_and(
        state.armour_enchantments != 0, state.armour_enchantments != enchantment_type[:, None]
    )

    armour_targets = (
        unenchanted_armour + (unenchanted_armour.sum(axis=1) == 0)[:, None] * opposite_enchanted_armour
    )

    _rngs = jax.random.split(rng, static_params.player_count+1)
    rng, _rng = _rngs[0], _rngs[1:]
    armour_target = jax.vmap(jax.random.choice, in_axes=(0, None, None, None, 0))(
        _rng, jnp.arange(4), (), True, armour_targets
    )

    is_enchanting = jnp.logical_or(
        is_enchanting_sword, jnp.logical_or(is_enchanting_bow, is_enchanting_armour)
    )

    new_sword_enchantment = (
        is_enchanting_sword * enchantment_type
        + (1 - is_enchanting_sword) * state.sword_enchantment
    )
    new_bow_enchantment = (
        is_enchanting_bow * enchantment_type
        + (1 - is_enchanting_bow) * state.bow_enchantment
    )

    new_armour_enchantments = state.armour_enchantments.at[jnp.arange(static_params.player_count), armour_target].set(
        is_enchanting_armour * enchantment_type
        + (1 - is_enchanting_armour) * state.armour_enchantments[jnp.arange(static_params.player_count), armour_target]
    )

    new_sapphire = state.inventory.sapphire - 1 * is_enchanting * (
        enchantment_type == 2
    )
    new_ruby = state.inventory.ruby - 1 * is_enchanting * (enchantment_type == 1)
    new_mana = state.player_mana - 9 * is_enchanting

    new_achievements = state.achievements.at[:, Achievement.ENCHANT_SWORD.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.ENCHANT_SWORD.value], is_enchanting_sword
        )
    )

    new_achievements = new_achievements.at[:, Achievement.ENCHANT_ARMOUR.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.ENCHANT_ARMOUR.value], is_enchanting_armour
        )
    )

    return state.replace(
        sword_enchantment=new_sword_enchantment,
        bow_enchantment=new_bow_enchantment,
        armour_enchantments=new_armour_enchantments,
        inventory=state.inventory.replace(
            sapphire=new_sapphire,
            ruby=new_ruby,
        ),
        player_mana=new_mana,
        achievements=new_achievements,
    )


def place_block(state, action, static_params):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]
    equal_block_placement = (jnp.expand_dims(placing_block_position, axis=1) == jnp.expand_dims(placing_block_position, axis=0)).all(axis=2)

    new_map = state.map[state.player_level]
    new_item_map = state.item_map[state.player_level]

    is_block_in_other_player = is_in_other_player(state, placing_block_position)
    is_block_in_mob = is_in_mob(state, placing_block_position)
    is_block_in_bounds = in_bounds(placing_block_position, static_params)
    is_placement_in_bounds_not_in_mobs = jnp.logical_and(
        is_block_in_bounds,
        jnp.logical_not(jnp.logical_or(
            is_block_in_other_player,
            is_block_in_mob
        ))
    )

    # Crafting table
    is_valid_placement = jnp.logical_and(
        is_placement_in_bounds_not_in_mobs,
        jnp.logical_and(
            jnp.logical_and(
                jnp.logical_not(is_in_solid_block(new_map, placing_block_position)),
                new_item_map[placing_block_position[:, 0], placing_block_position[:, 1]]
                == ItemType.NONE.value,
            ),
            state.player_specialization == Specialization.BUILDER.value
        )
    )
    crafting_table_key_down = action == Action.PLACE_TABLE.value
    has_wood = state.inventory.wood >= 2
    is_player_placing_crafting_table = jnp.logical_and(
        crafting_table_key_down,
        jnp.logical_and(is_valid_placement, has_wood),
    )
    is_any_player_placing_crafting_table = jnp.logical_and(
        equal_block_placement,
        is_player_placing_crafting_table[:, None]
    ).any(axis=0)

    placed_crafting_table_block = jnp.where(
        is_any_player_placing_crafting_table,
        BlockType.CRAFTING_TABLE.value,
        new_map[
            placing_block_position[:, 0], placing_block_position[:, 1]
        ],
    )
    new_map = (
        new_map
        .at[placing_block_position[:, 0], placing_block_position[:, 1]]
        .set(placed_crafting_table_block)
    )
    new_inventory = state.inventory.replace(
        wood=state.inventory.wood - 2 * is_player_placing_crafting_table
    )
    new_achievements = state.achievements.at[:, Achievement.PLACE_TABLE.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.PLACE_TABLE.value], is_player_placing_crafting_table
        )
    )


    # Furnace
    is_valid_placement = jnp.logical_and(
        is_placement_in_bounds_not_in_mobs,
        jnp.logical_and(
            jnp.logical_and(
                jnp.logical_not(is_in_solid_block(new_map, placing_block_position)),
                new_item_map[placing_block_position[:, 0], placing_block_position[:, 1]]
                == ItemType.NONE.value,
            ),
            state.player_specialization == Specialization.BUILDER.value
        )
    )

    furnace_key_down = action == Action.PLACE_FURNACE.value
    has_stone = new_inventory.stone > 0
    is_player_placing_furnace = jnp.logical_and(
        furnace_key_down,
        jnp.logical_and(
            is_valid_placement, has_stone
        ),
    )
    is_any_player_placing_furnace = jnp.logical_and(
        equal_block_placement,
        is_player_placing_furnace[:, None]
    ).any(axis=0)
    placed_furnace_block = jnp.where(
        is_any_player_placing_furnace,
        BlockType.FURNACE.value,
        new_map[placing_block_position[:, 0], placing_block_position[:, 1]],
    )
    new_map = new_map.at[placing_block_position[:, 0], placing_block_position[:, 1]].set(
        placed_furnace_block
    )
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_player_placing_furnace
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_FURNACE.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_FURNACE.value], is_player_placing_furnace
        )
    )

    # Stone
    stone_key_down = action == Action.PLACE_STONE.value
    has_stone = new_inventory.stone > 0
    is_valid_placement = jnp.logical_and(
        is_placement_in_bounds_not_in_mobs,
        new_item_map[placing_block_position[:, 0], placing_block_position[:, 1]]
        == ItemType.NONE.value,
    )
    is_valid_placement = jnp.logical_and(
        is_valid_placement,
        jnp.logical_or(
            new_map[
                placing_block_position[:, 0], placing_block_position[:, 1]
            ]
            == BlockType.WATER.value,
            jnp.logical_not(is_in_solid_block(new_map, placing_block_position)),
        )
    )
    is_player_placing_stone = jnp.logical_and(
        stone_key_down,
        jnp.logical_and(is_valid_placement, has_stone),
    )
    is_player_placing_stone = jnp.logical_and(
        is_player_placing_stone,
        state.player_specialization == Specialization.MINER.value
    )
    is_any_player_placing_stone = jnp.logical_and(
        equal_block_placement,
        is_player_placing_stone[:, None]
    ).any(axis=0)
    placed_stone_block = jnp.where(
        is_any_player_placing_stone,
        BlockType.STONE.value,
        new_map[placing_block_position[:, 0], placing_block_position[:, 1]],
    )
    new_map = new_map.at[placing_block_position[:, 0], placing_block_position[:, 1]].set(
        placed_stone_block
    )
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_player_placing_stone
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_STONE.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_STONE.value], is_player_placing_stone
        )
    )

    # Torch
    # TODO: Make more parallelized
    def _player_place_torch(action_info, player_index):
        (
            working_item_map,
            working_padded_light_map,
        ) = action_info

        torch_key_down = action[player_index] == Action.PLACE_TORCH.value
        has_torch = new_inventory.torches[player_index] > 0
        
        is_valid_placement = jnp.logical_and(
            CAN_PLACE_ITEM_MAPPING[
                new_map[
                    placing_block_position[player_index, 0], placing_block_position[player_index, 1]
                ]
            ],
            working_item_map[placing_block_position[player_index, 0], placing_block_position[player_index, 1]]
            == ItemType.NONE.value,
        )
        is_valid_placement = jnp.logical_and(
            is_placement_in_bounds_not_in_mobs[player_index],
            is_valid_placement
        )
        is_player_placing_torch = jnp.logical_and(
            torch_key_down,
            jnp.logical_and(is_valid_placement, has_torch),
        )
        placed_torch_item = jax.lax.select(
            is_player_placing_torch,
            ItemType.TORCH.value,
            working_item_map[placing_block_position[player_index, 0], placing_block_position[player_index, 1]],
        )
        working_item_map = working_item_map.at[
            placing_block_position[player_index, 0], placing_block_position[player_index, 1]
        ].set(placed_torch_item)

        current_light_map = jax.lax.dynamic_slice(
            working_padded_light_map,
            placing_block_position[player_index]
            - jnp.array([4, 4])
            + jnp.array([light_map_padding, light_map_padding]),
            (9, 9),
        )
        torch_light_map = jnp.clip(TORCH_LIGHT_MAP + current_light_map, 0.0, 1.0)
        torch_light_map = torch_light_map * is_player_placing_torch + current_light_map * (
            1 - is_player_placing_torch
        )
        working_padded_light_map = jax.lax.dynamic_update_slice(
            working_padded_light_map,
            torch_light_map,
            placing_block_position[player_index]
            - jnp.array([4, 4])
            + jnp.array([light_map_padding, light_map_padding]),
        )
        return (working_item_map, working_padded_light_map), is_player_placing_torch
    
    light_map_padding = 6
    padded_light_map_floor = jnp.pad(
        state.light_map[state.player_level],
        (light_map_padding, light_map_padding),
        constant_values=0,
    )
    (new_item_map, padded_light_map_floor), is_player_placing_torch = jax.lax.scan(
        _player_place_torch,
        (new_item_map, padded_light_map_floor),
        jnp.arange(static_params.player_count)
    )
    new_light_map_floor = padded_light_map_floor[
        light_map_padding:-light_map_padding, light_map_padding:-light_map_padding
    ]
    new_light_map = state.light_map.at[state.player_level].set(new_light_map_floor)

    new_inventory = new_inventory.replace(
        torches=new_inventory.torches - 1 * is_player_placing_torch
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_TORCH.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_TORCH.value], is_player_placing_torch
        )
    )


    # Plant
    # TODO: Make more parallelized
    def _player_place_plant(action_info, player_index):
        (
            working_map, 
            working_growing_plants_positions,
            working_growing_plants_age,
            working_growing_plants_mask,
        ) = action_info
        sapling_key_down = action[player_index] == Action.PLACE_PLANT.value
        has_sapling = state.inventory.sapling[player_index] > 0
        is_valid_placement = jnp.logical_and(
            is_placement_in_bounds_not_in_mobs[player_index],
            jnp.logical_and(
                working_map[placing_block_position[player_index, 0], placing_block_position[player_index, 1]]
                == BlockType.GRASS.value,
                new_item_map[placing_block_position[player_index, 0], placing_block_position[player_index, 1]]
                == ItemType.NONE.value,
            )
        )
        is_player_placing_sapling = jnp.logical_and(
            is_valid_placement,
            jnp.logical_and(
                sapling_key_down,
                has_sapling,
            )
        )
        (
            working_growing_plants_positions,
            working_growing_plants_age,
            working_growing_plants_mask,
            is_player_placing_sapling
        ) = add_new_growing_plant(
            working_growing_plants_positions,
            working_growing_plants_age,
            working_growing_plants_mask,
            placing_block_position[player_index], 
            is_player_placing_sapling,
        )
        placed_sapling_block = jax.lax.select(
            is_player_placing_sapling,
            BlockType.PLANT.value,
            working_map[placing_block_position[player_index, 0], placing_block_position[player_index, 1]],
        )
        working_map = working_map.at[placing_block_position[player_index, 0], placing_block_position[player_index, 1]].set(
            placed_sapling_block
        )
        return (
            working_map,
            working_growing_plants_positions,
            working_growing_plants_age,
            working_growing_plants_mask,
        ), is_player_placing_sapling

    (new_map, new_growing_plants_positions, new_growing_plants_age, new_growing_plants_mask), is_player_placing_sapling = jax.lax.scan(
        _player_place_plant,
        (new_map, state.growing_plants_positions, state.growing_plants_age, state.growing_plants_mask),
        jnp.arange(static_params.player_count)
    )

    new_inventory = new_inventory.replace(
        sapling=new_inventory.sapling - 1 * is_player_placing_sapling
    )
    new_achievements = new_achievements.at[:, Achievement.PLACE_PLANT.value].set(
        jnp.logical_or(
            new_achievements[:, Achievement.PLACE_PLANT.value], is_player_placing_sapling
        )
    )

    # Do?
    new_whole_map = state.map.at[state.player_level].set(new_map)
    new_whole_item_map = state.item_map.at[state.player_level].set(new_item_map)
    state = state.replace(
        map=new_whole_map,
        item_map=new_whole_item_map,
        light_map=new_light_map,
        inventory=new_inventory,
        achievements=new_achievements,
        growing_plants_positions=new_growing_plants_positions,
        growing_plants_age=new_growing_plants_age,
        growing_plants_mask=new_growing_plants_mask,
    )

    return state


def trade_materials(state, action, static_params):
    new_achievements = state.achievements
    new_trade_count = state.trade_count
    new_food_trade_count = state.food_trade_count
    new_drink_trade_count = state.drink_trade_count
    
    # Отслеживаем успешные передачи для награды
    successful_trades = jnp.zeros(static_params.player_count, dtype=jnp.int32)

    player_trading_to = action - Action.GIVE.value
    player_trading_to += 1 * (player_trading_to >= jnp.arange(static_params.player_count))
    is_giving = jnp.logical_and(
        action >= Action.GIVE.value, 
        action < (Action.GIVE.value + static_params.player_count - 1)
    )
    other_player_is_requesting = jnp.logical_and(
        state.request_duration[player_trading_to] > 0,
        state.player_alive[player_trading_to]
    )

    def _new_material_value(material_type, current_material_stock, material_max_value, old_trade_count, successful_trades, can_trade_mask=None):
        """
        can_trade_mask - маска булевых значений, указывающая, может ли каждый игрок передавать этот материал
        Если None, то все могут передавать
        """
        other_player_is_requesting_material = jnp.logical_and(
            other_player_is_requesting,
            state.request_type[player_trading_to] == material_type
        )
        # Проверка специализации (если указана)
        can_trade = jnp.ones(static_params.player_count, dtype=bool) if can_trade_mask is None else can_trade_mask
        is_giving_material = jnp.logical_and(
            jnp.logical_and( # Checks that other player is requesting and can take materials
                other_player_is_requesting_material,
                current_material_stock[player_trading_to] < material_max_value
            ),
            jnp.logical_and( # Checks that player has materials and is giving
                is_giving,
                jnp.logical_and(
                    current_material_stock > 0,
                    can_trade  # Проверка специализации
                )
            )
        )
        new_material = current_material_stock - 1 * is_giving_material
        new_material = new_material.at[player_trading_to].add(is_giving_material)
        # Отслеживаем успешные передачи
        successful_trades = successful_trades + is_giving_material.astype(jnp.int32)
        return new_material, old_trade_count + is_giving_material.sum(), successful_trades
    
    # Food - обмен едой отключен, агенты должны получать еду напрямую (растения, коровы)
    new_food = state.player_food
    new_hunger = state.player_hunger  # Не обновляется через обмен
    # new_achievements для COLLECT_FOOD не обновляется через обмен
    
    # Drink - обмен водой отключен, агенты должны пить напрямую из источника
    new_drink = state.player_drink
    new_thirst = state.player_thirst  # Не обновляется через обмен

    # Inventory Materials
    # Дерево и камень может передавать только MINER
    is_miner = state.player_specialization == Specialization.MINER.value
    new_wood, new_trade_count, successful_trades = _new_material_value(
        Action.REQUEST_WOOD.value, state.inventory.wood, 99, new_trade_count, successful_trades, can_trade_mask=is_miner
    )
    new_stone, new_trade_count, successful_trades = _new_material_value(
        Action.REQUEST_STONE.value, state.inventory.stone, 99, new_trade_count, successful_trades, can_trade_mask=is_miner
    )
    new_iron, new_trade_count, successful_trades = _new_material_value(
        Action.REQUEST_IRON.value, state.inventory.iron, 99, new_trade_count, successful_trades
    )
    new_coal, new_trade_count, successful_trades = _new_material_value(
        Action.REQUEST_COAL.value, state.inventory.coal, 99, new_trade_count, successful_trades
    )
    new_diamond, new_trade_count, successful_trades = _new_material_value(
        Action.REQUEST_DIAMOND.value, state.inventory.diamond, 99, new_trade_count, successful_trades
    )
    new_ruby, new_trade_count, successful_trades = _new_material_value(
        Action.REQUEST_RUBY.value, state.inventory.ruby, 99, new_trade_count, successful_trades
    )
    new_sapphire, new_trade_count, successful_trades = _new_material_value(
        Action.REQUEST_SAPPHIRE.value, state.inventory.sapphire, 99, new_trade_count, successful_trades
    )

    # Tools and Equipment
    # Pickaxe, Sword, Bow - передаем уровень инструмента, если у получающего хуже
    # Инструменты может передавать только CRAFTER
    def _new_tool_value(tool_type, current_tool_level, max_tool_level, old_trade_count, successful_trades, can_trade_mask=None):
        """
        can_trade_mask - маска булевых значений, указывающая, может ли каждый игрок передавать этот инструмент
        Если None, то все могут передавать
        """
        other_player_is_requesting_tool = jnp.logical_and(
            other_player_is_requesting,
            state.request_type[player_trading_to] == tool_type
        )
        # Проверяем, что у передающего есть инструмент и у получающего хуже или нет
        has_tool = current_tool_level > 0
        receiver_tool_level = current_tool_level[player_trading_to]
        giver_tool_level = current_tool_level
        receiver_needs_better = receiver_tool_level < giver_tool_level
        # Проверка специализации (если указана)
        can_trade = jnp.ones(static_params.player_count, dtype=bool) if can_trade_mask is None else can_trade_mask
        is_giving_tool = jnp.logical_and(
            jnp.logical_and(
                other_player_is_requesting_tool,
                receiver_needs_better
            ),
            jnp.logical_and(
                is_giving,
                jnp.logical_and(
                    has_tool,
                    can_trade  # Проверка специализации
                )
            )
        )
        # Передаем уровень инструмента получателю, если у него хуже
        new_receiver_level = jnp.where(
            is_giving_tool,
            jnp.maximum(receiver_tool_level, giver_tool_level),
            receiver_tool_level
        )
        new_tool_level = current_tool_level.at[player_trading_to].set(new_receiver_level)
        # Отслеживаем успешные передачи
        successful_trades = successful_trades + is_giving_tool.astype(jnp.int32)
        # У передающего инструмент остается (не тратится при обмене)
        return new_tool_level, old_trade_count + is_giving_tool.sum(), successful_trades
    
    # Инструменты может передавать только CRAFTER
    is_crafter = state.player_specialization == Specialization.CRAFTER.value
    new_pickaxe, new_trade_count, successful_trades = _new_tool_value(
        REQUEST_PICKAXE, state.inventory.pickaxe, 4, new_trade_count, successful_trades, can_trade_mask=is_crafter
    )
    new_sword, new_trade_count, successful_trades = _new_tool_value(
        REQUEST_SWORD, state.inventory.sword, 4, new_trade_count, successful_trades, can_trade_mask=is_crafter
    )
    new_bow, new_trade_count, successful_trades = _new_tool_value(
        REQUEST_BOW, state.inventory.bow, 1, new_trade_count, successful_trades, can_trade_mask=is_crafter
    )

    # Armour - передаем одну единицу брони, если у получающего есть свободный слот
    # Броню может передавать только CRAFTER
    def _new_armour_value(tool_type, current_armour, old_trade_count, successful_trades, can_trade_mask=None):
        """
        can_trade_mask - маска булевых значений, указывающая, может ли каждый игрок передавать броню
        Если None, то все могут передавать
        """
        other_player_is_requesting_armour = jnp.logical_and(
            other_player_is_requesting,
            state.request_type[player_trading_to] == tool_type
        )
        # Проверяем, что у передающего есть броня и у получающего есть свободный слот
        has_armour = current_armour.sum(axis=1) > 0
        receiver_armour = current_armour[player_trading_to]
        receiver_has_space = (receiver_armour < 2).any(axis=1)
        # Проверка специализации (если указана)
        can_trade = jnp.ones(static_params.player_count, dtype=bool) if can_trade_mask is None else can_trade_mask
        is_giving_armour = jnp.logical_and(
            jnp.logical_and(
                other_player_is_requesting_armour,
                receiver_has_space
            ),
            jnp.logical_and(
                is_giving,
                jnp.logical_and(
                    has_armour,
                    can_trade  # Проверка специализации
                )
            )
        )
        
        # Находим первый свободный слот у каждого получателя
        receiver_free_slots = receiver_armour < 2
        receiver_slot = jnp.argmax(receiver_free_slots, axis=1)
        
        # Находим первую единицу брони у каждого передающего
        giver_armour_slots = current_armour > 0
        giver_slot = jnp.argmax(giver_armour_slots, axis=1)
        giver_armour_level = current_armour[jnp.arange(static_params.player_count), giver_slot]
        
        # Передаем броню получателю
        new_receiver_armour = receiver_armour.at[
            jnp.arange(static_params.player_count),
            receiver_slot
        ].set(
            jnp.where(
                is_giving_armour,
                giver_armour_level,
                receiver_armour[jnp.arange(static_params.player_count), receiver_slot]
            )
        )
        new_armour = current_armour.at[player_trading_to].set(new_receiver_armour)
        # Отслеживаем успешные передачи
        successful_trades = successful_trades + is_giving_armour.astype(jnp.int32)
        # У передающего броня остается (не тратится при обмене)
        return new_armour, old_trade_count + is_giving_armour.sum(), successful_trades
    
    new_armour, new_trade_count, successful_trades = _new_armour_value(
        REQUEST_ARMOUR, state.inventory.armour, new_trade_count, successful_trades, can_trade_mask=is_crafter
    )

    # Arrows and Torches - как обычные материалы
    new_arrows, new_trade_count, successful_trades = _new_material_value(
        REQUEST_ARROWS, state.inventory.arrows, 99, new_trade_count, successful_trades
    )
    new_torches, new_trade_count, successful_trades = _new_material_value(
        REQUEST_TORCHES, state.inventory.torches, 99, new_trade_count, successful_trades
    )

    # Update State
    # Сохраняем количество успешных передач в состоянии для расчета награды
    state = state.replace(
        player_food=new_food,
        player_drink=new_drink,
        player_hunger=new_hunger,
        player_thirst=new_thirst,
        inventory=state.inventory.replace(
            wood=new_wood,
            stone=new_stone,
            iron=new_iron,
            coal=new_coal,
            diamond=new_diamond,
            ruby=new_ruby,
            sapphire=new_sapphire,
            pickaxe=new_pickaxe,
            sword=new_sword,
            bow=new_bow,
            armour=new_armour,
            arrows=new_arrows,
            torches=new_torches,
        ),
        achievements=new_achievements,
        trade_count=new_trade_count,
        food_trade_count=new_food_trade_count,
        drink_trade_count=new_drink_trade_count,
    )
    # Возвращаем состояние и количество успешных передач
    return state, successful_trades


def make_request(state, action):
    """Переопределенная функция для поддержки запросов инструментов"""
    from craftax_coop.constants import REQUEST_MAX_DURATION
    
    # Decrease Request Duration
    state = state.replace(
        request_duration=jnp.maximum(0, state.request_duration - 1)
    )

    # Initialize New Request - расширяем диапазон для запросов инструментов
    is_making_material_request = jnp.logical_and(
        action >= Action.REQUEST_FOOD.value, 
        action <= Action.REQUEST_SAPPHIRE.value
    )
    is_making_tool_request = jnp.logical_and(
        action >= REQUEST_PICKAXE,
        action <= REQUEST_TORCHES
    )
    is_making_request = jnp.logical_or(
        is_making_material_request,
        is_making_tool_request
    )
    
    new_request_type = jnp.where(
        is_making_request,
        action,
        state.request_type
    )
    state = state.replace(
        request_duration=jnp.maximum(state.request_duration, is_making_request * REQUEST_MAX_DURATION),
        request_type=new_request_type
    )
    return state


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


# ===========================
# Environment Wrappers
# ===========================
from craftax_coop.envs.craftax_symbolic_env import CraftaxCoopSymbolicEnv
from craftax_coop.envs.craftax_pixels_env import CraftaxCoopPixelsEnv
from jaxmarl.environments import spaces


class CraftaxCoopSymbolicEnvWithMyWrapper(CraftaxCoopSymbolicEnv):
    """
    Обертка для CraftaxCoopSymbolicEnv, которая использует craftax_step из my_wrapper.
    На каждый эпизод выбирается один из четырех паттернов для проверки.
    Если указан fixed_pattern_id, используется фиксированный паттерн для всех эпизодов.
    """
    def __init__(self, num_agents: int = 3, fixed_pattern_id: int = None):
        """
        Args:
            num_agents: Количество агентов
            fixed_pattern_id: Опциональный фиксированный ID паттерна (0-3). 
                             Если None, паттерн выбирается случайно каждый эпизод.
        """
        super().__init__(num_agents)
        # Используем -1 для обозначения случайного выбора (совместимо с JAX)
        self.fixed_pattern_id = jnp.int32(fixed_pattern_id) if fixed_pattern_id is not None else jnp.int32(-1)
    
    def action_shape(self) -> spaces.Discrete:
        """
        Переопределенный метод action_shape.
        Расширяет action space для поддержки запросов инструментов (REQUEST_PICKAXE, REQUEST_SWORD и т.д.).
        """
        # Базовый размер: len(Action) + (num_agents - 2) для GIVE действий
        base_size = len(Action) + (self.static_env_params.player_count - 2)
        # Добавляем 6 действий для запросов инструментов (REQUEST_PICKAXE до REQUEST_TORCHES)
        extended_size = base_size + 6
        return spaces.Discrete(extended_size)
    
    def observation_shape(self) -> spaces.Box:
        """
        Переопределенный метод observation_shape.
        Добавляет 4 элемента для one-hot кодирования паттерна.
        """
        base_obs_shape = (
            self.get_flat_map_obs_shape() + 
            self.get_teammate_dashboard_obs_shape() + 
            self.get_inventory_obs_shape()
        )
        # Добавляем 4 элемента для one-hot вектора паттерна
        obs_shape = base_obs_shape + 4
        
        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.int32,
        )
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, _=None) -> Tuple[Dict[str, chex.Array], EnvState]:
        """
        Переопределенный метод reset.
        Использует фиксированный паттерн (если указан) или выбирает случайный.
        """
        # Вызываем базовый reset
        obs, state = super().reset(key, _)
        
        # Выбираем паттерн: фиксированный или случайный
        # 0: check_two_tables_straight_line
        # 1: check_two_tables_diagonal_line
        # 2: check_two_furnaces_straight_line
        # 3: check_two_furnaces_diagonal_line
        # Если fixed_pattern_id >= 0 и <= 3, используем его, иначе выбираем случайно
        key, pattern_key = jax.random.split(key)
        random_pattern = jax.random.randint(pattern_key, shape=(), minval=0, maxval=4)
        # Используем jax.lax.cond для условной логики в JIT-функции
        target_pattern_id = jax.lax.cond(
            (self.fixed_pattern_id >= 0) & (self.fixed_pattern_id <= 3),
            lambda: jnp.clip(self.fixed_pattern_id, 0, 3),  # Используем фиксированный паттерн
            lambda: random_pattern,  # Используем случайный паттерн
        )
        
        # Сохраняем выбранный паттерн в ff_damage_dealt (используем это поле для хранения паттерна)
        # Преобразуем паттерн в float для совместимости с типом поля
        state = state.replace(ff_damage_dealt=jnp.float32(target_pattern_id))
        
        return obs, state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """
        Переопределенный метод is_terminal.
        Эпизод завершается, если:
        1. Все агенты мертвы
        2. Выполнен выбранный паттерн (определяется state.ff_damage_dealt)
        """
        # Проверка: все ли агенты мертвы
        is_dead = jnp.logical_not(state.player_alive).all()
        
        # Получаем ID выбранного паттерна из state.ff_damage_dealt (используем это поле для хранения паттерна)
        target_pattern_id = jnp.clip(jnp.int32(state.ff_damage_dealt), 0, 3)
        
        # Проверяем ТОЛЬКО выбранный паттерн, используя lax.switch для надежности
        # Это гарантирует, что проверяется именно выбранный паттерн, а не все паттерны
        def check_pattern_0(args):
            state, static_params = args
            return check_two_tables_straight_line(state, static_params)
        
        def check_pattern_1(args):
            state, static_params = args
            return check_two_tables_diagonal_line(state, static_params)
        
        def check_pattern_2(args):
            state, static_params = args
            return check_two_furnaces_straight_line(state, static_params)
        
        def check_pattern_3(args):
            state, static_params = args
            return check_two_furnaces_diagonal_line(state, static_params)
        
        # Используем lax.switch для выбора нужной функции проверки
        pattern_checkers = (check_pattern_0, check_pattern_1, check_pattern_2, check_pattern_3)
        pattern_completed = lax.switch(target_pattern_id, pattern_checkers, (state, self.static_env_params))
        
        # Эпизод завершается, если все мертвы ИЛИ выполнен выбранный паттерн
        is_terminal = jnp.logical_or(is_dead, pattern_completed)
        return is_terminal
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: EnvState) -> Dict[str, chex.Array]:
        """
        Переопределенный метод get_obs.
        Добавляет one-hot кодирование выбранного паттерна к наблюдениям.
        """
        # Получаем базовые наблюдения
        obs_sym = lax.stop_gradient(
            render_craftax_symbolic(
                state, 
                self.static_env_params,
            )
        )
        
        # Кодируем паттерн в one-hot формат (4 паттерна: 0-3)
        target_pattern_id = jnp.clip(jnp.int32(state.ff_damage_dealt), 0, 3)
        pattern_one_hot = jax.nn.one_hot(target_pattern_id, num_classes=4)
        
        # Добавляем one-hot вектор паттерна к наблюдениям каждого агента
        pattern_one_hot_broadcast = pattern_one_hot[None, :].repeat(obs_sym.shape[0], axis=0)
        obs_sym_with_pattern = jnp.concatenate([obs_sym, pattern_one_hot_broadcast], axis=1)
        
        obs = {n: o for n, o in zip(self.agents, obs_sym_with_pattern)}
        return obs
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], chex.Array, Dict[str, float], Dict[str, bool], Dict]:
        actions_array = jnp.array(list(actions.values()))
        state, reward = craftax_step(
            key, state, actions_array, 
            self.default_params, 
            self.static_env_params
        )
        
        obs = self.get_obs(state)
        done = self.is_terminal(state, self.default_params)
        
        info = {}
        from craftax_coop.envs.common import compute_score
        user_info = compute_score(state, done, self.static_env_params)
        
        # Добавляем метрики success_rate для каждого паттерна
        # Проверяем, завершился ли эпизод успешно (паттерн выполнен, а не все мертвы)
        is_dead = jnp.logical_not(state.player_alive).all()
        target_pattern_id = jnp.clip(jnp.int32(state.ff_damage_dealt), 0, 3)
        
        # Проверяем, какой паттерн был выполнен
        def check_pattern_0(args):
            state, static_params = args
            return check_two_tables_straight_line(state, static_params)
        
        def check_pattern_1(args):
            state, static_params = args
            return check_two_tables_diagonal_line(state, static_params)
        
        def check_pattern_2(args):
            state, static_params = args
            return check_two_furnaces_straight_line(state, static_params)
        
        def check_pattern_3(args):
            state, static_params = args
            return check_two_furnaces_diagonal_line(state, static_params)
        
        pattern_checkers = (check_pattern_0, check_pattern_1, check_pattern_2, check_pattern_3)
        pattern_completed = lax.switch(target_pattern_id, pattern_checkers, (state, self.static_env_params))
        
        # Эпизод успешен, если он завершился из-за выполнения паттерна (а не из-за смерти всех)
        episode_success = jnp.logical_and(done, jnp.logical_and(pattern_completed, jnp.logical_not(is_dead)))
        
        # Вычисляем success для каждого паттерна
        # Success = 1.0 если эпизод успешен И выбранный паттерн соответствует проверяемому
        # Используем векторизацию вместо цикла для совместимости с JAX
        pattern_indices = jnp.arange(4)  # 0, 1, 2, 3
        # Для каждого паттерна проверяем, был ли он выбран и успешно выполнен
        # Умножаем на done, чтобы метрика была только для завершенных эпизодов
        pattern_success_array = jnp.where(
            jnp.logical_and(episode_success, target_pattern_id == pattern_indices),
            1.0,
            0.0
        ) * done  # Умножаем на done для совместимости с форматом achievements
        
        # Добавляем метрики для каждого паттерна
        # Создаем массивы для каждого агента (для совместимости с форматом achievements)
        # Используем статический список для создания метрик (не в цикле)
        user_info["Patterns/pattern_0_two_tables_straight_success_rate"] = jnp.full(
            self.static_env_params.player_count, 
            pattern_success_array[0], 
            dtype=jnp.float32
        )
        user_info["Patterns/pattern_1_two_tables_diagonal_success_rate"] = jnp.full(
            self.static_env_params.player_count, 
            pattern_success_array[1], 
            dtype=jnp.float32
        )
        user_info["Patterns/pattern_2_two_furnaces_straight_success_rate"] = jnp.full(
            self.static_env_params.player_count, 
            pattern_success_array[2], 
            dtype=jnp.float32
        )
        user_info["Patterns/pattern_3_two_furnaces_diagonal_success_rate"] = jnp.full(
            self.static_env_params.player_count, 
            pattern_success_array[3], 
            dtype=jnp.float32
        )
        
        # Добавляем общий success_rate для всех паттернов (один график для всех задач)
        # Общий success = 1.0 если эпизод успешен (независимо от паттерна)
        overall_success = jnp.where(episode_success, 1.0, 0.0) * done
        user_info["Patterns/overall_success_rate"] = jnp.full(
            self.static_env_params.player_count, 
            overall_success, 
            dtype=jnp.float32
        )
        
        info["user_info"] = user_info
        
        agent_rewards = {n: r for n, r in zip(self.agents, reward)}
        agent_done = {n: done for n in self.agents}
        agent_done["__all__"] = done
        
        return (
            obs,
            lax.stop_gradient(state),
            agent_rewards,
            agent_done,
            info,
        )


class CraftaxCoopPixelsEnvWithMyWrapper(CraftaxCoopPixelsEnv):
    """
    Обертка для CraftaxCoopPixelsEnv, которая использует craftax_step из my_wrapper.
    На каждый эпизод выбирается один из четырех паттернов для проверки.
    Если указан fixed_pattern_id, используется фиксированный паттерн для всех эпизодов.
    """
    def __init__(self, num_agents: int = 3, fixed_pattern_id: int = None):
        """
        Args:
            num_agents: Количество агентов
            fixed_pattern_id: Опциональный фиксированный ID паттерна (0-3). 
                             Если None, паттерн выбирается случайно каждый эпизод.
        """
        super().__init__(num_agents)
        # Используем -1 для обозначения случайного выбора (совместимо с JAX)
        self.fixed_pattern_id = jnp.int32(fixed_pattern_id) if fixed_pattern_id is not None else jnp.int32(-1)
    
    def action_shape(self) -> spaces.Discrete:
        """
        Переопределенный метод action_shape.
        Расширяет action space для поддержки запросов инструментов (REQUEST_PICKAXE, REQUEST_SWORD и т.д.).
        """
        # Базовый размер: len(Action) + (num_agents - 2) для GIVE действий
        base_size = len(Action) + (self.static_env_params.player_count - 2)
        # Добавляем 6 действий для запросов инструментов (REQUEST_PICKAXE до REQUEST_TORCHES)
        extended_size = base_size + 6
        return spaces.Discrete(extended_size)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, _=None) -> Tuple[Dict[str, chex.Array], EnvState]:
        """
        Переопределенный метод reset.
        Использует фиксированный паттерн (если указан) или выбирает случайный.
        """
        # Вызываем базовый reset
        obs, state = super().reset(key, _)
        
        # Выбираем паттерн: фиксированный или случайный
        # 0: check_two_tables_straight_line
        # 1: check_two_tables_diagonal_line
        # 2: check_two_furnaces_straight_line
        # 3: check_two_furnaces_diagonal_line
        # Если fixed_pattern_id >= 0 и <= 3, используем его, иначе выбираем случайно
        key, pattern_key = jax.random.split(key)
        random_pattern = jax.random.randint(pattern_key, shape=(), minval=0, maxval=4)
        # Используем jax.lax.cond для условной логики в JIT-функции
        target_pattern_id = jax.lax.cond(
            (self.fixed_pattern_id >= 0) & (self.fixed_pattern_id <= 3),
            lambda: jnp.clip(self.fixed_pattern_id, 0, 3),  # Используем фиксированный паттерн
            lambda: random_pattern,  # Используем случайный паттерн
        )
        
        # Сохраняем выбранный паттерн в ff_damage_dealt (используем это поле для хранения паттерна)
        # Преобразуем паттерн в float для совместимости с типом поля
        state = state.replace(ff_damage_dealt=jnp.float32(target_pattern_id))
        
        return obs, state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """
        Переопределенный метод is_terminal.
        Эпизод завершается, если:
        1. Все агенты мертвы
        2. Выполнен выбранный паттерн (определяется state.ff_damage_dealt)
        """
        # Проверка: все ли агенты мертвы
        is_dead = jnp.logical_not(state.player_alive).all()
        
        # Получаем ID выбранного паттерна из state.ff_damage_dealt (используем это поле для хранения паттерна)
        target_pattern_id = jnp.clip(jnp.int32(state.ff_damage_dealt), 0, 3)
        
        # Проверяем ТОЛЬКО выбранный паттерн, используя lax.switch для надежности
        # Это гарантирует, что проверяется именно выбранный паттерн, а не все паттерны
        def check_pattern_0(args):
            state, static_params = args
            return check_two_tables_straight_line(state, static_params)
        
        def check_pattern_1(args):
            state, static_params = args
            return check_two_tables_diagonal_line(state, static_params)
        
        def check_pattern_2(args):
            state, static_params = args
            return check_two_furnaces_straight_line(state, static_params)
        
        def check_pattern_3(args):
            state, static_params = args
            return check_two_furnaces_diagonal_line(state, static_params)
        
        # Используем lax.switch для выбора нужной функции проверки
        pattern_checkers = (check_pattern_0, check_pattern_1, check_pattern_2, check_pattern_3)
        pattern_completed = lax.switch(target_pattern_id, pattern_checkers, (state, self.static_env_params))
        
        # Эпизод завершается, если все мертвы ИЛИ выполнен выбранный паттерн
        is_terminal = jnp.logical_or(is_dead, pattern_completed)
        return is_terminal
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: EnvState) -> Dict[str, chex.Array]:
        """
        Переопределенный метод get_obs.
        Добавляет one-hot кодирование выбранного паттерна к наблюдениям.
        Для pixels окружения паттерн добавляется в info.
        """
        # Получаем базовые наблюдения (изображения)
        pixels = lax.stop_gradient(
            render_craftax_pixels(
                state,
                self.pixel_size,
                self.static_env_params,
                self.player_specific_textures
            ) / 255.0
        )
        
        # Для pixels окружения возвращаем изображения как есть
        # (паттерн можно добавить в info, но это требует изменения структуры наблюдений)
        obs = {n: o for n, o in zip(self.agents, pixels)}
        return obs
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], chex.Array, Dict[str, float], Dict[str, bool], Dict]:
        actions_array = jnp.array(list(actions.values()))
        state, reward = craftax_step(
            key, state, actions_array, 
            self.default_params, 
            self.static_env_params
        )
        
        obs = self.get_obs(state)
        done = self.is_terminal(state, self.default_params)
        
        info = {}
        from craftax_coop.envs.common import compute_score
        user_info = compute_score(state, done, self.static_env_params)
        info["discount"] = self.discount(state, self.default_params)
        
        # Добавляем информацию о паттерне в info для pixels окружения
        target_pattern_id = jnp.clip(jnp.int32(state.ff_damage_dealt), 0, 3)
        pattern_one_hot = jax.nn.one_hot(target_pattern_id, num_classes=4)
        info["target_pattern_one_hot"] = pattern_one_hot
        
        # Добавляем метрики success_rate для каждого паттерна
        # Проверяем, завершился ли эпизод успешно (паттерн выполнен, а не все мертвы)
        is_dead = jnp.logical_not(state.player_alive).all()
        
        # Проверяем, какой паттерн был выполнен
        def check_pattern_0(args):
            state, static_params = args
            return check_two_tables_straight_line(state, static_params)
        
        def check_pattern_1(args):
            state, static_params = args
            return check_two_tables_diagonal_line(state, static_params)
        
        def check_pattern_2(args):
            state, static_params = args
            return check_two_furnaces_straight_line(state, static_params)
        
        def check_pattern_3(args):
            state, static_params = args
            return check_two_furnaces_diagonal_line(state, static_params)
        
        pattern_checkers = (check_pattern_0, check_pattern_1, check_pattern_2, check_pattern_3)
        pattern_completed = lax.switch(target_pattern_id, pattern_checkers, (state, self.static_env_params))
        
        # Эпизод успешен, если он завершился из-за выполнения паттерна (а не из-за смерти всех)
        episode_success = jnp.logical_and(done, jnp.logical_and(pattern_completed, jnp.logical_not(is_dead)))
        
        # Вычисляем success для каждого паттерна
        # Success = 1.0 если эпизод успешен И выбранный паттерн соответствует проверяемому
        # Используем векторизацию вместо цикла для совместимости с JAX
        pattern_indices = jnp.arange(4)  # 0, 1, 2, 3
        # Для каждого паттерна проверяем, был ли он выбран и успешно выполнен
        # Умножаем на done, чтобы метрика была только для завершенных эпизодов
        pattern_success_array = jnp.where(
            jnp.logical_and(episode_success, target_pattern_id == pattern_indices),
            1.0,
            0.0
        ) * done  # Умножаем на done для совместимости с форматом achievements
        
        # Добавляем метрики для каждого паттерна
        # Создаем массивы для каждого агента (для совместимости с форматом achievements)
        # Используем статический список для создания метрик (не в цикле)
        user_info["Patterns/pattern_0_two_tables_straight_success_rate"] = jnp.full(
            self.static_env_params.player_count, 
            pattern_success_array[0], 
            dtype=jnp.float32
        )
        user_info["Patterns/pattern_1_two_tables_diagonal_success_rate"] = jnp.full(
            self.static_env_params.player_count, 
            pattern_success_array[1], 
            dtype=jnp.float32
        )
        user_info["Patterns/pattern_2_two_furnaces_straight_success_rate"] = jnp.full(
            self.static_env_params.player_count, 
            pattern_success_array[2], 
            dtype=jnp.float32
        )
        user_info["Patterns/pattern_3_two_furnaces_diagonal_success_rate"] = jnp.full(
            self.static_env_params.player_count, 
            pattern_success_array[3], 
            dtype=jnp.float32
        )
        
        # Добавляем общий success_rate для всех паттернов (один график для всех задач)
        # Общий success = 1.0 если эпизод успешен (независимо от паттерна)
        overall_success = jnp.where(episode_success, 1.0, 0.0) * done
        user_info["Patterns/overall_success_rate"] = jnp.full(
            self.static_env_params.player_count, 
            overall_success, 
            dtype=jnp.float32
        )
        
        info["user_info"] = user_info
        
        agent_rewards = {n: r for n, r in zip(self.agents, reward)}
        agent_done = {n: done for n in self.agents}
        agent_done["__all__"] = done
        
        return (
            obs,
            lax.stop_gradient(state),
            agent_rewards,
            agent_done,
            info,
        )


def make_craftax_env_with_my_wrapper(env_name: str, fixed_pattern_id: int = None):
    """
    Создает окружение Craftax с использованием my_wrapper вместо game_logic.
    
    Args:
        env_name: Имя окружения ("Craftax-Coop-Symbolic" или "Craftax-Coop-Pixels")
        fixed_pattern_id: Опциональный фиксированный ID паттерна (0-3). 
                         Если None, паттерн выбирается случайно каждый эпизод.
                         По умолчанию: None (случайный выбор)
        
    Returns:
        Окружение с использованием craftax_step из my_wrapper
    """
    if env_name == "Craftax-Coop-Symbolic":
        return CraftaxCoopSymbolicEnvWithMyWrapper(fixed_pattern_id=fixed_pattern_id)
    elif env_name == "Craftax-Coop-Pixels":
        return CraftaxCoopPixelsEnvWithMyWrapper(fixed_pattern_id=fixed_pattern_id)
    else:
        # Для других окружений используем стандартный способ
        from craftax.craftax_env import make_craftax_env_from_name
        return make_craftax_env_from_name(env_name)