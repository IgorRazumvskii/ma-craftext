import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from craftax_together_talens.game_logic.constants import *

from craftax_together_talens.util.game_logic_utils import (
    is_near_block,
    attack_mob,
    get_player_damage_vector,
    get_max_drink,
    get_max_food,
    get_max_health,
    is_boss_vulnerable,
    is_fighting_boss,
    spawn_projectile,
    in_bounds,
    is_in_other_player,
    is_in_mob,
    is_in_solid_block,
    get_damage_between_players,
)
from craftax_together_talens.constants import (
    Achievement,
    DIRECTIONS,
    BOSS_FIGHT_SPAWN_TURNS,
    ProjectileType,
    BlockType,
    Action,
    ItemType,
    CAN_PLACE_ITEM_MAPPING,
    TORCH_LIGHT_MAP,
)
from craftax_together_talens.craftax_state import EnvState, StaticEnvParams, EnvParams


def interplayer_interaction(state, block_position, is_doing_action, env_params, static_params):
    # If other player is down revive them, otherwise damage (if friendly fire is enabled)
    in_other_player = (jnp.expand_dims(state.player_position, axis=1) == jnp.expand_dims(block_position, axis=0)).all(axis=2).T
    player_interacting_with = jnp.argmax(in_other_player, axis=-1)

    is_interacting_with_other_player = jnp.logical_and(
        in_other_player.any(axis=-1),
        is_doing_action,
    )
    is_player_being_interacted_with = jnp.any(
        jnp.logical_and(
            jnp.arange(static_params.player_count)[:, None] == player_interacting_with,
            is_interacting_with_other_player[None, :]
        ),
        axis=-1
    )
    is_player_being_revived = jnp.logical_and(
        is_player_being_interacted_with,
        jnp.logical_not(state.player_alive),
    )

    damage_taken = jnp.zeros(static_params.player_count).at[player_interacting_with].add(
        is_interacting_with_other_player * get_damage_between_players(state, player_interacting_with)
    )
    damage_taken *= env_params.friendly_fire

    new_player_health = jnp.where(
        is_player_being_revived,
        1.0,
        state.player_health - damage_taken,
    )
    state = state.replace(
        player_health=new_player_health,
        revives=state.revives+is_player_being_revived.sum(),
        ff_damage_dealt=state.ff_damage_dealt+damage_taken.sum(),
    )
    return state


def update_plants_with_eat(state, plant_position, is_eating_plant):
    is_plant = jax.vmap(
        jnp.equal, in_axes=(0, None)
    )(
        plant_position, 
        state.growing_plants_positions
    ).all(axis=-1)
    plant_index = jnp.argmax(
        is_plant, axis=-1
    )
    selected_plant_age = jnp.where(
        is_eating_plant,
        0,
        state.growing_plants_age[plant_index]
    )
    return state.growing_plants_age.at[plant_index].set(selected_plant_age)


def add_new_growing_plant(growing_plant_positions, growing_plant_age, growing_plant_mask, position, is_placing_sapling):
    is_empty = jnp.logical_not(growing_plant_mask)
    plant_index = jnp.argmax(is_empty)
    is_an_empty_slot = is_empty.any()
    is_adding_plant = jnp.logical_and(is_an_empty_slot, is_placing_sapling)

    new_growing_plants_positions = jax.lax.select(
        is_adding_plant,
        growing_plant_positions.at[plant_index].set(position),
        growing_plant_positions,
    )
    new_growing_plants_age = jax.lax.select(
        is_adding_plant,
        growing_plant_age.at[plant_index].set(0),
        growing_plant_age,
    )
    new_growing_plants_mask = jax.lax.select(
        is_adding_plant,
        growing_plant_mask.at[plant_index].set(True),
        growing_plant_mask,
    )
    return new_growing_plants_positions, new_growing_plants_age, new_growing_plants_mask, is_adding_plant


def add_items_from_chest(rng, state, inventory, is_opening_chest):
    is_miner = state.player_specialization == Specialization.MINER.value
    is_warrior = state.player_specialization == Specialization.CRAFTER.value

    # Wood (60%)
    rng, _rng = jax.random.split(rng)
    is_looting_wood = jax.random.uniform(_rng) < 0.6 * is_opening_chest * is_miner
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
    is_crafter = state.player_specialization == Specialization.BUILDER.value

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
    is_warrior = state.player_specialization == Specialization.CRAFTER.value
    is_crafter = state.player_specialization == Specialization.BUILDER.value
    is_forager = state.player_specialization == Specialization.MINER.value  # MINER собирает саженцы и пьет воду

    block_position = state.player_position + DIRECTIONS[state.player_direction]
    equal_block_placement = (jnp.expand_dims(block_position, axis=1) == jnp.expand_dims(block_position, axis=0)).all(axis=2)

    doing_action = jnp.logical_and(
        in_bounds(block_position, static_params),
        action == Action.DO.value
    )

    # Only CRAFTER can perform melee attacks against mobs.
    doing_mob_attack = jnp.logical_and(doing_action, is_warrior)
    state, did_attack_mob, did_kill_mob = attack_mob(
        state,
        doing_mob_attack,
        block_position,
        get_player_damage_vector(state),
        is_warrior,
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
        wood=state.inventory.wood + MINER_MINED_RESOURCE_YIELD * is_mining_tree
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
        stone=new_inventory.stone + MINER_MINED_RESOURCE_YIELD * is_mining_stone
    )

    # Furnace
    is_block_furnace = (
        state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
        == BlockType.FURNACE.value
    )
    is_mining_furnace = jnp.logical_and(
        jnp.logical_and(is_block_furnace, is_crafter),
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
        jnp.logical_and(is_block_crafting_table, is_crafter),
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
        coal=new_inventory.coal + MINER_MINED_RESOURCE_YIELD * is_mining_coal
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
        iron=new_inventory.iron + MINER_MINED_RESOURCE_YIELD * is_mining_iron
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
        diamond=new_inventory.diamond + MINER_MINED_RESOURCE_YIELD * is_mining_diamond
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
        sapphire=new_inventory.sapphire + MINER_MINED_RESOURCE_YIELD * is_mining_sapphire
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
        ruby=new_inventory.ruby + MINER_MINED_RESOURCE_YIELD * is_mining_ruby
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
        sapling=new_inventory.sapling + MINER_MINED_RESOURCE_YIELD * is_mining_sapling
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
        stone=new_inventory.stone + MINER_MINED_RESOURCE_YIELD * is_mining_stalagmite
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
        jnp.logical_and(
            state.map[state.player_level, block_position[:, 0], block_position[:, 1]]
            == BlockType.NECROMANCER.value,
            is_warrior,
        ),
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
    is_warrior = state.player_specialization == Specialization.CRAFTER.value
    is_crafter = state.player_specialization == Specialization.BUILDER.value

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
        jnp.logical_and(
            is_player_placing_furnace[:, None],
            state.player_specialization == Specialization.BUILDER.value
        ),
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
        state.player_specialization == Specialization.BUILDER.value
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
            jnp.logical_and(
                torch_key_down, 
                state.player_specialization[player_index] == Specialization.BUILDER.value
            ),
            jnp.logical_and(
                is_valid_placement, 
                has_torch
            ),
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
            jnp.logical_and(
                is_valid_placement,
                state.player_specialization[player_index] == Specialization.BUILDER.value,
            ),
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
