import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from craftax_together_talens.util.game_logic_utils import *
from craftax_together_talens.util.maths_utils import *
from craftax_together_talens.constants import *
from craftax_together_talens.craftax_state import EnvState, StaticEnvParams, EnvParams
from craftax_together_talens.game_logic.constants import ALLOW_PLAYER_DESCEND


def move_player(state, actions, params, static_params):
    proposed_position = state.player_position + DIRECTIONS[actions]

    valid_move = is_position_in_bounds_not_in_mob_not_colliding(
        state, proposed_position, COLLISION_LAND_CREATURE, static_params
    ) 
    valid_move = jnp.logical_and(valid_move, is_position_not_colliding_other_player(state, proposed_position))
    valid_move = jnp.logical_or(valid_move, params.god_mode)

    position = state.player_position + jnp.expand_dims(valid_move, axis=1).astype(jnp.int32) * DIRECTIONS[actions]

    is_new_direction = jnp.sum(jnp.abs(DIRECTIONS[actions]), axis=1) != 0
    new_direction = (
        state.player_direction * (1 - is_new_direction) + actions * is_new_direction
    )

    state = state.replace(
        player_position=position,
        player_direction=new_direction,
    )

    return state


def update_plants(state, static_params):
    growing_plants_age = state.growing_plants_age + 1
    growing_plants_age *= state.growing_plants_mask

    finished_growing_plants = growing_plants_age >= 500

    new_plant_blocks = jnp.where(
        finished_growing_plants,
        BlockType.RIPE_PLANT.value,
        BlockType.PLANT.value,
    )

    def _set_plant_block(map, plant_index):
        new_block = jax.lax.select(
            finished_growing_plants[plant_index],
            new_plant_blocks[plant_index],
            map[
                state.growing_plants_positions[plant_index][0],
                state.growing_plants_positions[plant_index][1],
            ],
        )
        map = map.at[
            state.growing_plants_positions[plant_index][0],
            state.growing_plants_positions[plant_index][1],
        ].set(new_block)
        return map, None

    new_map, _ = jax.lax.scan(
        _set_plant_block,
        state.map[0],
        jnp.arange(static_params.max_growing_plants * static_params.player_count),
    )

    new_whole_map = state.map.at[0].set(new_map)

    state = state.replace(
        map=new_whole_map,
        growing_plants_age=growing_plants_age,
    )

    return state


def change_floor(
    state: EnvState, actions, env_params: EnvParams, static_params: StaticEnvParams
):
    descend_ok = jnp.array(ALLOW_PLAYER_DESCEND, dtype=jnp.bool_)
    is_moving_down = jnp.logical_and(
        descend_ok,
        jnp.logical_and(
            actions == Action.DESCEND.value,
            jnp.logical_or(
                env_params.god_mode,
                jnp.logical_and(
                    state.item_map[
                        state.player_level, state.player_position[:, 0], state.player_position[:, 1]
                    ]
                    == ItemType.LADDER_DOWN.value,
                    state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
                )
            )
        ),
    )
    is_moving_down = jnp.logical_and(
        is_moving_down,
        state.player_level < static_params.num_levels - 1
    )
    is_moving_down = is_moving_down.any()

    moving_down_position = state.up_ladders[state.player_level + 1]

    is_moving_up = jnp.logical_and(
        actions == Action.ASCEND.value,
        jnp.logical_or(
            env_params.god_mode,
            state.item_map[
                state.player_level, state.player_position[:, 0], state.player_position[:, 1]
            ]
            == ItemType.LADDER_UP.value
        )
    )
    is_moving_up = jnp.logical_and(
        is_moving_up,
        state.player_level > 0
    )
    is_moving_up = is_moving_up.any()
    
    moving_up_position = state.down_ladders[state.player_level - 1]
        
    # prioritizes moving players down levels if two players are conflicted
    position = jax.lax.select(is_moving_down, moving_down_position,
                              jax.lax.select(is_moving_up, moving_up_position, state.player_position))
    delta_floor = jax.lax.select(is_moving_down, 1,
                                 jax.lax.select(is_moving_up, -1, 0))
    
    move_down_achievement = LEVEL_ACHIEVEMENT_MAP[state.player_level + delta_floor]

    new_achievements = state.achievements.at[:, move_down_achievement].set(
        jnp.logical_or(
            (state.player_level + delta_floor) != 0,
            state.achievements[:, move_down_achievement],
        )
    )

    new_floor = jnp.logical_and(
        (state.player_level + delta_floor) != 0,
        jnp.logical_not(state.achievements[:, move_down_achievement]),
    )

    state = state.replace(
        player_level=state.player_level + delta_floor,
        player_position=position,
        achievements=new_achievements,
        player_xp=state.player_xp + 1 * new_floor,
    )

    return state


def shoot_projectile(state: EnvState, action: int, static_params: StaticEnvParams):
    # Arrow
    def _spawn_player_projectiles(projectile_info, player_index):
        player_projectiles, player_projectile_directions, player_projectile_owners = projectile_info

        is_shooting_arrow = jnp.logical_and(
            action[player_index] == Action.SHOOT_ARROW.value,
            jnp.logical_and(
                state.inventory.bow[player_index] >= 1,
                jnp.logical_and(
                    state.inventory.arrows[player_index] >= 1,
                    player_projectiles.mask[state.player_level].sum()
                    < (static_params.max_player_projectiles * static_params.player_count),
                ),
            ),
        )

        new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners = spawn_projectile(
            state, 
            static_params, 
            player_projectiles, 
            player_projectile_directions, 
            player_projectile_owners,
            state.player_position[player_index], 
            is_shooting_arrow,
            player_index,
            DIRECTIONS[state.player_direction[player_index]],
            ProjectileType.ARROW2.value,
        )

        return (new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners), is_shooting_arrow

    (new_player_projectiles, new_player_projectile_directions, new_player_projectile_owners), is_shooting_arrow = jax.lax.scan(
        _spawn_player_projectiles, 
        (state.player_projectiles, state.player_projectile_directions, state.player_projectile_owners),
        jnp.arange(static_params.player_count),
    )

    new_achievements = state.achievements.at[:, Achievement.FIRE_BOW.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.FIRE_BOW.value], is_shooting_arrow
        )
    )

    return state.replace(
        player_projectiles=new_player_projectiles,
        player_projectile_directions=new_player_projectile_directions,
        player_projectile_owners=new_player_projectile_owners,
        inventory=state.inventory.replace(
            arrows=state.inventory.arrows - 1 * is_shooting_arrow
        ),
        achievements=new_achievements,
    )


def read_book(state, action):
    is_reading_book = jnp.logical_and(
        action == Action.READ_BOOK.value, state.inventory.books > 0
    )
    new_spells = jnp.logical_or(state.learned_spells, is_reading_book)
    new_achievements = state.achievements.at[:, Achievement.LEARN_SPELL.value].set(
        jnp.logical_or(state.achievements[:, Achievement.LEARN_SPELL.value], is_reading_book)
    )

    return state.replace(
        inventory=state.inventory.replace(
            books=state.inventory.books - 1 * is_reading_book
        ),
        learned_spells=new_spells,
        achievements=new_achievements,
    )


def boss_logic(state, static_params):
    new_achievements = state.achievements.at[:, Achievement.DEFEAT_NECROMANCER.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.DEFEAT_NECROMANCER.value],
            has_beaten_boss(state, static_params),
        )
    )

    return state.replace(
        boss_timesteps_to_spawn_this_round=state.boss_timesteps_to_spawn_this_round
        - 1 * is_fighting_boss(state, static_params),
        achievements=new_achievements,
    )


def level_up_attributes(state: EnvState, action: jnp.array, params: EnvParams) -> EnvState:
    can_level_up = state.player_xp >= 1

    # Levelling up attributes
    is_levelling_up_dex = jnp.logical_and(
        can_level_up,
        jnp.logical_and(
            action == Action.LEVEL_UP_DEXTERITY.value,
            state.player_dexterity < params.max_attribute,
        ),
    )
    is_levelling_up_str = jnp.logical_and(
        can_level_up,
        jnp.logical_and(
            action == Action.LEVEL_UP_STRENGTH.value,
            state.player_strength < params.max_attribute,
        ),
    )
    is_levelling_up_int = jnp.logical_and(
        can_level_up,
        jnp.logical_and(
            action == Action.LEVEL_UP_INTELLIGENCE.value,
            state.player_intelligence < params.max_attribute,
        ),
    )
    is_levelling_up = jnp.logical_or(
        is_levelling_up_dex, jnp.logical_or(is_levelling_up_str, is_levelling_up_int)
    )

    return state.replace(
        player_dexterity=state.player_dexterity + 1 * is_levelling_up_dex,
        player_strength=state.player_strength + 1 * is_levelling_up_str,
        player_intelligence=state.player_intelligence + 1 * is_levelling_up_int,
        player_xp=state.player_xp - 1 * is_levelling_up,
    )


def drink_potion(state, action):
    drinking_potion_index = -1
    is_drinking_potion = False

    # Red
    is_drinking_red_potion = jnp.logical_and(
        action == Action.DRINK_POTION_RED.value, state.inventory.potions[:, 0] > 0
    )
    drinking_potion_index = (
        is_drinking_red_potion * 0
        + (1 - is_drinking_red_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_red_potion)

    # Green
    is_drinking_green_potion = jnp.logical_and(
        action == Action.DRINK_POTION_GREEN.value, state.inventory.potions[:, 1] > 0
    )
    drinking_potion_index = (
        is_drinking_green_potion * 1
        + (1 - is_drinking_green_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_green_potion)

    # Blue
    is_drinking_blue_potion = jnp.logical_and(
        action == Action.DRINK_POTION_BLUE.value, state.inventory.potions[:, 2] > 0
    )
    drinking_potion_index = (
        is_drinking_blue_potion * 2
        + (1 - is_drinking_blue_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_blue_potion)

    # Pink
    is_drinking_pink_potion = jnp.logical_and(
        action == Action.DRINK_POTION_PINK.value, state.inventory.potions[:, 3] > 0
    )
    drinking_potion_index = (
        is_drinking_pink_potion * 3
        + (1 - is_drinking_pink_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_pink_potion)

    # Cyan
    is_drinking_cyan_potion = jnp.logical_and(
        action == Action.DRINK_POTION_CYAN.value, state.inventory.potions[:, 4] > 0
    )
    drinking_potion_index = (
        is_drinking_cyan_potion * 4
        + (1 - is_drinking_cyan_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_cyan_potion)

    # Yellow
    is_drinking_yellow_potion = jnp.logical_and(
        action == Action.DRINK_POTION_YELLOW.value, state.inventory.potions[:, 5] > 0
    )
    drinking_potion_index = (
        is_drinking_yellow_potion * 5
        + (1 - is_drinking_yellow_potion) * drinking_potion_index
    )
    is_drinking_potion = jnp.logical_or(is_drinking_potion, is_drinking_yellow_potion)

    # Potion mapping
    potion_effect_index = state.potion_mapping[drinking_potion_index]

    # Potion effect
    delta_health = 0
    delta_health += is_drinking_potion * (potion_effect_index == 0) * 8
    delta_health += is_drinking_potion * (potion_effect_index == 1) * (-3)

    delta_mana = 0
    delta_mana += is_drinking_potion * (potion_effect_index == 2) * 8
    delta_mana += is_drinking_potion * (potion_effect_index == 3) * (-3)

    delta_energy = 0
    delta_energy += is_drinking_potion * (potion_effect_index == 4) * 8
    delta_energy += is_drinking_potion * (potion_effect_index == 5) * (-3)

    new_achievements = state.achievements.at[:, Achievement.DRINK_POTION.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.DRINK_POTION.value], is_drinking_potion
        )
    )

    return state.replace(
        inventory=state.inventory.replace(
            potions=state.inventory.potions.at[jnp.arange(state.inventory.potions.shape[0]), drinking_potion_index].set(
                state.inventory.potions[jnp.arange(state.inventory.potions.shape[0]), drinking_potion_index] - 1 * is_drinking_potion
            )
        ),
        player_health=state.player_health + delta_health,
        player_mana=state.player_mana + delta_mana,
        player_energy=state.player_energy + delta_energy,
        achievements=new_achievements,
    )


def calculate_inventory_achievements(state):
    # Some achievements (e.g. make_diamond_pickaxe) can be achieved in multiple ways (finding in chest or crafting)
    # Rather than duplicating achievement code, we simply look in the inventory for these types of achievements
    # at the end of each timestep
    # Wood
    achievements = state.achievements.at[:, Achievement.COLLECT_WOOD.value].set(
        jnp.logical_or(
            state.achievements[:, Achievement.COLLECT_WOOD.value], state.inventory.wood > 0
        )
    )
    # Stone
    achievements = achievements.at[:, Achievement.COLLECT_STONE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_STONE.value], state.inventory.stone > 0
        )
    )
    # Coal
    achievements = achievements.at[:, Achievement.COLLECT_COAL.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_COAL.value], state.inventory.coal > 0
        )
    )
    # Iron
    achievements = achievements.at[:, Achievement.COLLECT_IRON.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_IRON.value], state.inventory.iron > 0
        )
    )
    # Diamond
    achievements = achievements.at[:, Achievement.COLLECT_DIAMOND.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_DIAMOND.value], state.inventory.diamond > 0
        )
    )
    # Ruby
    achievements = achievements.at[:, Achievement.COLLECT_RUBY.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_RUBY.value], state.inventory.ruby > 0
        )
    )
    # Sapphire
    achievements = achievements.at[:, Achievement.COLLECT_SAPPHIRE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_SAPPHIRE.value],
            state.inventory.sapphire > 0,
        )
    )
    # Sapling
    achievements = achievements.at[:, Achievement.COLLECT_SAPLING.value].set(
        jnp.logical_or(
            achievements[:, Achievement.COLLECT_SAPLING.value], state.inventory.sapling > 0
        )
    )
    # Bow
    achievements = achievements.at[:, Achievement.FIND_BOW.value].set(
        jnp.logical_or(
            achievements[:, Achievement.FIND_BOW.value], state.inventory.bow > 0
        )
    )
    # Arrow
    achievements = achievements.at[:, Achievement.MAKE_ARROW.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_ARROW.value], state.inventory.arrows > 0
        )
    )
    # Torch
    achievements = achievements.at[:, Achievement.MAKE_TORCH.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_TORCH.value], state.inventory.torches > 0
        )
    )

    # Pickaxe
    achievements = achievements.at[:, Achievement.MAKE_WOOD_PICKAXE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_WOOD_PICKAXE.value],
            state.inventory.pickaxe >= 1,
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_STONE_PICKAXE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_STONE_PICKAXE.value],
            state.inventory.pickaxe >= 2,
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_IRON_PICKAXE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_IRON_PICKAXE.value],
            state.inventory.pickaxe >= 3,
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_DIAMOND_PICKAXE.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_DIAMOND_PICKAXE.value],
            state.inventory.pickaxe >= 4,
        )
    )

    # Sword
    achievements = achievements.at[:, Achievement.MAKE_WOOD_SWORD.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_WOOD_SWORD.value], state.inventory.sword >= 1
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_STONE_SWORD.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_STONE_SWORD.value], state.inventory.sword >= 2
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_IRON_SWORD.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_IRON_SWORD.value], state.inventory.sword >= 3
        )
    )
    achievements = achievements.at[:, Achievement.MAKE_DIAMOND_SWORD.value].set(
        jnp.logical_or(
            achievements[:, Achievement.MAKE_DIAMOND_SWORD.value],
            state.inventory.sword >= 4,
        )
    )

    return state.replace(achievements=achievements)


def update_player_intrinsics(state, action, static_params):
    # Start sleeping?
    is_starting_sleep = jnp.logical_and(
        action == Action.SLEEP.value, state.player_energy < get_max_energy(state)
    )
    new_is_sleeping = jnp.logical_or(state.is_sleeping, is_starting_sleep)
    state = state.replace(
        is_sleeping=jnp.where(state.player_alive, new_is_sleeping, state.is_sleeping)
    )

    # Wake up?
    is_waking_up = jnp.logical_and(
        state.player_energy >= get_max_energy(state), state.is_sleeping
    )
    new_is_sleeping = jnp.logical_and(state.is_sleeping, jnp.logical_not(is_waking_up))
    new_achievements = state.achievements.at[:, Achievement.WAKE_UP.value].set(
        jnp.logical_or(state.achievements[:, Achievement.WAKE_UP.value], is_waking_up)
    )
    state = state.replace(
        is_sleeping=jnp.where(state.player_alive, new_is_sleeping, state.is_sleeping),
        achievements=jnp.where(state.player_alive[:, None], new_achievements, state.achievements),
    )

    # Start resting?
    is_starting_rest = jnp.logical_and(
        action == Action.REST.value, state.player_health < get_max_health(state)
    )
    new_is_resting = jnp.logical_or(state.is_resting, is_starting_rest)
    state = state.replace(is_resting=new_is_resting)

    # Wake up from resting
    is_waking_up = jnp.logical_and(
        state.is_resting,
        jnp.logical_or(
            state.player_health >= get_max_health(state),
            jnp.logical_or(state.player_food <= 0, state.player_drink <= 0),
        ),
    )
    new_is_resting = jnp.logical_and(state.is_resting, jnp.logical_not(is_waking_up))
    state = state.replace(
        is_resting=jnp.where(state.player_alive, new_is_resting, state.is_resting),
    )

    not_boss = jnp.logical_not(is_fighting_boss(state, static_params))

    intrinsic_decay_coeff = (1.0 - (0.125 * (state.player_dexterity - 1)))

    # Hunger
    hunger_add = jnp.where(
        state.is_sleeping, 
        0.5, 
        1.0,
    ) * intrinsic_decay_coeff
    new_hunger = state.player_hunger + hunger_add

    hungered_food = jnp.maximum(state.player_food - 1 * not_boss, 0)
    new_food = jnp.where(new_hunger > 25, hungered_food, state.player_food)
    new_hunger = jnp.where(
        new_hunger > 25, 
        0.0, 
        new_hunger
    )

    state = state.replace(
        player_hunger=jnp.where(state.player_alive, new_hunger, state.player_hunger),
        player_food=jnp.where(state.player_alive, new_food, state.player_food),
    )

    # Thirst
    thirst_add = jnp.where(
        state.is_sleeping, 
        0.5, 
        1.0,
    ) * intrinsic_decay_coeff
    new_thirst = state.player_thirst + thirst_add
    thirsted_drink = jnp.maximum(state.player_drink - 1 * not_boss, 0)
    new_drink = jnp.where(new_thirst > 20, thirsted_drink, state.player_drink)
    new_thirst = jnp.where(
        new_thirst > 20, 
        0.0, 
        new_thirst
    )

    state = state.replace(
        player_thirst=jnp.where(state.player_alive, new_thirst, state.player_thirst),
        player_drink=jnp.where(state.player_alive, new_drink, state.player_drink),
    )

    # Fatigue
    new_fatigue = jnp.where(
        state.is_sleeping,
        jnp.minimum(state.player_fatigue - 1, 0),
        state.player_fatigue + intrinsic_decay_coeff,
    )

    new_energy = jnp.where(
        new_fatigue > 30,
        jnp.maximum(state.player_energy - 1 * not_boss, 0),
        state.player_energy,
    )
    new_fatigue = jnp.where(
        new_fatigue > 30, 
        0.0, 
        new_fatigue
    )

    new_energy = jnp.where(
        new_fatigue < -10,
        jnp.minimum(state.player_energy + 1, get_max_energy(state)),
        new_energy,
    )
    new_fatigue = jnp.where(
        new_fatigue < -10, 
        0.0, 
        new_fatigue
    )

    state = state.replace(
        player_fatigue=jnp.where(state.player_alive, new_fatigue, state.player_fatigue),
        player_energy=jnp.where(state.player_alive, new_energy, state.player_energy),
    )

    # Health
    necessities = jnp.stack(
        [
            state.player_food > 0,
            state.player_drink > 0,
            jnp.logical_or(state.player_energy > 0, state.is_sleeping)
        ],
        axis=1
    )

    all_necessities = necessities.all(axis=1)
    # all_necessities = jnp.full((static_params.player_count,), True, dtype=bool)

    new_all_necessities_frac = (state.all_necessities_frac * state.timestep + all_necessities)/(state.timestep+1)

    recover_all = jnp.where(
        state.is_sleeping, 
        2.0, 
        1.0,
    )
    recover_not_all = jnp.where(
        state.is_sleeping, 
        -0.5, 
        -1.0,
    ) * not_boss
    recover_add = jnp.where(all_necessities, recover_all, recover_not_all)

    new_recover = state.player_recover + recover_add

    recovered_health = jnp.minimum(state.player_health + 2, get_max_health(state))
    derecovered_health = state.player_health - 1

    new_health = jnp.where(new_recover > 25, recovered_health, state.player_health)
    new_recover = jnp.where(
        new_recover > 25, 
        0.0, 
        new_recover
    )
    new_health = jnp.where(new_recover < -15, derecovered_health, new_health)
    new_recover = jnp.where(
        new_recover < -15, 
        0.0, 
        new_recover
    )

    state = state.replace(
        player_recover=jnp.where(state.player_alive, new_recover, state.player_recover),
        player_health=jnp.where(state.player_alive, new_health, state.player_health),
    )

    # Mana
    mana_recover_coeff = 1 + 0.25 * (state.player_intelligence - 1)
    new_recover_mana = (
        jnp.where(
            state.is_sleeping,
            state.player_recover_mana + 2,
            state.player_recover_mana + 1,
        )
        * mana_recover_coeff
    )

    new_mana = jnp.where(
        new_recover_mana > 30, state.player_mana + 1, state.player_mana
    )
    new_recover_mana = jnp.where(
        new_recover_mana > 30, 
        0.0, 
        new_recover_mana
    )

    state = state.replace(
        player_recover_mana=jnp.where(state.player_alive, new_recover_mana, state.player_recover_mana),
        player_mana=jnp.where(state.player_alive, new_mana, state.player_mana),
        all_necessities_frac=new_all_necessities_frac
    )

    return state


def update_mobs(rng, state, params, env_params, static_params):

    # Move melee_mobs
    def _move_melee_mob(rng_and_state, melee_mob_index):
        rng, state = rng_and_state
        melee_mobs = state.melee_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        valid_random_moves = in_bounds(
            DIRECTIONS[1:5] + melee_mobs.position[state.player_level, melee_mob_index], 
            static_params
        )
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:5],
            p=valid_random_moves
        )
        random_move_proposed_position = (
            melee_mobs.position[state.player_level, melee_mob_index]
            + random_move_direction
        )

        # Move towards closest player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        all_players_move_direction_abs = jnp.abs(
            state.player_position
            - melee_mobs.position[state.player_level, melee_mob_index]
        )
        distance_to_players = all_players_move_direction_abs.sum(axis=1)
        player_targetted = jnp.argmin(jnp.where(
            state.player_alive,
            distance_to_players,
            jnp.inf
        ))
        player_move_direction_abs = all_players_move_direction_abs[player_targetted]

        player_move_direction_index_p = (
            player_move_direction_abs == player_move_direction_abs.max()
        ) / player_move_direction_abs.sum()
        rng, _rng = jax.random.split(rng)
        player_move_direction_index = jax.random.choice(
            _rng,
            jnp.arange(2),
            p=player_move_direction_index_p,
        )

        player_move_direction = player_move_direction.at[
            player_move_direction_index
        ].set(
            jnp.sign(
                state.player_position[player_targetted, player_move_direction_index]
                - melee_mobs.position[state.player_level, melee_mob_index, player_move_direction_index]
            ).astype(jnp.int32)
        )
        player_move_proposed_position = (
            melee_mobs.position[state.player_level, melee_mob_index]
            + player_move_direction
        )

        # Choose movement
        close_to_player = distance_to_players < 10
        close_to_player = jnp.logical_and(
            close_to_player,
            state.player_alive
        ).any()
        close_to_player = jnp.logical_or(
            close_to_player, is_fighting_boss(state, static_params)
        )
        rng, _rng = jax.random.split(rng)
        close_to_player = jnp.logical_and(
            close_to_player, jax.random.uniform(_rng) < 0.75
        )
        proposed_position = jax.lax.select(
            close_to_player,
            player_move_proposed_position,
            random_move_proposed_position,
        )

        # Choose attack or not
        is_attacking_player = distance_to_players == 1
        is_attacking_player = jnp.logical_and(
            is_attacking_player,
            state.player_alive
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player,
            melee_mobs.attack_cooldown[state.player_level, melee_mob_index] <= 0,
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player, melee_mobs.mask[state.player_level, melee_mob_index]
        )

        proposed_position = jax.lax.select(
            is_attacking_player.any(),
            melee_mobs.position[state.player_level, melee_mob_index],
            proposed_position,
        )

        melee_mob_base_damage = MOB_TYPE_DAMAGE_MAPPING[
            melee_mobs.type_id[state.player_level, melee_mob_index], MobType.MELEE.value
        ]

        melee_mob_damage = get_damage_done_to_player(
            state, static_params, melee_mob_base_damage * (1 + 2.5 * state.is_sleeping[:, None])
        )

        new_cooldown = jnp.where(
            is_attacking_player.any(),
            5,
            melee_mobs.attack_cooldown[state.player_level, melee_mob_index] - 1,
        )

        is_waking_player = jnp.logical_and(state.is_sleeping, is_attacking_player)

        state = state.replace(
            player_health=state.player_health - melee_mob_damage * is_attacking_player,
            is_sleeping=jnp.logical_and(
                state.is_sleeping, jnp.logical_not(is_attacking_player)
            ),
            is_resting=jnp.logical_and(
                state.is_resting, jnp.logical_not(is_attacking_player)
            ),
            achievements=state.achievements.at[:, Achievement.WAKE_UP.value].set(
                jnp.logical_or(
                    state.achievements[:, Achievement.WAKE_UP.value], is_waking_player
                )
            ),
        )

        mob_type = melee_mobs.type_id[state.player_level, melee_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 1]
        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position[None, :], collision_map, static_params
        )[0]
        in_other_player = is_in_other_player(state, proposed_position[None, :])[0]
        valid_move = jnp.logical_and(
            valid_move,
            jnp.logical_not(in_other_player)
        )

        
        position = jax.lax.select(
            valid_move,
            proposed_position,
            melee_mobs.position[state.player_level, melee_mob_index],
        )

        should_not_despawn = distance_to_players < params.mob_despawn_distance
        should_not_despawn = jnp.logical_and(
            should_not_despawn,
            state.player_alive
        ).any()
        should_not_despawn = jnp.logical_or(
            should_not_despawn, is_fighting_boss(state, static_params)
        )

        rng, _rng = jax.random.split(rng)

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.player_level,
            state.melee_mobs.position[state.player_level, melee_mob_index, 0],
            state.melee_mobs.position[state.player_level, melee_mob_index, 1],
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.player_level,
                    state.melee_mobs.position[state.player_level, melee_mob_index, 0],
                    state.melee_mobs.position[state.player_level, melee_mob_index, 1],
                ],
                jnp.logical_not(melee_mobs.mask[state.player_level, melee_mob_index]),
            )
        )
        new_mask = jnp.logical_and(
            state.melee_mobs.mask[state.player_level, melee_mob_index],
            should_not_despawn,
        )
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
            jnp.logical_or(
                new_mob_map[state.player_level, position[0], position[1]], new_mask
            )
        )

        state = state.replace(
            melee_mobs=state.melee_mobs.replace(
                position=state.melee_mobs.position.at[
                    state.player_level, melee_mob_index
                ].set(position),
                attack_cooldown=state.melee_mobs.attack_cooldown.at[
                    state.player_level, melee_mob_index
                ].set(new_cooldown),
                mask=state.melee_mobs.mask.at[state.player_level, melee_mob_index].set(
                    new_mask
                ),
            ),
            mob_map=new_mob_map,
        )

        return (_rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, state), _ = jax.lax.scan(
        _move_melee_mob, (rng, state), jnp.arange(static_params.max_melee_mobs * static_params.player_count)
    )

    # Move passive_mobs
    def _move_passive_mob(rng_and_state, passive_mob_index):
        rng, state = rng_and_state
        passive_mobs = state.passive_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        valid_random_moves = in_bounds(
            DIRECTIONS[1:9] + passive_mobs.position[state.player_level, passive_mob_index], 
            static_params
        )
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:9],  # 50% chance of not moving
            p=valid_random_moves
        )
        proposed_position = (
            passive_mobs.position[state.player_level, passive_mob_index]
            + random_move_direction
        )

        mob_type = passive_mobs.type_id[state.player_level, passive_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 0]
        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position[None, :], collision_map, static_params
        )[0]
        in_other_player = is_in_other_player(state, proposed_position[None, :])[0]
        valid_move = jnp.logical_and(
            valid_move,
            jnp.logical_not(in_other_player)
        )
        position = jax.lax.select(
            valid_move,
            proposed_position,
            passive_mobs.position[state.player_level, passive_mob_index],
        )

        distance_to_players = jnp.abs(
            state.player_position
            - passive_mobs.position[state.player_level, passive_mob_index]
        ).sum(axis=1)
        should_not_despawn = jnp.logical_and(
            distance_to_players < params.mob_despawn_distance,
            state.player_alive
        ).any()

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.player_level,
            state.passive_mobs.position[state.player_level, passive_mob_index, 0],
            state.passive_mobs.position[state.player_level, passive_mob_index, 1],
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.player_level,
                    state.passive_mobs.position[
                        state.player_level, passive_mob_index, 0
                    ],
                    state.passive_mobs.position[
                        state.player_level, passive_mob_index, 1
                    ],
                ],
                jnp.logical_not(
                    passive_mobs.mask[state.player_level, passive_mob_index]
                ),
            )
        )
        new_mask = jnp.logical_and(
            state.passive_mobs.mask[state.player_level, passive_mob_index],
            should_not_despawn,
        )
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
            jnp.logical_or(
                new_mob_map[state.player_level, position[0], position[1]], new_mask
            )
        )

        state = state.replace(
            passive_mobs=state.passive_mobs.replace(
                position=state.passive_mobs.position.at[
                    state.player_level, passive_mob_index
                ].set(position),
                mask=state.passive_mobs.mask.at[
                    state.player_level, passive_mob_index
                ].set(
                    jnp.logical_and(
                        state.passive_mobs.mask[state.player_level, passive_mob_index],
                        should_not_despawn,
                    )
                ),
            ),
            mob_map=new_mob_map,
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, state), _ = jax.lax.scan(
        _move_passive_mob, (rng, state), jnp.arange(static_params.max_passive_mobs * static_params.player_count)
    )

    # Move ranged_mobs

    def _move_ranged_mob(rng_and_state, ranged_mob_index):
        rng, state = rng_and_state
        ranged_mobs = state.ranged_mobs

        # Random move
        rng, _rng = jax.random.split(rng)
        valid_random_moves = in_bounds(
            DIRECTIONS[1:5] + ranged_mobs.position[state.player_level, ranged_mob_index], 
            static_params
        )
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:5],
            p=valid_random_moves
        )
        random_move_proposed_position = (
            ranged_mobs.position[state.player_level, ranged_mob_index]
            + random_move_direction
        )

        # Move towards closest player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        all_players_move_direction_abs = jnp.abs(
            state.player_position
            - ranged_mobs.position[state.player_level, ranged_mob_index]
        )
        distance_to_players = all_players_move_direction_abs.sum(axis=1)
        player_targetted = jnp.argmin(jnp.where(
            state.player_alive,
            distance_to_players,
            jnp.inf
        ))
        player_move_direction_abs = all_players_move_direction_abs[player_targetted]
        player_move_direction_index_p = (
            player_move_direction_abs == player_move_direction_abs.max()
        ) / player_move_direction_abs.sum()
        rng, _rng = jax.random.split(rng)
        player_move_direction_index = jax.random.choice(
            _rng,
            jnp.arange(2),
            p=player_move_direction_index_p,
        )

        player_move_direction = player_move_direction.at[
            player_move_direction_index
        ].set(
            jnp.sign(
                state.player_position[player_targetted, player_move_direction_index]
                - ranged_mobs.position[state.player_level, ranged_mob_index, player_move_direction_index]
            ).astype(jnp.int32)
        )
        player_move_towards_proposed_position = (
            ranged_mobs.position[state.player_level, ranged_mob_index]
            + player_move_direction
        )
        player_move_away_proposed_position = (
            ranged_mobs.position[state.player_level, ranged_mob_index]
            - player_move_direction
        )

        # Choose movement
        far_from_player = player_move_direction_abs[player_move_direction_index] >= 6
        too_close_to_player = player_move_direction_abs[player_move_direction_index] <= 3

        proposed_position = jax.lax.select(
            far_from_player,
            player_move_towards_proposed_position,
            random_move_proposed_position,
        )
        proposed_position = jax.lax.select(
            too_close_to_player,
            player_move_away_proposed_position,
            proposed_position,
        )

        rng, _rng = jax.random.split(rng)

        proposed_position = jax.lax.select(
            jax.random.uniform(_rng) < 0.15,
            proposed_position,
            random_move_proposed_position,
        )

        # Choose attack or not
        is_attacking_player = jnp.logical_not(far_from_player)
        is_attacking_player = jnp.logical_and(
            is_attacking_player,
            ranged_mobs.attack_cooldown[state.player_level, ranged_mob_index] <= 0,
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player, ranged_mobs.mask[state.player_level, ranged_mob_index]
        )

        # Spawn projectile
        can_spawn_projectile = (
            state.mob_projectiles.mask[state.player_level].sum()
            < static_params.max_mob_projectiles * static_params.player_count
        )
        new_projectile_position = ranged_mobs.position[
            state.player_level, ranged_mob_index
        ]

        is_spawning_projectile = jnp.logical_and(
            is_attacking_player, can_spawn_projectile
        )

        new_mob_projectiles, new_mob_projectile_directions, new_mob_projectile_owners = spawn_projectile(
            state,
            static_params,
            state.mob_projectiles,
            state.mob_projectile_directions,
            state.mob_projectile_owners,
            new_projectile_position,
            is_spawning_projectile,
            ranged_mob_index,
            player_move_direction,
            RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING[
                ranged_mobs.type_id[state.player_level, ranged_mob_index]
            ],
        )

        state = state.replace(
            mob_projectiles=new_mob_projectiles,
            mob_projectile_directions=new_mob_projectile_directions,
            mob_projectile_owners=new_mob_projectile_owners,
        )

        proposed_position = jax.lax.select(
            is_attacking_player,
            ranged_mobs.position[state.player_level, ranged_mob_index],
            proposed_position,
        )

        new_cooldown = jax.lax.select(
            is_attacking_player,
            4,
            ranged_mobs.attack_cooldown[state.player_level, ranged_mob_index] - 1,
        )

        mob_type = ranged_mobs.type_id[state.player_level, ranged_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 2]
        valid_move = is_position_in_bounds_not_in_mob_not_colliding(
            state, proposed_position[None, :], collision_map, static_params
        )[0]
        in_other_player = is_in_other_player(state, proposed_position[None, :])[0]
        valid_move = jnp.logical_and(
            valid_move,
            jnp.logical_not(in_other_player)
        )

        position = jax.lax.select(
            valid_move,
            proposed_position,
            ranged_mobs.position[state.player_level, ranged_mob_index],
        )

        should_not_despawn = distance_to_players < params.mob_despawn_distance
        should_not_despawn = jnp.logical_and(
            should_not_despawn,
            state.player_alive
        ).any()
        should_not_despawn = jnp.logical_or(
            should_not_despawn, is_fighting_boss(state, static_params)
        )

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.player_level,
            state.ranged_mobs.position[state.player_level, ranged_mob_index, 0],
            state.ranged_mobs.position[state.player_level, ranged_mob_index, 1],
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.player_level,
                    state.ranged_mobs.position[state.player_level, ranged_mob_index, 0],
                    state.ranged_mobs.position[state.player_level, ranged_mob_index, 1],
                ],
                jnp.logical_not(ranged_mobs.mask[state.player_level, ranged_mob_index]),
            )
        )
        new_mask = jnp.logical_and(
            state.ranged_mobs.mask[state.player_level, ranged_mob_index],
            should_not_despawn,
        )
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[state.player_level, position[0], position[1]].set(
            jnp.logical_or(
                new_mob_map[state.player_level, position[0], position[1]], new_mask
            )
        )

        state = state.replace(
            ranged_mobs=state.ranged_mobs.replace(
                position=state.ranged_mobs.position.at[
                    state.player_level, ranged_mob_index
                ].set(position),
                attack_cooldown=state.ranged_mobs.attack_cooldown.at[
                    state.player_level, ranged_mob_index
                ].set(new_cooldown),
                mask=state.ranged_mobs.mask.at[
                    state.player_level, ranged_mob_index
                ].set(
                    jnp.logical_and(
                        state.ranged_mobs.mask[state.player_level, ranged_mob_index],
                        should_not_despawn,
                    )
                ),
            ),
            mob_map=new_mob_map,
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, state), _ = jax.lax.scan(
        _move_ranged_mob, (rng, state), jnp.arange(static_params.max_ranged_mobs * static_params.player_count)
    )

    # Move projectiles
    def _move_mob_projectile(rng_and_state, projectile_index):
        rng, state = rng_and_state
        projectiles = state.mob_projectiles

        proposed_position = (
            projectiles.position[state.player_level, projectile_index]
            + state.mob_projectile_directions[state.player_level, projectile_index]
        )


        proposed_position_in_bounds = in_bounds(proposed_position[None, :], static_params)[0]
        in_wall = is_in_solid_block(state.map[state.player_level], proposed_position[None, :])[0]
        in_wall = jnp.logical_and(
            in_wall,
            jnp.logical_not(
                state.map[state.player_level][
                    proposed_position[0], proposed_position[1]
                ]
                == BlockType.WATER.value
            ),
        )  # Arrows can go over water
        in_mob = is_in_mob(state, proposed_position[None, :])[0]

        continue_move = jnp.logical_and(
            proposed_position_in_bounds, jnp.logical_not(in_wall)
        )
        continue_move = jnp.logical_and(continue_move, jnp.logical_not(in_mob))

        hit_player0 = jnp.logical_and(
            (
                projectiles.position[state.player_level, projectile_index]
                == state.player_position
            ).all(axis=1),
            projectiles.mask[state.player_level, projectile_index],
        )

        proposed_position_in_player = (proposed_position == state.player_position).all(axis=1)
        hit_player1 = jnp.logical_and(
            proposed_position_in_player,
            projectiles.mask[state.player_level, projectile_index],
        )
        hit_player = jnp.logical_or(hit_player0, hit_player1)
        hit_player = jnp.logical_and(hit_player, state.player_alive)

        continue_move = jnp.logical_and(continue_move, jnp.logical_not(hit_player.any()))

        position = proposed_position

        # Clear our old entry if we are alive
        new_mask = jnp.logical_and(
            continue_move, projectiles.mask[state.player_level, projectile_index]
        )

        hit_bench_or_furnace = jnp.logical_or(
            state.map[state.player_level, position[0], position[1]]
            == BlockType.FURNACE.value,
            state.map[state.player_level, position[0], position[1]]
            == BlockType.CRAFTING_TABLE.value,
        )
        removing_block = jnp.logical_and(
            hit_bench_or_furnace, projectiles.mask[state.player_level, projectile_index]
        )

        new_block = jax.lax.select(
            removing_block,
            BlockType.PATH.value,
            state.map[state.player_level, position[0], position[1]],
        )

        projectile_type = state.mob_projectiles.type_id[
            state.player_level, projectile_index
        ]
        projectile_damage = get_damage_done_to_player(
            state,
            static_params,
            MOB_TYPE_DAMAGE_MAPPING[projectile_type, MobType.PROJECTILE.value][None, :],
        )

        state = state.replace(
            mob_projectiles=state.mob_projectiles.replace(
                position=state.mob_projectiles.position.at[
                    state.player_level, projectile_index
                ].set(position),
                mask=state.mob_projectiles.mask.at[
                    state.player_level, projectile_index
                ].set(new_mask),
            ),
            player_health=state.player_health - projectile_damage * hit_player,
            is_sleeping=jnp.logical_and(state.is_sleeping, jnp.logical_not(hit_player)),
            is_resting=jnp.logical_and(state.is_resting, jnp.logical_not(hit_player)),
            map=state.map.at[state.player_level, position[0], position[1]].set(
                new_block
            ),
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, state), _ = jax.lax.scan(
        _move_mob_projectile,
        (rng, state),
        jnp.arange(static_params.max_mob_projectiles * static_params.player_count),
    )

    def _move_player_projectile(rng_and_state, projectile_index):
        rng, state = rng_and_state
        projectiles = state.player_projectiles

        projectile_owner = state.player_projectile_owners[
            state.player_level, projectile_index
        ]

        projectile_type = state.player_projectiles.type_id[
            state.player_level, projectile_index
        ]

        projectile_damage_vector = (
            MOB_TYPE_DAMAGE_MAPPING[projectile_type, MobType.PROJECTILE.value]
            * projectiles.mask[state.player_level, projectile_index]
        )

        is_arrow = jnp.logical_or(
            projectile_type == ProjectileType.ARROW.value,
            projectile_type == ProjectileType.ARROW2.value,
        )

        # Bow enchantment
        arrow_damage_add = jnp.zeros(3, dtype=jnp.float32)
        arrow_damage_add = arrow_damage_add.at[state.bow_enchantment[projectile_owner]].set(
            projectile_damage_vector[0] / 2
        )
        arrow_damage_add = arrow_damage_add.at[0].set(0)

        projectile_damage_vector += jax.lax.select(
            is_arrow,
            arrow_damage_add,
            jnp.zeros(3, dtype=jnp.float32),
        )

        # Apply attribute scaling
        arrow_damage_coeff = 1 + 0.2 * (state.player_dexterity[projectile_owner] - 1)
        magic_damage_coeff = 1 + 0.5 * (state.player_intelligence[projectile_owner] - 1)

        projectile_damage_vector *= jax.lax.select(
            is_arrow,
            arrow_damage_coeff,
            1.0,
        )

        projectile_damage_vector *= jax.lax.select(
            projectile_type == ProjectileType.FIREBALL.value,
            magic_damage_coeff,
            1.0,
        )

        proposed_position = (
            projectiles.position[state.player_level, projectile_index]
            + state.player_projectile_directions[state.player_level, projectile_index]
        )
        
        proposed_position_in_bounds = in_bounds(proposed_position[None, :], static_params)[0]
        in_wall = is_in_solid_block(state.map[state.player_level], proposed_position[None, :])[0]
        in_wall = jnp.logical_and(
            in_wall,
            jnp.logical_not(
                state.map[state.player_level][
                    proposed_position[0], proposed_position[1]
                ]
                == BlockType.WATER.value
            ),
        )  # Arrows can go over water

        # Check if we hit a player
        deal_damage = projectiles.mask[state.player_level, projectile_index]

        per_player_contact = (state.player_position == proposed_position[None, :]).all(axis=-1)
        did_attack_player = per_player_contact.any()    
        player_attack_index = jnp.argmax(per_player_contact)

        player_defense_vector = get_player_defense_vector(state)[player_attack_index]
        player_damage_dealt = get_damage(projectile_damage_vector, player_defense_vector) * did_attack_player * env_params.friendly_fire
        new_player_health = state.player_health.at[player_attack_index].add(
            -player_damage_dealt
        )
        
        state, did_attack_mob0, did_kill_mob0 = attack_mob(
            state,
            deal_damage,
            projectiles.position[None, state.player_level, projectile_index],
            projectile_damage_vector[None, :],
            jnp.array([False]),
        )
        did_attack_mob0 = did_attack_mob0[0]

        did_attack_mob = jnp.logical_or(did_attack_player, did_attack_mob0)

        projectile_damage_vector = projectile_damage_vector * (1 - did_attack_mob0)

        state, did_attack_mob1, did_kill_mob1 = attack_mob(
            state,
            deal_damage,
            proposed_position[None, :],
            projectile_damage_vector[None, :],
            jnp.array([False])
        )
        did_attack_mob1 = did_attack_mob1[0]

        did_attack_mob = jnp.logical_or(did_attack_mob, did_attack_mob1)

        continue_move = jnp.logical_and(
            proposed_position_in_bounds, jnp.logical_not(in_wall)
        )
        continue_move = jnp.logical_and(continue_move, jnp.logical_not(did_attack_mob))
        position = proposed_position

        # Clear our old entry if we are alive
        new_mask = jnp.logical_and(
            continue_move, projectiles.mask[state.player_level, projectile_index]
        )

        state = state.replace(
            player_health=new_player_health,
            player_projectiles=state.player_projectiles.replace(
                position=state.player_projectiles.position.at[
                    state.player_level, projectile_index
                ].set(position),
                mask=state.player_projectiles.mask.at[
                    state.player_level, projectile_index
                ].set(new_mask),
            ),
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, state), _ = jax.lax.scan(
        _move_player_projectile,
        (rng, state),
        jnp.arange(static_params.max_player_projectiles * static_params.player_count),
    )

    return state


def spawn_mobs(state, rng, params, static_params):
    player_distance_map = get_all_players_distance_map(
        state.player_position, state.player_alive, static_params
    )
    grave_map = jnp.logical_or(
        state.map[state.player_level] == BlockType.GRAVE.value,
        jnp.logical_or(
            state.map[state.player_level] == BlockType.GRAVE2.value,
            state.map[state.player_level] == BlockType.GRAVE3.value,
        ),
    )

    floor_mob_spawn_chance = FLOOR_MOB_SPAWN_CHANCE * static_params.player_count
    monster_spawn_coeff = (
        1
        + (state.monsters_killed[state.player_level] < MONSTERS_KILLED_TO_CLEAR_LEVEL)
        * 2
    )  # Triple spawn rate if we are on an uncleared level

    monster_spawn_coeff *= jax.lax.select(
        is_fighting_boss(state, static_params),
        is_boss_spawn_wave(state, static_params) * 1000,
        1,
    )

    # Passive mobs
    can_spawn_passive_mob = (
        state.passive_mobs.mask[state.player_level].sum()
        < static_params.max_passive_mobs * static_params.player_count
    )

    rng, _rng = jax.random.split(rng)
    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob,
        jax.random.uniform(_rng) < floor_mob_spawn_chance[state.player_level, 0],
    )

    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob, jnp.logical_not(is_fighting_boss(state, static_params))
    )

    all_valid_blocks_map = jnp.logical_or(
        state.map[state.player_level] == BlockType.GRASS.value,
        jnp.logical_or(
            state.map[state.player_level] == BlockType.PATH.value,
            jnp.logical_or(
                state.map[state.player_level] == BlockType.FIRE_GRASS.value,
                state.map[state.player_level] == BlockType.ICE_GRASS.value,
            ),
        ),
    )
    new_passive_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.PASSIVE.value]

    passive_mobs_can_spawn_map = all_valid_blocks_map

    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, player_distance_map > 3
    )
    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
    )

    # To avoid spawning mobs ontop of dead players
    passive_mobs_can_spawn_map = passive_mobs_can_spawn_map.at[
        state.player_position[:, 0], state.player_position[:, 1]
    ].set(False)

    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob, passive_mobs_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    passive_mob_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(passive_mobs_can_spawn_map, -1)
        / jnp.sum(passive_mobs_can_spawn_map),
    )
    passive_mob_position = jnp.array(
        [
            passive_mob_position // static_params.map_size[0],
            passive_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_passive_mob_index = jnp.argmax(
        jnp.logical_not(state.passive_mobs.mask[state.player_level])
    )

    new_passive_mob_position = jax.lax.select(
        can_spawn_passive_mob,
        passive_mob_position,
        state.passive_mobs.position[state.player_level, new_passive_mob_index],
    )

    new_passive_mob_health = jax.lax.select(
        can_spawn_passive_mob,
        MOB_TYPE_HEALTH_MAPPING[new_passive_mob_type, MobType.PASSIVE.value],
        state.passive_mobs.health[state.player_level, new_passive_mob_index],
    )

    new_passive_mob_mask = jax.lax.select(
        can_spawn_passive_mob,
        True,
        state.passive_mobs.mask[state.player_level, new_passive_mob_index],
    )

    passive_mobs = Mobs(
        position=state.passive_mobs.position.at[
            state.player_level, new_passive_mob_index
        ].set(new_passive_mob_position),
        health=state.passive_mobs.health.at[
            state.player_level, new_passive_mob_index
        ].set(new_passive_mob_health),
        mask=state.passive_mobs.mask.at[state.player_level, new_passive_mob_index].set(
            new_passive_mob_mask
        ),
        attack_cooldown=state.passive_mobs.attack_cooldown,
        type_id=state.passive_mobs.type_id.at[
            state.player_level, new_passive_mob_index
        ].set(new_passive_mob_type),
    )

    state = state.replace(
        passive_mobs=passive_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_passive_mob_position[0], new_passive_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_passive_mob_position[0],
                    new_passive_mob_position[1],
                ],
                new_passive_mob_mask,
            )
        ),
    )

    # Monsters
    DUNGEONS = jnp.array([1, 3, 4])
    in_dungeon = (state.player_level == DUNGEONS).any()

    monsters_can_spawn_player_range_map = player_distance_map > 9
    monsters_can_spawn_player_range_map_boss = player_distance_map <= 6

    monsters_can_spawn_player_range_map = jax.lax.select(
        is_fighting_boss(state, static_params),
        monsters_can_spawn_player_range_map_boss,
        monsters_can_spawn_player_range_map,
    )

    # Melee mobs
    can_spawn_melee_mob = (
        state.melee_mobs.mask[state.player_level].sum() < 
        (
            static_params.max_melee_mobs * 
            (
                static_params.player_count * (1 - in_dungeon) + 
                1 * in_dungeon # reduce number of mobs if in dungeons to avoid crowdedness
            )
        )
    )

    new_melee_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.MELEE.value]
    new_melee_mob_type_boss = FLOOR_MOB_MAPPING[
        state.boss_progress, MobType.MELEE.value
    ]

    new_melee_mob_type = jax.lax.select(
        is_fighting_boss(state, static_params),
        new_melee_mob_type_boss,
        new_melee_mob_type,
    )

    rng, _rng = jax.random.split(rng)
    melee_mob_spawn_chance = floor_mob_spawn_chance[
        state.player_level, 1
    ] + floor_mob_spawn_chance[state.player_level, 3] * jnp.square(
        1 - state.light_level
    )
    can_spawn_melee_mob = jnp.logical_and(
        can_spawn_melee_mob,
        jax.random.uniform(_rng) < melee_mob_spawn_chance * monster_spawn_coeff,
    )

    melee_mobs_can_spawn_map = jax.lax.select(
        is_fighting_boss(state, static_params), grave_map, all_valid_blocks_map
    )

    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, monsters_can_spawn_player_range_map
    )
    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
    )
    melee_mobs_can_spawn_map = melee_mobs_can_spawn_map.at[
        state.player_position[:, 0], state.player_position[:, 1]
    ].set(False)

    can_spawn_melee_mob = jnp.logical_and(
        can_spawn_melee_mob, melee_mobs_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    melee_mob_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(melee_mobs_can_spawn_map, -1) / jnp.sum(melee_mobs_can_spawn_map),
    )
    melee_mob_position = jnp.array(
        [
            melee_mob_position // static_params.map_size[0],
            melee_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_melee_mob_index = jnp.argmax(
        jnp.logical_not(state.melee_mobs.mask[state.player_level])
    )

    new_melee_mob_position = jax.lax.select(
        can_spawn_melee_mob,
        melee_mob_position,
        state.melee_mobs.position[state.player_level, new_melee_mob_index],
    )

    new_melee_mob_health = jax.lax.select(
        can_spawn_melee_mob,
        MOB_TYPE_HEALTH_MAPPING[new_melee_mob_type, MobType.MELEE.value],
        state.melee_mobs.health[state.player_level, new_melee_mob_index],
    )

    new_melee_mob_mask = jax.lax.select(
        can_spawn_melee_mob,
        True,
        state.melee_mobs.mask[state.player_level, new_melee_mob_index],
    )

    melee_mobs = Mobs(
        position=state.melee_mobs.position.at[
            state.player_level, new_melee_mob_index
        ].set(new_melee_mob_position),
        health=state.melee_mobs.health.at[state.player_level, new_melee_mob_index].set(
            new_melee_mob_health
        ),
        mask=state.melee_mobs.mask.at[state.player_level, new_melee_mob_index].set(
            new_melee_mob_mask
        ),
        attack_cooldown=state.melee_mobs.attack_cooldown,
        type_id=state.melee_mobs.type_id.at[
            state.player_level, new_melee_mob_index
        ].set(new_melee_mob_type),
    )

    state = state.replace(
        melee_mobs=melee_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_melee_mob_position[0], new_melee_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_melee_mob_position[0],
                    new_melee_mob_position[1],
                ],
                new_melee_mob_mask,
            )
        ),
    )

    # Ranged mobs
    can_spawn_ranged_mob = (
        state.ranged_mobs.mask[state.player_level].sum() <
        (
            static_params.max_melee_mobs * 
            (
                static_params.player_count * (1 - in_dungeon) + 
                1 * in_dungeon # reduce number of mobs if in dungeons to avoid crowdedness
            )
        )
    )

    new_ranged_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.RANGED.value]
    new_ranged_mob_type_boss = FLOOR_MOB_MAPPING[
        state.boss_progress, MobType.RANGED.value
    ]

    new_ranged_mob_type = jax.lax.select(
        is_fighting_boss(state, static_params),
        new_ranged_mob_type_boss,
        new_ranged_mob_type,
    )

    rng, _rng = jax.random.split(rng)
    can_spawn_ranged_mob = jnp.logical_and(
        can_spawn_ranged_mob,
        jax.random.uniform(_rng)
        < floor_mob_spawn_chance[state.player_level, 2] * monster_spawn_coeff,
    )

    # Hack for deep thing
    ranged_mobs_can_spawn_map = jax.lax.select(
        new_ranged_mob_type == 5,
        state.map[state.player_level] == BlockType.WATER.value,
        all_valid_blocks_map,
    )
    ranged_mobs_can_spawn_map = jax.lax.select(
        is_fighting_boss(state, static_params), grave_map, ranged_mobs_can_spawn_map
    )

    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, monsters_can_spawn_player_range_map
    )
    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, jnp.logical_not(state.mob_map[state.player_level])
    )
    ranged_mobs_can_spawn_map = ranged_mobs_can_spawn_map.at[
        state.player_position[:, 0], state.player_position[:, 1]
    ].set(False)

    can_spawn_ranged_mob = jnp.logical_and(
        can_spawn_ranged_mob, ranged_mobs_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    ranged_mob_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(ranged_mobs_can_spawn_map, -1)
        / jnp.sum(ranged_mobs_can_spawn_map),
    )
    ranged_mob_position = jnp.array(
        [
            ranged_mob_position // static_params.map_size[0],
            ranged_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_ranged_mob_index = jnp.argmax(
        jnp.logical_not(state.ranged_mobs.mask[state.player_level])
    )

    new_ranged_mob_position = jax.lax.select(
        can_spawn_ranged_mob,
        ranged_mob_position,
        state.ranged_mobs.position[state.player_level, new_ranged_mob_index],
    )

    new_ranged_mob_health = jax.lax.select(
        can_spawn_ranged_mob,
        MOB_TYPE_HEALTH_MAPPING[new_ranged_mob_type, MobType.RANGED.value],
        state.ranged_mobs.health[state.player_level, new_ranged_mob_index],
    )

    new_ranged_mob_mask = jax.lax.select(
        can_spawn_ranged_mob,
        True,
        state.ranged_mobs.mask[state.player_level, new_ranged_mob_index],
    )

    ranged_mobs = Mobs(
        position=state.ranged_mobs.position.at[
            state.player_level, new_ranged_mob_index
        ].set(new_ranged_mob_position),
        health=state.ranged_mobs.health.at[
            state.player_level, new_ranged_mob_index
        ].set(new_ranged_mob_health),
        mask=state.ranged_mobs.mask.at[state.player_level, new_ranged_mob_index].set(
            new_ranged_mob_mask
        ),
        attack_cooldown=state.ranged_mobs.attack_cooldown,
        type_id=state.ranged_mobs.type_id.at[
            state.player_level, new_ranged_mob_index
        ].set(new_ranged_mob_type),
    )

    state = state.replace(
        ranged_mobs=ranged_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_ranged_mob_position[0], new_ranged_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_ranged_mob_position[0],
                    new_ranged_mob_position[1],
                ],
                new_ranged_mob_mask,
            )
        ),
    )

    return state
