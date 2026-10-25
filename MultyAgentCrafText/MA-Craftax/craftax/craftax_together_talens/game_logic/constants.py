from enum import Enum
import jax
import jax.numpy as jnp
from craftax_together_talens.constants import Action

REQUEST_PICKAXE = 54
REQUEST_SWORD = 55
REQUEST_ARMOUR = 56
REQUEST_BOW = 57
REQUEST_ARROWS = 58
REQUEST_TORCHES = 59

BASE_REQUEST_COUNT = (
    Action.REQUEST_SAPPHIRE.value - Action.REQUEST_FOOD.value + 1
)
EXTENDED_REQUEST_COUNT = REQUEST_TORCHES - REQUEST_PICKAXE + 1
REQUEST_OBSERVATION_DIM = BASE_REQUEST_COUNT + EXTENDED_REQUEST_COUNT


def get_request_observation_index(request_type):
    """Map disjoint request action IDs to a compact observation index."""
    material_index = request_type - Action.REQUEST_FOOD.value
    extended_index = BASE_REQUEST_COUNT + request_type - REQUEST_PICKAXE
    return jnp.where(request_type >= REQUEST_PICKAXE, extended_index, material_index)


# Miner gathering balance: amount of resources per successful mine action.
MINER_MINED_RESOURCE_YIELD = 2

# If False, agents cannot use DESCEND (no ladder down / dungeon); masks RL and step logic.
ALLOW_PLAYER_DESCEND = False


class Specialization(Enum):
    UNASSIGNED = 0
    MINER = 1  # Добывает камни и деревья
    BUILDER = 2  # Строит столы и печки
    CRAFTER = 3  # Warrior


_COMMON_ACTION_LIST = [
    Action.NOOP.value,
    Action.LEFT.value,
    Action.RIGHT.value,
    Action.UP.value,
    Action.DOWN.value,
    Action.DO.value,
    Action.SLEEP.value,
    Action.REST.value,
]
if ALLOW_PLAYER_DESCEND:
    _COMMON_ACTION_LIST.append(Action.DESCEND.value)
_COMMON_ACTION_LIST.append(Action.ASCEND.value)
COMMON_ACTIONS = tuple(_COMMON_ACTION_LIST)

MINER_ACTIONS = (
    REQUEST_PICKAXE,
)

WARRIOR_ACTIONS = (
    REQUEST_SWORD,
)

BUILDER_ACTIONS = (
    Action.REQUEST_WOOD.value,
    Action.REQUEST_STONE.value,
    Action.REQUEST_IRON.value,
    Action.REQUEST_COAL.value,
    Action.REQUEST_DIAMOND.value,
    Action.REQUEST_RUBY.value,
    Action.REQUEST_SAPPHIRE.value,
    Action.PLACE_STONE.value,
    Action.PLACE_TABLE.value,
    Action.PLACE_FURNACE.value,
    Action.PLACE_PLANT.value,
    Action.PLACE_TORCH.value,
    Action.MAKE_WOOD_PICKAXE.value,
    Action.MAKE_STONE_PICKAXE.value,
    Action.MAKE_IRON_PICKAXE.value,
    Action.MAKE_DIAMOND_PICKAXE.value,
    Action.MAKE_WOOD_SWORD.value,
    Action.MAKE_STONE_SWORD.value,
    Action.MAKE_IRON_SWORD.value,
    Action.MAKE_DIAMOND_SWORD.value,
    Action.MAKE_IRON_ARMOUR.value,
    Action.MAKE_DIAMOND_ARMOUR.value,
    Action.MAKE_ARROW.value,
    Action.MAKE_TORCH.value,
)


def _in_actions(action, action_ids):
    return jnp.isin(action, jnp.array(action_ids, dtype=jnp.int32))


def is_action_allowed_for_specialization(action, specialization):
    # GIVE actions are encoded as a contiguous range:
    # Action.GIVE and (num_agents - 2) extra GIVE_TO_PLAYER actions.
    # In the current project layout these values are placed before REQUEST_PICKAXE.
    give_allowed = jnp.logical_and(
        action >= Action.GIVE.value,
        action < REQUEST_PICKAXE,
    )
    common_allowed = jnp.logical_or(_in_actions(action, COMMON_ACTIONS), give_allowed)
    miner_allowed = jnp.logical_or(common_allowed, _in_actions(action, MINER_ACTIONS))
    warrior_allowed = jnp.logical_or(common_allowed, _in_actions(action, WARRIOR_ACTIONS))
    builder_allowed = jnp.logical_or(common_allowed, _in_actions(action, BUILDER_ACTIONS))
    return jnp.select(
        [
            specialization == Specialization.MINER.value,
            specialization == Specialization.CRAFTER.value,
            specialization == Specialization.BUILDER.value,
        ],
        [
            miner_allowed,
            warrior_allowed,
            builder_allowed,
        ],
        default=common_allowed,
    )


def get_allowed_action_flags(actions, specializations):
    return jax.vmap(is_action_allowed_for_specialization)(actions, specializations)


def get_role_action_mask(specialization, action_dim):
    action_ids = jnp.arange(action_dim, dtype=jnp.int32)
    return jax.vmap(lambda action: is_action_allowed_for_specialization(action, specialization))(action_ids)


def get_team_action_masks(specializations, action_dim):
    return jax.vmap(lambda spec: get_role_action_mask(spec, action_dim))(specializations)


def get_action_dim(player_count):
    """Size required for all GIVE targets and extended request actions."""
    give_action_dim = Action.GIVE.value + player_count - 1
    request_action_dim = REQUEST_TORCHES + 1
    return max(give_action_dim, request_action_dim)