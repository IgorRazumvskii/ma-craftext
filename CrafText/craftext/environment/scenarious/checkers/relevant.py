import jax
from jax import (
    numpy as jnp,
    lax
)

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state import LocalizaPlacingState
from functools import partial


@partial(jax.jit, static_argnames=['max_radius'])
def safe_dynamic_slice(game_map, x, y, radius, max_radius):
    full_region_size = 2 * max_radius + 1

    x_padded = x + max_radius
    y_padded = y + max_radius

    region = lax.dynamic_slice(
        game_map,
        start_indices=(x_padded - max_radius, y_padded - max_radius),
        slice_sizes=(full_region_size, full_region_size)
    )

    coord_range = jnp.arange(full_region_size) - max_radius
    mask_x = jnp.abs(coord_range) <= radius
    mask_y = mask_x[:, None]
    mask = mask_x & mask_y

    region_masked = jnp.where(mask, region, -1)
    return region_masked


def cheker_localization(game_data: Union[GameDataClassic, GameData], target_state: LocalizaPlacingState) -> jax.Array:
    
    object_name = target_state.object_name
    target_object_name = target_state.target_object_name 
    side = target_state.side 
    distance = target_state.distance

    return place_object_relevant_to(game_data=game_data, object_name=object_name, target_object_name=target_object_name, side=side, distance=distance)


from functools import partial

MAX_RADIUS = 5
REGION_SIZE = 2 * MAX_RADIUS + 1

@jax.jit
def place_object_relevant_to(
    game_data: Union[GameDataClassic, GameData], 
    object_name: str, 
    target_object_name: str, 
    side: int,    
    distance: int   
) -> jax.Array:
    # Extract a square region of size REGION_SIZE×REGION_SIZE around the player
    x, y = game_data.states[0].variables.player_position
    padded_map = jnp.pad(
        game_data.states[0].map.game_map,
        ((MAX_RADIUS, MAX_RADIUS), (MAX_RADIUS, MAX_RADIUS)),
        constant_values=-1  # any out-of-bounds marker
    )
    region = lax.dynamic_slice(
        padded_map,
        (x, y),
        (REGION_SIZE, REGION_SIZE)
    )  # shape [REGION_SIZE, REGION_SIZE]

    # Build boolean masks for the target and the object
    #    (you can compare against .value directly, or if you use integer codes, omit .value)
    tgt_mask = (region == target_object_name)
    obj_mask = (region == object_name)

    # To shift obj_mask by (side, distance), pad obj_mask by MAX_RADIUS on each side
    #    and safely extract a REGION_SIZE×REGION_SIZE slice
    padded_obj = jnp.pad(
        obj_mask,
        ((MAX_RADIUS, MAX_RADIUS), (MAX_RADIUS, MAX_RADIUS)),
        constant_values=False
    )

    # Compute the dynamic offset in padded_obj coordinates:
    #    Right  (0): di=0,  dj=+distance
    #    Left   (1): di=0,  dj=-distance
    #    Up     (2): di=-distance, dj=0
    #    Down   (3): di=+distance, dj=0
    di = jnp.where(side == 2, -distance,
         jnp.where(side == 3,  distance, 0))
    dj = jnp.where(side == 0,  distance,
         jnp.where(side == 1, -distance, 0))

    # Starting index in padded_obj: center + (di, dj)
    start_i = MAX_RADIUS + di
    start_j = MAX_RADIUS + dj

    # Perform a single dynamic_slice of fixed size REGION_SIZE×REGION_SIZE
    shifted_obj = lax.dynamic_slice(
        padded_obj,
        (start_i, start_j),
        (REGION_SIZE, REGION_SIZE)
    )

    # Check if there is any position (i, j) where both tgt_mask and shifted_obj are True
    hit = tgt_mask & shifted_obj

    # If a `need_to_achieve` flag is required, handle it externally via select()
    return jnp.any(hit)