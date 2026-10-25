import jax.numpy as jnp
import jax.lax as lax
import jax

from typing import Union
from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic

from craftext.environment.scenarious.checkers.target_state import BuildStarState

from functools import partial


import jax
import jax.numpy as jnp

def checker_star(game_data: Union[GameDataClassic, GameData],  target_state: BuildStarState) -> jax.Array:
    
    block_index = target_state.block_type
    radius = target_state.radius
    size = target_state.size
    cross_type = target_state.cross_type

    return is_cross_formed(10, 7, game_data, block_index, radius, size, cross_type)

@partial(jax.jit, static_argnums=(0,1))
def is_cross_formed(
    max_radius: int,
    max_size:   int,
    game_data,
    block_index: int,    
    cross_type:  int,    
    radius:      int,
    size:        int     
):

    """
    Detects whether a cross of a given size and type is formed around the player.
    """

    #Extract player position
    x, y = game_data.states[0].variables.player_position

    #Define region parameters
    R = max_radius                  # maximum allowed radius
    FULL = 2 * R + 1                # full side length of padded region

    # Pad the full game map to safely slice around edges
    padded = jnp.pad(
        game_data.states[0].map.game_map,
        ((R, R), (R, R)),
        constant_values=-1          # out‐of‐bounds marker
    )

    # Slice out the full square region centered on the player
    region_full = lax.dynamic_slice(
        padded,
        (x, y),
        (FULL, FULL)
    )  # shape [FULL, FULL]

    # Build a boolean mask for the “search radius” within that region
    coords = jnp.arange(-R, R + 1)               # coordinates relative to center
    mask1d = jnp.abs(coords) <= radius           # which offsets are within the actual radius
    mask2d = mask1d[:, None] & mask1d[None, :]   # 2D circular (square) mask
    # apply mask: out‐of‐radius cells get -1
    region = jnp.where(mask2d, region_full, -1)

    # Create a 4D tensor for convolution: shape [batch=1, chan_in=1, H, W]
    B = (region == block_index).astype(jnp.float32)[None, None, ...]

    # Prepare filters for horizontal, vertical, and two diagonal lines
    S = max_size        # maximum filter side length
    C = S // 2          # center index in filter

    idxs = jnp.arange(S)
    half = size // 2
    start = C - half
    end = start + size
    mask_range = (idxs >= start) & (idxs < end)  # which rows/cols fall inside the desired size

    row_idx = idxs[:, None]
    col_idx = idxs[None, :]

    # horizontal line filter at center row
    filt_h  = (row_idx == C) & mask_range[None, :]
    # vertical line filter at center column
    filt_v  = (col_idx == C) & mask_range[:, None]
    # main diagonal filter
    filt_d1 = (row_idx == col_idx) & mask_range[:, None] & mask_range[None, :]
    # anti-diagonal filter
    filt_d2 = (row_idx + col_idx == 2 * C) & mask_range[:, None] & mask_range[None, :]

    # convert boolean filters to float32 and add batch/channel dims
    kh  = filt_h.astype(jnp.float32)[None, None, ...]
    kv  = filt_v.astype(jnp.float32)[None, None, ...]
    kd1 = filt_d1.astype(jnp.float32)[None, None, ...]
    kd2 = filt_d2.astype(jnp.float32)[None, None, ...]

    # Helper partial for 2D convolution with VALID padding, NCHW layout
    conv = partial(
        lax.conv_general_dilated,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW")
    )

    # Convolve the region tensor with each filter
    h_out  = conv(B, kh)[0, 0]   # horizontal sum at each center
    v_out  = conv(B, kv)[0, 0]   # vertical sum
    d1_out = conv(B, kd1)[0, 0]  # main diagonal sum
    d2_out = conv(B, kd2)[0, 0]  # anti-diagonal sum

    # Determine which lines are fully filled (sum == size)
    straight = (h_out == size) & (v_out == size)
    diagonal = (d1_out == size) & (d2_out == size)

    # Select based on cross_type: 0=straight only, 1=diagonal only, else either
    mask = lax.cond(
        cross_type == 0,
        lambda _: straight,
        lambda _: lax.cond(
            cross_type == 1,
            lambda _: diagonal,
            lambda _: straight | diagonal,
            operand=None
        ),
        operand=None
    )

    # Return True if any center yields a valid cross
    return jnp.any(mask)

