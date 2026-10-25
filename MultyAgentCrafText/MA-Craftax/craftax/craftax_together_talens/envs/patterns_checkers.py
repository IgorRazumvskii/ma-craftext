import jax
import jax.numpy as jnp
from craftax_together_talens.craftax_state import EnvState, StaticEnvParams
from craftax_together_talens.constants import BlockType


def check_one_table(state: EnvState, static_params: StaticEnvParams) -> bool:
    """
    Проверяет, есть ли хотя бы один стол (CRAFTING_TABLE) на карте.
    
    Args:
        state: Состояние игры
        static_params: Статические параметры окружения
        
    Returns:
        True, если найден хотя бы один стол, иначе False
    """
    current_level = state.player_level
    level_map = state.map[current_level]
    has_table = (level_map == BlockType.CRAFTING_TABLE.value).any()
    return has_table


def check_two_tables_straight_line(state: EnvState, static_params: StaticEnvParams) -> bool:
    """
    Проверяет, есть ли 2 стола (CRAFTING_TABLE) в прямой линии (горизонтально или вертикально).
    
    Args:
        state: Состояние игры
        static_params: Статические параметры окружения
        
    Returns:
        True, если найдены 2 стола в прямой линии, иначе False
    """
    return _check_two_blocks_straight_line(
        state, static_params, BlockType.CRAFTING_TABLE.value
    )


def check_two_tables_diagonal_line(state: EnvState, static_params: StaticEnvParams) -> bool:
    """
    Проверяет, есть ли 2 стола (CRAFTING_TABLE) в диагональной линии.
    
    Args:
        state: Состояние игры
        static_params: Статические параметры окружения
        
    Returns:
        True, если найдены 2 стола в диагональной линии, иначе False
    """
    return _check_two_blocks_diagonal_line(
        state, static_params, BlockType.CRAFTING_TABLE.value
    )


def check_two_furnaces_straight_line(state: EnvState, static_params: StaticEnvParams) -> bool:
    """
    Проверяет, есть ли 2 печи (FURNACE) в прямой линии (горизонтально или вертикально).
    
    Args:
        state: Состояние игры
        static_params: Статические параметры окружения
        
    Returns:
        True, если найдены 2 печи в прямой линии, иначе False
    """
    return _check_two_blocks_straight_line(
        state, static_params, BlockType.FURNACE.value
    )


def check_two_furnaces_diagonal_line(state: EnvState, static_params: StaticEnvParams) -> bool:
    """
    Проверяет, есть ли 2 печи (FURNACE) в диагональной линии.
    
    Args:
        state: Состояние игры
        static_params: Статические параметры окружения
        
    Returns:
        True, если найдены 2 печи в диагональной линии, иначе False
    """
    return _check_two_blocks_diagonal_line(
        state, static_params, BlockType.FURNACE.value
    )


def _check_two_blocks_straight_line(
    state: EnvState, static_params: StaticEnvParams, block_type: int
) -> bool:
    """
    Вспомогательная функция для проверки двух блоков одного типа в прямой линии.
    Проверяет горизонтальные и вертикальные соседние позиции.
    """
    current_level = state.player_level
    level_map = state.map[current_level]
    map_height, map_width = static_params.map_size
    
    # Создаем маску блоков нужного типа
    block_mask = level_map == block_type
    
    # Проверяем соседние позиции в прямой линии
    # Используем срезы для правильной обработки границ
    
    # Проверка горизонтальных пар (слева-справа)
    # Справа: проверяем (x, y) и (x+1, y)
    right_neighbor = jnp.pad(block_mask[1:, :], ((0, 1), (0, 0)), mode='constant', constant_values=False)
    has_right_pair = jnp.logical_and(block_mask, right_neighbor)
    
    # Слева: проверяем (x, y) и (x-1, y)
    left_neighbor = jnp.pad(block_mask[:-1, :], ((1, 0), (0, 0)), mode='constant', constant_values=False)
    has_left_pair = jnp.logical_and(block_mask, left_neighbor)
    
    # Проверка вертикальных пар (сверху-снизу)
    # Снизу: проверяем (x, y) и (x, y+1)
    bottom_neighbor = jnp.pad(block_mask[:, 1:], ((0, 0), (0, 1)), mode='constant', constant_values=False)
    has_bottom_pair = jnp.logical_and(block_mask, bottom_neighbor)
    
    # Сверху: проверяем (x, y) и (x, y-1)
    top_neighbor = jnp.pad(block_mask[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=False)
    has_top_pair = jnp.logical_and(block_mask, top_neighbor)
    
    # Проверяем, есть ли хотя бы одна пара
    has_straight_line = jnp.logical_or(
        has_right_pair.any(),
        jnp.logical_or(
            has_left_pair.any(),
            jnp.logical_or(
                has_bottom_pair.any(),
                has_top_pair.any()
            )
        )
    )
    
    return has_straight_line


def _check_two_blocks_diagonal_line(
    state: EnvState, static_params: StaticEnvParams, block_type: int
) -> bool:
    """
    Вспомогательная функция для проверки двух блоков одного типа в диагональной линии.
    Проверяет диагональные соседние позиции.
    """
    current_level = state.player_level
    level_map = state.map[current_level]
    map_height, map_width = static_params.map_size
    
    # Создаем маску блоков нужного типа
    block_mask = level_map == block_type
    
    # Проверяем соседние позиции по диагонали
    # Используем срезы для правильной обработки границ
    
    # Вправо-вниз: проверяем (x, y) и (x+1, y+1)
    right_bottom_neighbor = jnp.pad(
        block_mask[1:, 1:], 
        ((0, 1), (0, 1)), 
        mode='constant', 
        constant_values=False
    )
    has_right_bottom_pair = jnp.logical_and(block_mask, right_bottom_neighbor)
    
    # Вправо-вверх: проверяем (x, y) и (x+1, y-1)
    right_top_neighbor = jnp.pad(
        block_mask[1:, :-1], 
        ((0, 1), (1, 0)), 
        mode='constant', 
        constant_values=False
    )
    has_right_top_pair = jnp.logical_and(block_mask, right_top_neighbor)
    
    # Влево-вниз: проверяем (x, y) и (x-1, y+1)
    left_bottom_neighbor = jnp.pad(
        block_mask[:-1, 1:], 
        ((1, 0), (0, 1)), 
        mode='constant', 
        constant_values=False
    )
    has_left_bottom_pair = jnp.logical_and(block_mask, left_bottom_neighbor)
    
    # Влево-вверх: проверяем (x, y) и (x-1, y-1)
    left_top_neighbor = jnp.pad(
        block_mask[:-1, :-1], 
        ((1, 0), (1, 0)), 
        mode='constant', 
        constant_values=False
    )
    has_left_top_pair = jnp.logical_and(block_mask, left_top_neighbor)
    
    # Проверяем, есть ли хотя бы одна диагональная пара
    has_diagonal_line = jnp.logical_or(
        has_right_bottom_pair.any(),
        jnp.logical_or(
            has_right_top_pair.any(),
            jnp.logical_or(
                has_left_bottom_pair.any(),
                has_left_top_pair.any()
            )
        )
    )
    
    return has_diagonal_line

