# ВАЖНО: Добавляем путь в sys.path ПЕРЕД всеми импортами!
# Используем тот же подход, что и в wrapper_game_logic.py
import os
from re import I
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
from jax import lax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
import functools

from step import *

from craftax_coop.craftax_state import EnvState, EnvParams
from craftax_coop.checker import (
    check_two_tables_straight_line,
    check_two_tables_diagonal_line,
    check_two_furnaces_straight_line,
    check_two_furnaces_diagonal_line,
)
from craftax_coop.renderer.renderer_symbolic import render_craftax_symbolic
from craftax_coop.renderer.renderer_pixels import render_craftax_pixels

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