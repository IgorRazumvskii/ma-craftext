# ВАЖНО: Добавляем путь в sys.path ПЕРЕД всеми импортами!
# Используем тот же подход, что и в wrapper_game_logic.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
from jax import lax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
import functools

from craftax_together_talens.game_logic.step import craftax_step

from craftax_together_talens.craftax_state import EnvState, EnvParams
from craftax_together_talens.constants import Action
# NOTE: pattern checkers are intentionally disabled in instruction-only mode.
from craftax_together_talens.renderer.renderer_symbolic import render_craftax_symbolic
from craftax_together_talens.renderer.renderer_pixels import render_craftax_pixels

# ===========================
# Environment Wrappers
# ===========================
from craftax_together_talens.envs.craftax_symbolic_env import CraftaxCoopSymbolicEnv
from craftax_together_talens.envs.craftax_pixels_env import CraftaxCoopPixelsEnv
from craftax_together_talens.envs.craftext_adapter import MultiAgentInstructionAdapter
from jaxmarl.environments import spaces


class CraftaxCoopSymbolicEnvWithMyWrapper(CraftaxCoopSymbolicEnv):
    """
    Обертка для CraftaxCoopSymbolicEnv, которая использует craftax_step из my_wrapper.
    """
    def __init__(self, num_agents: int = 3):
        """
        Args:
            num_agents: Количество агентов
        """
        super().__init__(num_agents)
    
    def action_shape(self) -> spaces.Discrete:
        """
        Использует полное пространство базовой среды, включая запросы инструментов.
        """
        return super().action_shape()
    
    def observation_shape(self) -> spaces.Box:
        """
        Переопределенный observation_shape.
        """
        base_obs_shape = (
            self.get_flat_map_obs_shape() + 
            self.get_teammate_dashboard_obs_shape() + 
            self.get_inventory_obs_shape()
        )
        # Паттерн one-hot отключен: используем только базовое наблюдение.
        obs_shape = base_obs_shape
        
        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.int32,
        )
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, _=None) -> Tuple[Dict[str, chex.Array], EnvState]:
        """
        Переопределенный reset без паттернов.
        """
        # Паттерны отключены: обычный reset.
        return super().reset(key, _)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """
        Переопределенный is_terminal без паттернов:
        используем базовую терминальность (death/timeout/boss).
        """
        return super().is_terminal(state, params)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: EnvState) -> Dict[str, chex.Array]:
        """
        Переопределенный get_obs без паттерн-признаков.
        """
        # Получаем только базовые наблюдения.
        obs_sym = lax.stop_gradient(
            render_craftax_symbolic(
                state, 
                self.static_env_params,
            )
        )

        obs = {n: o for n, o in zip(self.agents, obs_sym)}
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
        from craftax_together_talens.envs.common import compute_score
        user_info = compute_score(state, done, self.static_env_params)
        
        # Паттерн-метрики отключены: теперь используем только метрики инструкций (CrafText).
        
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
    """
    def __init__(self, num_agents: int = 3):
        """
        Args:
            num_agents: Количество агентов
        """
        super().__init__(num_agents)
    
    def action_shape(self) -> spaces.Discrete:
        """
        Использует полное пространство базовой среды, включая запросы инструментов.
        """
        return super().action_shape()
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, _=None) -> Tuple[Dict[str, chex.Array], EnvState]:
        """
        Переопределенный reset без паттернов.
        """
        # Паттерны отключены: обычный reset.
        return super().reset(key, _)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """
        Переопределенный is_terminal без паттернов:
        используем базовую терминальность (death/timeout/boss).
        """
        return super().is_terminal(state, params)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: EnvState) -> Dict[str, chex.Array]:
        """
        Переопределенный get_obs для пиксельных наблюдений.
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
        
        # Для pixels окружения возвращаем изображения как есть.
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
        from craftax_together_talens.envs.common import compute_score
        user_info = compute_score(state, done, self.static_env_params)
        info["discount"] = self.discount(state, self.default_params)
        
        # Паттерн-метрики отключены: теперь используем только метрики инструкций (CrafText).
        
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


def make_craftax_env_with_my_wrapper(
    env_name: str,
    craftext_config_name: str = "easy_train",
):
    """
    Создает окружение Craftax и оборачивает его в CrafText-инструкции.
    """
    if env_name == "Craftax-Coop-Symbolic":
        env = CraftaxCoopSymbolicEnvWithMyWrapper()
    elif env_name == "Craftax-Coop-Pixels":
        env = CraftaxCoopPixelsEnvWithMyWrapper()
    else:
        # Для других окружений используем стандартный способ
        from craftax.craftax_env import make_craftax_env_from_name
        env = make_craftax_env_from_name(env_name)

    return MultiAgentInstructionAdapter(env, config_name=craftext_config_name)