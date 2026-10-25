import jax.numpy as jnp

from constants import *
from craftax_coop.constants import Action

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
