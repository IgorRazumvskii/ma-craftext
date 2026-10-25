from flax import struct
import jax.numpy as jnp
from craftext.environment.craftext_constants import Achievement, AchievementState, BlockType, TimeState
import jax 

@struct.dataclass
class BuildLineState:

    block_type: int = BlockType.INVALID
    size:int = 3
    radius: int = 3
    is_diagonal:bool = False

@struct.dataclass
class BuildSquareState:
    
    block_type: int = BlockType.INVALID
    size: int = 3
    radius: int = 5
    
@struct.dataclass
class BuildStarState:
    
    block_type: int = BlockType.INVALID
    size: int = 3
    radius: int = 3
    cross_type: int = -1

@struct.dataclass
class ConditionalPlacingState:
    
    object_inventory_enum: int = -1
    object_to_place: int = 0
    count_to_collect: int = 0
    count_to_stand: int = 1
    
@struct.dataclass
class LocalizaPlacingState:
    
    object_name: int = -1
    target_object_name: int = -1
    side: int = -1
    distance: int = 5

@struct.dataclass
class Achievements:
    
    achievement_mask: list = struct.field(default_factory=lambda: tuple([AchievementState.NOT_MATTER for i in range(Achievement.MAKE_IRON_SWORD + 1)]))

@struct.dataclass
class TimeCosntrainedPlacmentState:
    
    block_type: int = BlockType.INVALID
    time_state: int = TimeState.DAY
    radius: int = 5

@struct.dataclass
class UnifiedPatternState:
    
    block_type: int = BlockType.INVALID
    pattern_type: jax.Array = struct.field(default_factory=lambda: jnp.zeros(1))
    size: int = 3
    radius: int = 3

@struct.dataclass
class TargetState:
    achievements: Achievements = struct.field(default_factory=Achievements)
    building_line: BuildLineState =  struct.field(default_factory=BuildLineState)
    building_square: BuildSquareState = struct.field(default_factory=BuildSquareState)
    building_star: BuildStarState =  struct.field(default_factory=BuildStarState)
    conditional_placing: ConditionalPlacingState = struct.field(default_factory=ConditionalPlacingState)
    Localization_placing: LocalizaPlacingState = struct.field(default_factory=LocalizaPlacingState)
    time_placement: TimeCosntrainedPlacmentState = struct.field(default_factory=TimeCosntrainedPlacmentState)
    unified_pattern_state: UnifiedPatternState = struct.field(default_factory=UnifiedPatternState)

    @classmethod
    def stack(cls, lst: list['TargetState']) -> 'TargetState':
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *lst)

    def select(self, idx: jnp.ndarray) -> 'TargetState':
        return jax.tree_util.tree_map(lambda arr: arr[idx], self)
