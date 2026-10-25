from craftext.dataset.scenarious.jax_build_line import instructions as build_line_instructions
from craftext.dataset.scenarious.jax_build_square import instructions as build_square_instructions
from craftext.dataset.scenarious.jax_localization_place import instructions as localization_place_instructions
from craftext.dataset.scenarious.jax_conditional_placing import instructions as conditional_place_instructions
from craftext.dataset.scenarious.jax_conditional_achievements import instructions as conditional_achievements
from craftext.dataset.scenarious.jax_build_star import instructions as build_star_instructions

easy = { 
        **conditional_achievements.easy,
}

medium = {
        **build_line_instructions.medium,
        **build_square_instructions.medium,
        **localization_place_instructions.medium,
        **conditional_place_instructions.medium,
        **conditional_achievements.easy,

}


hard = {
        **build_line_instructions.hard, 
        **build_square_instructions.hard, 
        **localization_place_instructions.hard,
        **conditional_place_instructions.hard,
        **build_star_instructions.medium
}
