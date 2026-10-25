from enum import Enum

REQUEST_PICKAXE = 54
REQUEST_SWORD = 55
REQUEST_ARMOUR = 56
REQUEST_BOW = 57
REQUEST_ARROWS = 58
REQUEST_TORCHES = 59


class Specialization(Enum):
    UNASSIGNED = 0
    MINER = 1  # Добывает камни и деревья
    BUILDER = 2  # Строит столы и печки
    CRAFTER = 3  # Отвечает за крафт