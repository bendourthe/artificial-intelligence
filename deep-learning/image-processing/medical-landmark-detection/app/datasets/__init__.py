# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

from .chest_dataset import Chest
from .spine_dataset import Spine


# --------------------------------------------------------------------------- #
# METHODS DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def get_dataset(s):
    return {'chest': Chest,
            'cervical_spine': Spine}[s]
