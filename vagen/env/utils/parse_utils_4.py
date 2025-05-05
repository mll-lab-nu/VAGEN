from .parse_utils import parse_freethink,parse_no_think
from .parse_utils_3 import parse_grounding,parse_grounding_worldmodeling,parse_worldmodeling 
parse_function_map = {
    "free_think": parse_freethink,
    "no_think": parse_no_think,
    "grounding": parse_grounding,
    "worldmodeling": parse_worldmodeling,
    "grounding_worldmodeling": parse_grounding_worldmodeling,
}
