from typing import List
from config import settings as st


def get_cache_capacities() -> List[int]:
    """
    Parses the parameter file and generates a list
    of the cache capacities of each player
    """
    param_type = st.CACHE_CAPACITY["type"]
    if param_type == "constant":
        quantity = st.CACHE_CAPACITY["quantity"]
        capacities = [quantity for _ in range(st.NUM_CLIENTS)]

    return capacities
