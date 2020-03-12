def needs_chain_conversion(placement):
    """
    Helper function to determine whether or not an input is in a chain-ready data structure.
    """
    for v in placement.values():
        if isinstance(v, (list, frozenset, set)):
            return False
        return True
