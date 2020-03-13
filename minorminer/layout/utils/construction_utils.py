def convert_to_chains(placement):
    """
    Helper function to convert a placement to a chain-ready data structure.
    """
    for v in placement.values():
        if isinstance(v, (list, frozenset, set)):
            return dict(placement)
        return {v: [q] for v, q in placement.items()}
