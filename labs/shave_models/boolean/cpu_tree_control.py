"""Fit a compact CPU decision tree to the exact seven-bit policy outputs."""
from itertools import permutations


NODE = 128  # Leaves hold raw output bits. The high bit refers to a node.


def fit_tree(outputs):
    """Try every fact order and share identical branches within each tree."""
    facts = (len(outputs) - 1).bit_length()
    if not 0 < len(outputs) == 1 << facts <= NODE or any(code >= NODE for code in outputs):
        raise ValueError("Expected a complete truth table with up to seven facts and output bits")

    def ordered_tree(order):
        nodes = bytearray()
        addresses = {}

        def add(depth, mask):
            if depth == facts:
                return outputs[mask]
            fact = order[depth]
            off = add(depth + 1, mask)
            on = add(depth + 1, mask | (1 << fact))
            if off == on:
                return off
            key = (fact, off, on)
            if key not in addresses:
                addresses[key] = NODE | (len(nodes) // 3)
                nodes.extend(key)
            return addresses[key]

        root = add(0, 0)
        return bytes([root]) + nodes

    return min(map(ordered_tree, permutations(range(facts))), key=len)


def decide_tree(tree, mask):
    code = tree[0]
    while code & NODE:
        offset = 1 + (code & (NODE - 1)) * 3
        fact = tree[offset]
        code = tree[offset + 1 + ((mask >> fact) & 1)]
    return code
