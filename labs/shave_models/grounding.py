"""Encode public DOM or desktop accessibility candidates for device-side ranking."""
from collections import Counter
from functools import lru_cache
import hashlib
import re

import numpy as np


COUNT = 32
WIDTH = 96
STOP = frozenset("a an at by for from in of on the to with".split())


def words(value):
    return set(re.findall(r"[a-z0-9]+", str(value).casefold())) - STOP


@lru_cache(maxsize=8192)
def word_bucket(value):
    return int.from_bytes(hashlib.blake2s(value.encode(), digest_size=2).digest(), "little") % 32


def encode_screen(task, controls, history=""):
    """Return fixed features and corresponding control IDs. Ties follow input order."""
    query, previous = words(task), words(history)
    entries = []
    ids = set()
    for item in controls:
        key = str(item["id"])
        if key in ids:
            raise ValueError("Duplicate control ID: " + key)
        ids.add(key)
        name, role = words(item.get("name", "")), str(item.get("role", "")).lower()
        entries.append((key, name, role, item))
    frequencies = Counter(term for _, name, _, _ in entries for term in name)
    candidates = []
    for key, name, role, item in entries:
        overlap = query & name
        rarity = sum(1 / frequencies[w] for w in overlap)
        vector = np.zeros(WIDTH, dtype=np.float32)
        vector[:8] = (len(overlap) / max(1, len(query)), len(overlap) / max(1, len(name)),
                      min(1, rarity / 3), min(1, len(name) / 15), bool(name),
                      len(name & previous) / max(1, len(name)), bool(item.get("editable")),
                      bool(item.get("clickable")))
        roles = ("button", "link", "textbox", "combobox", "option", "checkbox", "tab",
                 "menuitem", "radio", "listitem", "text", "other")
        vector[8 + (roles.index(role) if role in roles else 11)] = 1
        for term in name:
            vector[32 + word_bucket(term)] = 1
        for term in query:
            vector[64 + word_bucket(role + ":" + term)] = 1
        priority = rarity + vector[0] + 0.05 * vector[7] + 0.03 * bool(name)
        candidates.append((priority, key, vector))
    ranked = sorted(candidates, key=lambda row: -row[0])[:COUNT]
    packed = np.zeros((WIDTH, 1, COUNT), dtype=np.float32)
    # The sentinel lets the model recognize padding. Padding is also masked at decode.
    packed[31] = 1
    for i, (_, _, vector) in enumerate(ranked):
        packed[:, 0, i] = vector
    return packed, [key for _, key, _ in ranked]
