"""Build the CPU truth table for the Rack Lab MagicEightBall policy."""

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys


def build(root):
    sys.path.insert(0, str(root / "labs" / "shave_models" / "boolean"))
    from policy import Policy

    directory = root / "src" / "myriadforge" / "racklab" / "packs"
    policy_file = (directory / "policy.json").read_bytes()
    policy = Policy(json.loads(policy_file))
    table = [policy.result(policy.reference(atoms))["action"]
             for atoms in product((False, True), repeat=len(policy.literals))]
    path = directory / "policy-table.json"
    return path, (json.dumps({
        "policy_sha256": sha256(policy_file).hexdigest(),
        "facts": [" ".join(phrase) for phrase in policy.literals],
        "actions": table,
    }, indent=2) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path, expected = build(Path(__file__).resolve().parents[1])
    if args.check:
        if json.loads(path.read_bytes()) != json.loads(expected):
            raise SystemExit("Rack Lab policy table does not match its MagicEightBall policy")
    else:
        path.write_bytes(expected)


if __name__ == "__main__":
    main()
