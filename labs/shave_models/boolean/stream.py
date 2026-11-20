"""Incremental evidence and Needle token counts for one append-only document."""
from dataclasses import dataclass
import time

import numpy as np

from policy import SegmentMemo, needle_count, validate_count


@dataclass(frozen=True)
class StreamState:
    tokens: int
    suffix: str
    halo: bytes
    stable_facts: np.ndarray
    explanation_text: str


class DecisionStream:
    def __init__(self, engine, explain=False):
        if engine.info["variant"] == "scan":
            raise ValueError("Streaming requires a decision head that accepts fact bits")
        self.engine = engine
        self.explain = explain
        self.memo = SegmentMemo()
        self.halo_size = max(map(len, engine.policy.patterns)) - 1
        self.state = self.empty_state()

    def empty_state(self):
        facts = np.zeros(len(self.engine.policy.literals), dtype=np.float32)
        return StreamState(0, "", b"", facts, "")

    def _initial_halo(self, text):
        if not self.halo_size:
            return b""
        width = 256
        while width < len(text):
            begin = text.find(" ", len(text) - width)
            if begin >= 0:
                encoded = self.engine.policy.project(text[begin:])
                if len(encoded) >= self.halo_size:
                    return encoded[-self.halo_size:]
            width *= 2
        return self.engine.policy.project(text)[-self.halo_size:]

    def append(self, fragment, reset=False):
        if not isinstance(fragment, str) or type(reset) is not bool:
            raise ValueError("Append must be a string and reset must be a Boolean")
        current = self.empty_state() if reset else self.state
        next_state = None
        policy = self.engine.policy

        def prepare(delta):
            nonlocal next_state
            pending = current.suffix + delta
            split = pending.rfind(" ")
            committed = pending[:split] if split > 0 else ""
            suffix = pending[split:] if split > 0 else pending

            # The pinned Needle model cannot merge across an ASCII space. Keep
            # that space with the uncommitted segment so its prefix is exact.
            stable_tokens = current.tokens + needle_count(self.engine.tokenizer, committed, memo=self.memo)
            tokens = validate_count(stable_tokens + needle_count(self.engine.tokenizer, suffix, memo=self.memo))
            counted = time.perf_counter()

            if not current.halo and len(committed) >= 4096:
                stable_facts = np.maximum(current.stable_facts, policy.text_atoms(committed))
                halo = self._initial_halo(committed)
            else:
                text = current.halo + policy.project(committed)
                stable_facts = np.maximum(current.stable_facts, policy.atoms(text))
                halo = text[-self.halo_size:] if self.halo_size else b""
            tail_facts = policy.atoms(halo + policy.project(suffix)) if halo else policy.text_atoms(suffix)
            facts = np.maximum(stable_facts, tail_facts)
            explanation_text = current.explanation_text + delta if self.explain else ""
            next_state = StreamState(stable_tokens, suffix, halo, stable_facts, explanation_text)
            return tokens, facts, counted

        prepared = self.engine.prepare_decision(fragment, prepare=prepare)
        result = self.engine.decide_prepared([prepared])[0]
        if self.explain:
            result["explanation"] = policy.explain(next_state.explanation_text, prepared.atoms)
        # A rejected append or a failed inference leaves the previous stream intact.
        self.state = next_state
        return result
