"""Boolean expressions, evidence extraction and a reference independent of OpenVINO."""
import json
from pathlib import Path
import re
from collections import Counter, OrderedDict

import numpy as np

from compact_tokenizer import read_tokenizer


LIMIT = 131072
PARTS = re.compile(r'"(?:[^"\\]|\\.)*"|@[a-zA-Z_][a-zA-Z_0-9]*|AND\b|OR\b|NOT\b|[()]')
LEXEME = re.compile(r"\w+|[^\w\s]", re.UNICODE)
GROUPS = ("answers", "categories", "actions")
RULE_NAME = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*")


class Expression:
    def __init__(self, text, named):
        self.tokens = []
        self.named = named
        pos = 0
        while pos < len(text):
            if text[pos].isspace():
                pos += 1
                continue
            match = PARTS.match(text, pos)
            if not match:
                raise ValueError("Unrecognized expression near %r" % text[pos:pos+20])
            self.tokens.append(match.group())
            pos = match.end()
        self.i = 0

    def take(self, token):
        if self.i < len(self.tokens) and self.tokens[self.i] == token:
            self.i += 1
            return True
        return False

    def parse(self):
        expr = self.either()
        if self.i != len(self.tokens):
            raise ValueError("Unexpected trailing expression")
        return expr

    def either(self):
        expr = self.both()
        while self.take("OR"):
            expr = ("or", expr, self.both())
        return expr

    def both(self):
        expr = self.term()
        while self.take("AND"):
            expr = ("and", expr, self.term())
        return expr

    def term(self):
        if self.take("NOT"):
            return ("not", self.term())
        if self.take("("):
            expr = self.either()
            if not self.take(")"):
                raise ValueError("Missing closing parenthesis")
            return expr
        if self.i >= len(self.tokens):
            raise ValueError("Missing expression")
        token = self.tokens[self.i]
        self.i += 1
        if token.startswith("@") and token[1:] in self.named:
            return self.named[token[1:]]
        if token.startswith('"'):
            words = tuple(LEXEME.findall(json.loads(token).casefold()))
            if words:
                return ("literal", words)
        raise ValueError("Expected a nonempty literal or earlier rule: " + token)


def value(node, facts):
    op = node[0]
    if op == "literal":
        return facts[node[1]]
    if op == "not":
        return not value(node[1], facts)
    if op == "and":
        return value(node[1], facts) and value(node[2], facts)
    return value(node[1], facts) or value(node[2], facts)


def word_character(character):
    return character == "_" or character.isalnum()


def token_at(text, token, start):
    if start < 0 or not text.startswith(token, start):
        return False
    if not word_character(token[0]):
        return True
    end = start + len(token)
    return ((start == 0 or not word_character(text[start - 1]))
            and (end == len(text) or not word_character(text[end])))


def anchored_match(text, phrase, anchor):
    offset = 0
    word = phrase[anchor]
    # A dense adversarial anchor is cheaper to check with the full projection.
    for _ in range(64):
        match = text.find(word, offset)
        if match < 0:
            return False
        offset = match + 1
        if not token_at(text, word, match):
            continue
        left = match
        for token in reversed(phrase[:anchor]):
            while left and text[left - 1].isspace():
                left -= 1
            left -= len(token)
            if not token_at(text, token, left):
                break
        else:
            right = match + len(word)
            for token in phrase[anchor + 1:]:
                while right < len(text) and text[right].isspace():
                    right += 1
                if not token_at(text, token, right):
                    break
                right += len(token)
            else:
                return True
    return None


class Policy:
    def __init__(self, spec):
        if not isinstance(spec, dict):
            raise ValueError("Expected a JSON policy object")
        for group in ("rules",) + GROUPS:
            items = spec.get(group, {})
            if not isinstance(items, dict) or any(not isinstance(expr, str) for expr in items.values()):
                raise ValueError("Expected an object of Boolean strings for " + group)
        rules = {}
        for name, expression in spec.get("rules", {}).items():
            if not RULE_NAME.fullmatch(name):
                raise ValueError("Invalid rule name: " + name)
            rules[name] = Expression(expression, rules).parse()
        self.rules = rules
        self.names, self.outputs = [], []
        self.literals = []
        def visit(node, found):
            if node[0] == "literal":
                if node[1] not in found:
                    found.append(node[1])
            else:
                for child in node[1:]:
                    visit(child, found)
        for group in GROUPS:
            for name, expression in spec.get(group, {}).items():
                node = Expression(expression, rules).parse()
                self.names.append((group, name))
                self.outputs.append(node)
                visit(node, self.literals)
        if not self.literals:
            raise ValueError("At least one evidence literal is required")
        words = sorted({w for phrase in self.literals for w in phrase})
        if len(words) > 255:
            raise ValueError("The byte projection supports up to 255 different literal lexemes")
        self.word_ids = {w: i+1 for i, w in enumerate(words)}
        self.patterns = [bytes(self.word_ids[w] for w in p) for p in self.literals]
        self.text_patterns = [self.literal_regex(words) for words in self.literals]
        self.anchors = [max(range(len(words)), key=lambda i: len(words[i])) for words in self.literals]
        explained = list(self.literals)
        for node in self.rules.values():
            visit(node, explained)
        self.explanation_patterns = {words: self.literal_regex(words)
                                     for words in explained[len(self.literals):]}

    @staticmethod
    def literal_regex(words):
        # Place the left boundary check after the fixed prefix so the regex
        # engine can use its native substring search on long documents.
        first = re.escape(words[0])
        def is_word(w):
            return bool(re.fullmatch(r"\w+", w))
        expr = first + (r"(?<!\w" + first + ")" if is_word(words[0]) else "")
        for left, right in zip(words, words[1:]):
            expr += (r"\s+" if is_word(left) and is_word(right) else r"\s*") + re.escape(right)
        if is_word(words[-1]):
            expr += r"(?!\w)"
        return re.compile(expr)

    def text_atoms(self, text):
        folded = text.casefold()
        return np.array([pattern.search(folded) is not None for pattern in self.text_patterns],
                        dtype=np.float32)

    def anchored_atoms(self, text):
        """Independently verify literal evidence, with a bounded projection fallback."""
        folded = text.casefold()
        flags = []
        for phrase, anchor in zip(self.literals, self.anchors):
            matched = anchored_match(folded, phrase, anchor)
            if matched is None:
                return self.atoms(self.project(text))
            flags.append(matched)
        return np.asarray(flags, dtype=np.float32)

    def project(self, text):
        # 0 represents every lexeme unused by this policy. Preserve occurrences
        # and their order, including punctuation between adjacent phrase words.
        return bytes(self.word_ids.get(w, 0) for w in LEXEME.findall(text.casefold()))

    def atoms(self, encoded):
        return np.array([p in encoded for p in self.patterns], dtype=np.float32)

    def reference(self, atoms):
        facts = dict(zip(self.literals, map(bool, atoms)))
        return np.array([value(node, facts) for node in self.outputs], dtype=np.float32)

    def result(self, bits):
        out = {"answers": {}, "category": None, "action": None}
        singular = {"categories": "category", "actions": "action"}
        for (group, name), bit in zip(self.names, bits):
            if group == "answers":
                out[group][name] = bool(bit)
            elif bit and out[singular[group]] is None:
                out[singular[group]] = name
        return out

    def explain(self, text, atoms):
        facts = dict(zip(self.literals, map(bool, atoms)))
        if self.explanation_patterns:
            folded = text.casefold()
            facts.update({words: pattern.search(folded) is not None
                          for words, pattern in self.explanation_patterns.items()})
        return {"matched": [" ".join(p) for p, present in facts.items() if present],
                "named_rules": {key: bool(value(node, facts)) for key,node in self.rules.items()}}


def load_tokenizer(path=None):
    import sentencepiece as spm
    path = Path(path) if path else Path(__file__).with_name("needle3-tokenizer.m8t.xz")
    return spm.SentencePieceProcessor(model_proto=read_tokenizer(path))


def needle_count(tokenizer, text, fast=True, memo=None):
    if not fast or len(text) < 4096 or " " not in text:
        return len(tokenizer.encode(text, out_type=int))
    # The SHA-pinned BPE has identity normalization, no added leading space,
    # byte fallback, and no regular piece with a non-initial metaspace. No
    # user-defined symbol spans ASCII space. Thus merges cannot span these
    # boundaries. Count each distinct space-prefixed segment once per request.
    first, _, rest = text.partition(" ")
    frequency = Counter(rest.split(" "))
    total = len(tokenizer.encode(first, out_type=int))
    if memo is not None:
        missing = {}
        for segment, repetitions in frequency.items():
            count = memo.prior.get(segment)
            if count is None:
                count = memo.counts.get(segment)
            if count is None:
                missing[segment] = repetitions
            else:
                total += repetitions * count
        frequency = missing
    if not frequency:
        return total
    segments = list(frequency)
    encodings = tokenizer.encode([" " + s for s in segments], out_type=int, num_threads=4)
    for segment, ids in zip(segments, encodings):
        count = len(ids)
        total += frequency[segment] * count
        if memo is not None and len(segment) <= 96:
            memo.counts[segment] = count
            if len(memo.counts) > memo.capacity:
                memo.counts.popitem(last=False)
    return total


class SegmentMemo:
    """Bounded, process-local counts of exact space-prefixed segments."""

    def __init__(self, capacity=8192, prior=None):
        self.counts = OrderedDict()
        self.capacity = capacity
        self.prior = prior if prior is not None else {}


def tokenizer_prior(tokenizer):
    """Seed reusable counts from the pinned vocabulary itself."""
    metaspace = "\u2581"
    words = (tokenizer.id_to_piece(i) for i in range(tokenizer.get_piece_size()))
    # The pinned model encodes each of these 4,200 segments as one token.
    # The differential test checks the entire set against SentencePiece.
    return {word[1:]: 1 for word in words if word.startswith(metaspace)}


def validate_count(count):
    if count > LIMIT:
        raise ValueError("Input has %d Needle tokens; maximum is %d" % (count, LIMIT))
    return count


def count_tokens(tokenizer, text, fast=False):
    return validate_count(needle_count(tokenizer, text, fast=fast))
