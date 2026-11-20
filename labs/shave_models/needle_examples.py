"""Original tool-proposal examples for a Cactus Needle UI adapter."""
import argparse
import json
from pathlib import Path


def schema(name, field, values=None):
    spec = {"type": "string"}
    if values:
        spec["enum"] = values
    return {"name": name, "parameters": {"type": "object", "properties": {field: spec},
                                          "required": [field]}}


TOOLS = [schema("find_control", "label"), schema("scroll_view", "direction", ["up", "down"])]


def examples(test):
    labels = ["Bookmarks", "Keyboard shortcuts", "Reader mode"] if test else [
        "Favorites", "Downloads", "Preferences", "Print preview", "Reading list", "Captions"]
    prompts = ["Highlight the control for {}", "Where is the {} control?"] if test else [
        "Find the {} button", "Show me where {} is", "Locate the control named {}"]
    rows = [(p.format(s), [{"name": "find_control", "arguments": {"label": s}}])
            for p in prompts for s in labels]
    for direction in ["up", "down"]:
        phrases = ["Scroll this panel {} please"] if test else [
            "Scroll {}", "Move the page {}", "Scroll the current view {}", "Please scroll {}"]
        rows.extend((p.format(direction), [{"name": "scroll_view", "arguments":
                                             {"direction": direction}}]) for p in phrases)
    negatives = ["Count these leaves", "What time is it in Tokyo?"] if test else [
        "Sing a song", "Convert ten inches to centimeters", "What is the forecast?",
        "List three colors", "Tell me about the moon", "What is two times four?"]
    rows.extend((p, []) for p in negatives)
    return [dict(query=q, tools=TOOLS, answers=a) for q, a in rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    for name in ("train", "test"):
        rows = examples(name == "test")
        (args.output / ("ui-" + name + ".jsonl")).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
