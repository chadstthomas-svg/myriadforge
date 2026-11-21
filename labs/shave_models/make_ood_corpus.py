"""Broad OOD text corpus for the voicetools reject head (train-side only).

The sealed run showed unseen in-domain-shaped language fires the tool heads
at 50-87%. This generator produces a DISJOINT training corpus of out-of-domain
utterances: 26 topic families x phrase frames x slot-ish decorations, seeded -
none of these texts appear in the sealed set (different families, different
frames, disjoint seeds). Rendered via render_voice_tools.ps1 like every
fixture set; ~600 clips.

Usage (main env, from the lab dir):
    python make_ood_corpus.py --out D:/models/ncs-shave-research/voicetools-ood
    powershell -NoProfile -ExecutionPolicy Bypass -File render_voice_tools.ps1 -Plan <out>/plan.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from tools_spec import TTS_PITCHES, TTS_RATES, TTS_VOICES

LAB = Path(__file__).resolve().parent
SEED = 4242

TOPICS = [
    ("weather", ["what is the forecast for {}", "will it rain in {}", "how hot is it in {}", "is it windy outside"]),
    ("sports", ["what is the score of the {} game", "when do the {} play next", "who won the {} match", "highlights from the {} game"]),
    ("news", ["read the top story", "what is happening in the world today", "any headlines about {}", "summarize today's news"]),
    ("food", ["how do I cook {}", "what goes well with {}", "recipe for {}", "is {} healthy"]),
    ("travel", ["flights to {}", "hotels near {}", "things to do in {}", "how far is {} from here"]),
    ("shopping", ["find me a deal on {}", "price of {}", "reviews for {}", "where can I buy {}"]),
    ("finance", ["stock price of {}", "convert {} dollars to euros", "what is inflation", "mortgage rates today"]),
    ("health", ["symptoms of {}", "how much water should I drink", "best exercise for {}", "what causes {}"]),
    ("music", ["who sings {}", "lyrics of {}", "play songs like {}", "what genre is {}"]),
    ("movies", ["what time is {} playing", "who directed {}", "cast of {}", "is {} any good"]),
    ("math", ["what is {} plus {}", "multiply {} by {}", "square root of {}", "percent of {} is {}"]),
    ("time", ["what time is it in {}", "how many days until {}", "sunrise time tomorrow", "what year is it"]),
    ("directions", ["how do I get to {}", "directions to the nearest {}", "is {} open now", "walking time to {}"]),
    ("social", ["post a status about {}", "message {} on chat", "who liked my photo", "follow {} for me"]),
    ("device-control-nondomain", ["increase the screen brightness", "turn on do not disturb", "change my wallpaper", "enable dark mode"]),
    ("communication", ["call {}", "video call {}", "email {} about the meeting", "reply to {}"]),
    ("photos", ["show my photos from {}", "delete this picture", "crop the image", "make a collage of {}"]),
    ("alarms-nondomain", ["snooze for five more minutes", "change my wake up time to seven", "delete my morning alarm"]),
    ("reminders", ["remind me to {} tomorrow", "note that I need {}", "add {} to my list"]),
    ("definitions", ["what does {} mean", "define {}", "use {} in a sentence", "origin of the word {}"]),
    ("animals", ["how fast is a {}", "what do {} eat", "lifespan of a {}", "are {} dangerous"]),
    ("science", ["how does photosynthesis work", "what is the speed of light", "why is the sky blue", "what is dark matter"]),
    ("history", ["when did {} begin", "who was {}", "what happened in {}", "why did the {} fall"]),
    ("smalltalk", ["good morning", "how are you today", "nice weather we are having", "tell me something interesting"]),
    ("nonsense", ["the purple elephant sings quietly", "seven green ideas dream loudly", "bananas negotiate on tuesdays", "the window remembers yesterday"]),
    ("home-misc", ["is the dryer done yet", "did anyone ring the bell", "the thermostat reads sixty eight", "who left the garage open"]),
]
FILLERS = ["paris", "london", "chicken soup", "the dog", "my brother", "spain", "jazz", "tuesday",
           "the library", "coffee", "mountains", "tennis", "the bus", "winter", "the museum",
           "new york", "the beach", "sunday", "chocolate", "the moon"]

FRAMES = ["{}", "please {}", "hey assistant, {}", "can you {}", "{} for me", "I want to know {}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("D:/models/ncs-shave-research/voicetools-ood"))
    ap.add_argument("--per-topic", type=int, default=24)
    args = ap.parse_args()
    rng = random.Random(SEED)
    rows = []
    idx = 0
    for topic, frames in TOPICS:
        for _ in range(args.per_topic):
            base = rng.choice(frames)
            if "{}" in base:
                text = base.format(rng.choice(FILLERS), *(rng.choice(FILLERS) for _ in range(base.count("{}") - 1)))
            else:
                text = base
            text = rng.choice(FRAMES).format(text)
            voice = TTS_VOICES[rng.randrange(len(TTS_VOICES))]
            rate = TTS_RATES[rng.randrange(len(TTS_RATES))]
            pitch = TTS_PITCHES[rng.randrange(len(TTS_PITCHES))]
            rows.append({"id": f"ood{idx:05d}", "wav": f"ood{idx:05d}.wav", "text": text,
                         "calls": [], "kind": f"ood_{topic}", "voice": voice, "rate": rate,
                         "pitch": pitch, "split": "train", "out": str(args.out / f"ood{idx:05d}.wav")})
            idx += 1
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "plan.jsonl").open("w", encoding="utf-8") as dst:
        for r in rows:
            dst.write(json.dumps(r) + "\n")
    # the trainer's extra-negatives adapter reads manifest.jsonl
    (args.out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"OOD corpus: {len(rows)} clips across {len(TOPICS)} families (seed {SEED}, "
          f"disjoint from the sealed set)")
    print(f"  render: powershell -NoProfile -ExecutionPolicy Bypass -File "
          f"{LAB / 'render_voice_tools.ps1'} -Plan {args.out / 'plan.jsonl'}")


if __name__ == "__main__":
    main()
