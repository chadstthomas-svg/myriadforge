"""Build the voicetools SEALED negative set (>=300 held-out negatives).

Acceptance requires >=300 independent held-out negatives; the diagnostic
split has 74. This builder emits NEW speech negatives (unseen out-of-domain
texts, unseen unsupported devices, unseen off-grid slot values - rendered via
SAPI in the SAME 30 condition cells) plus reused vad-suite SEALED noise mixes
(unseen generator seeds the trainer never saw). Rendered once, then frozen:
the sealed manifest is only ever evaluated, never trained on.

Usage (main env, from the lab dir):
    python make_sealed_negatives.py --out D:/models/ncs-shave-research/voicetools-sealed
    (then render_voice_tools.ps1 via the printed plan path, then verify)
"""
import argparse
import json
import random
from pathlib import Path

from tools_spec import TTS_PITCHES, TTS_RATES, TTS_VOICES

RATE = 16000
LAB = Path(__file__).resolve().parent
SEED = 9000

NEG_OOD_NEW = (
    "read me my messages", "send a text to maria", "what is on my calendar today",
    "schedule a meeting for tomorrow", "cancel my three o clock",
    "navigate to the airport", "what is the traffic like downtown",
    "order a large pizza", "track my package", "what is the exchange rate for euros",
    "play my workout playlist", "like this video", "subscribe to the channel",
    "what is the score of the game", "who is winning", "open the camera app",
    "take a screenshot", "set an alarm for six thirty", "what is my heart rate",
    "start a workout", "pause my meditation", "how many calories in an apple",
    "what is the protein in an egg", "define serendipity", "synonym for happy",
    "calculate the tip on forty dollars", "split the bill three ways",
    "what is seventeen percent of two hundred", "convert ten kilometers to miles",
    "what is the boiling point of water", "when was the moon landing",
    "who wrote pride and prejudice", "tell me a riddle", "guess what I am thinking",
    "help me relax", "play rain sounds", "what is the fastest animal",
    "how tall is mount everest", "what is my ip address", "scan for wifi networks",
    "turn on airplane mode", "pair my headphones", "what is the battery level",
    "update my apps", "clear my history", "what is two plus two",
    "sing the alphabet", "say hello in japanese", "count to twenty",
    "what is the largest planet", "name the colors of the rainbow",
)
NEG_UNSUPPORTED_NEW = (
    "feed the cat", "order dog food", "water the lawn tomorrow", "mow the grass",
    "start the washing machine", "dry the towels", "iron my shirt",
    "make toast", "boil water for tea", "blend the smoothie",
    "open the curtains", "close the window blinds", "light the fireplace",
    "start the sauna", "fill the bathtub", "flush the toilet",
    "start the pool pump", "heat the hot tub", "open the skylight",
    "extend the awning", "start the sprinklers", "charge my scooter",
    "inflate the tires", "check the oil", "honk the horn",
    "lock the car", "start the engine", "open the trunk",
    "ring the doorbell", "let the guest in", "call the elevator",
    "start the escalator", "raise the flag", "spin the windmill",
    "start the generator", "turn on the heater", "ignite the stove",
    "warm the plates", "chill the wine", "crush the ice",
)
NEG_UNKNOWN_NEW = (
    "set the pantry lights to sixty five percent", "set the balcony lights to fifteen percent",
    "turn on the cellar lights", "lights on in the loft", "dim the attic lights to five percent",
    "set the den lights to ninety five percent", "turn the foyer lights to eight percent",
    "play music on the projector", "play music on the amplifier", "start music on the receiver",
    "play something on the turntable", "start music on the cassette deck",
    "set the kitchen scene to carnival", "set the bedroom scene to disco",
    "focus party in the office", "make the hall lights romance",
    "set the living lights to thirty three percent", "turn the office lights to seventy seven percent",
    "set the hall lights to eleven percent", "volume thirteen", "turn the volume to forty six percent",
    "volume zero point five", "set a timer for twelve seconds", "start a ninety nine second timer",
    "set a timer for twenty minutes", "wake me in seven minutes", "start a three hour timer",
    "set the bathroom lights to sixty six percent", "lights off in the garage apartment",
    "turn the closet lights to twenty three percent",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("D:/models/ncs-shave-research/voicetools-sealed"))
    args = ap.parse_args()
    rng = random.Random(SEED)
    rows = []
    idx = 0

    def add(text, kind):
        nonlocal idx
        voice = TTS_VOICES[rng.randrange(len(TTS_VOICES))]
        rate = TTS_RATES[rng.randrange(len(TTS_RATES))]
        pitch = TTS_PITCHES[rng.randrange(len(TTS_PITCHES))]
        rows.append({"id": f"sealed{idx:05d}", "wav": f"sealed{idx:05d}.wav", "text": text,
                     "calls": [], "kind": kind, "voice": voice, "rate": rate,
                     "pitch": pitch, "split": "sealed", "out": str(args.out / f"sealed{idx:05d}.wav")})
        idx += 1

    for t in NEG_OOD_NEW:
        add(t, "sealed_out_of_domain")
    for t in NEG_UNSUPPORTED_NEW:
        add(t, "sealed_unsupported")
    for t in NEG_UNKNOWN_NEW:
        add(t, "sealed_unknown_slot")

    # unseen noise mixes from the vad SEALED set (trainer never saw these seeds)
    vad = Path("D:/models/ncs-shave-research/vad")
    sealed_noise = [json.loads(line) for line
                    in (vad / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                    if json.loads(line)["kind"].startswith("mix_")]
    for r in sealed_noise:
        rows.append({"id": f"sealed{idx:05d}", "wav": str(vad / r["wav"]), "text": "",
                     "calls": [], "kind": "sealed_noise", "voice": "", "rate": 0,
                     "pitch": "0%", "split": "sealed",
                     "out": str(vad / r["wav"])})
        idx += 1

    rng.shuffle(rows[:idx - len(sealed_noise)])  # shuffle only the synthesized rows
    args.out.mkdir(parents=True, exist_ok=True)
    synth = [r for r in rows if r["kind"] != "sealed_noise"]
    with (args.out / "plan.jsonl").open("w", encoding="utf-8") as dst:
        for r in synth:
            dst.write(json.dumps(r) + "\n")
    (args.out / "manifest-sealed.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"sealed set: {len(rows)} negatives (>=300 required)")
    print(f"  {kinds}")
    print(f"  render the {len(synth)} synthesized clips:")
    print(f"    powershell -NoProfile -ExecutionPolicy Bypass -File {LAB / 'render_voice_tools.ps1'} -Plan {args.out / 'plan.jsonl'}")
    print(f"  then verify + evaluate with eval_voicetools.py --fixtures {args.out} "
          f"--manifest manifest-sealed.jsonl (silence rows carry absolute wav paths)")


if __name__ == "__main__":
    main()
