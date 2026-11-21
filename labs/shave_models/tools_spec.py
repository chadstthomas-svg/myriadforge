"""Frozen task spec for the voice-to-tool vertical slice (voicetools/1).

One spoken request -> an ordered list of typed tool calls, all learned
inference on one MA2450 (strict NCS1: the host only frames, tokenizes,
greedy-decodes, serializes, executes). This module is the CONTRACT the
fixtures, the trainer, the device evaluation, and the host reference all
read - change nothing here without bumping TASK_SPEC_VERSION and re-recording
every downstream artifact.

Domains: lighting / timers / media only (benign). Enum room/device/scene
slots, exactly one numeric slot per call (level or duration), max 2 ordered
calls per utterance, explicit out-of-scope rejection.
"""

TASK_SPEC_VERSION = "voicetools/1"

# --- graph windowing contract ----------------------------------------------
# The on-device graph input is fixed [1,1,40,200]: a 2.0 s, 40-bin log-mel
# window at 16 kHz (hop 160). Utterances longer than the window are covered by
# stride-1.0 s rolling windows (50% overlap); the v1 decision rule scores every
# window and takes ordered unique tools above threshold; latency is measured on
# the end-aligned (last) window. Host framing + window inference + JSON must
# land inside the p95 budget.
WINDOW_SECONDS = 2.0
WINDOW_STRIDE = 1.0
MEL_BINS = 40
SAMPLE_RATE = 16000

# --- slot enums -------------------------------------------------------------
ROOMS = ("kitchen", "living", "bedroom", "office", "hall")
DEVICES = ("speaker", "kitchen", "bedroom", "office", "tv")
SCENES = ("relax", "focus", "movie", "night")

# Numeric slots are step-10 percentages (0..100) and fixed timer durations;
# both are written as words because the fixtures are synthesized speech.
LEVELS = tuple(range(0, 101, 10))
LEVEL_WORDS = {0: "zero", 10: "ten", 20: "twenty", 30: "thirty", 40: "forty",
               50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty",
               90: "ninety", 100: "one hundred"}
DURATIONS = ((60, "one minute"), (120, "two minutes"), (300, "five minutes"),
             (600, "ten minutes"), (900, "fifteen minutes"))
DURATION_SECONDS = tuple(seconds for seconds, _ in DURATIONS)
DURATION_WORDS = {seconds: words for seconds, words in DURATIONS}

# --- the 12 benign tools ----------------------------------------------------
# tool -> slots; every tool accepts at most one numeric slot (level|seconds).
TOOLS = {
    "lights.set":    {"room": ROOMS, "level": LEVELS},
    "lights.on":     {"room": ROOMS},
    "lights.off":    {"room": ROOMS},
    "lights.scene":  {"room": ROOMS, "scene": SCENES},
    "timer.start":   {"seconds": DURATION_SECONDS},
    "timer.cancel":  {},
    "timer.pause":   {},
    "timer.resume":  {},
    "media.play":    {"device": DEVICES},
    "media.pause":   {},
    "media.stop":    {},
    "media.volume":  {"level": LEVELS},
}

NUMERIC_SLOT = {"lights.set": "level", "timer.start": "seconds", "media.volume": "level"}

# --- surface templates ------------------------------------------------------
# {room}/{device}/{scene} fill from the enums; {level}/{dur} from the word maps.
TEMPLATES = {
    "lights.set": ("set the {room} lights to {level} percent",
                   "turn the {room} lights to {level} percent",
                   "dim the {room} lights to {level} percent"),
    "lights.on":  ("turn on the {room} lights",
                   "lights on in the {room}"),
    "lights.off": ("turn off the {room} lights",
                   "lights off in the {room}",
                   "kill the {room} lights"),
    "lights.scene": ("set the {room} scene to {scene}",
                     "{scene} mode in the {room}",
                     "make the {room} lights {scene}"),
    "timer.start": ("start a {dur} timer",
                    "set a timer for {dur}",
                    "wake me in {dur}"),
    "timer.cancel": ("cancel the timer",
                     "cancel my timer"),
    "timer.pause":  ("pause the timer",
                     "hold the timer"),
    "timer.resume": ("resume the timer",
                     "continue the timer"),
    "media.play":  ("play music on the {device}",
                    "start music on the {device}",
                    "play something on the {device}"),
    "media.pause": ("pause the music",
                    "pause playback"),
    "media.stop":  ("stop the music",
                    "stop playback"),
    "media.volume": ("set the volume to {level} percent",
                     "turn the volume to {level} percent",
                     "volume {level}"),
}

PAIR_CONNECTIVES = (" and ", " and then ", " then ")

# --- negatives --------------------------------------------------------------
# Four negative kinds; all must NOT produce any tool call (empty-call verdict).
NEG_OUT_OF_DOMAIN = (
    "what is the weather tomorrow", "will it rain this weekend", "tell me a joke",
    "how old is the eiffel tower", "what time is it right now", "what day is it",
    "add coffee beans to my shopping list", "remind me to buy stamps",
    "how do I make pancakes", "read me the news headlines", "who won the game last night",
    "what is the capital of portugal", "tell me a fun fact about octopuses",
    "how far is the moon from earth", "translate good morning into french",
    "spell the word necessary", "what is twelve times fifteen", "flip a coin",
    "roll a dice for me", "pick a number between one and ten", "say something nice",
    "sing me a song", "tell me about your day", "do you like music",
)
NEG_IN_DOMAIN_UNSUPPORTED = (
    "turn on the television", "switch the tv to hdmi two", "lock the front door",
    "unlock the back door", "open the garage", "close the garage",
    "start the coffee maker", "brew a cup of coffee", "lower the blinds",
    "raise the blinds", "turn on the ceiling fan", "set the thermostat to seventy",
    "arm the security system", "disarm the alarm", "water the plants",
    "preheat the oven to four hundred", "turn on the porch light",
    "flash the kitchen lights", "start the dishwasher", "vacuum the living room",
)
NEG_UNKNOWN_SLOTS = (
    "set the garage lights to thirty percent", "set the attic lights to fifty percent",
    "turn on the basement lights", "lights on in the sunroom",
    "play music on the stereo", "play music on the radio", "start music on the cd player",
    "set the living room scene to party", "movie night mode party in the bedroom",
    "set the kitchen lights to thirty seven percent", "set the office lights to eighty five percent",
    "turn the volume to twelve percent", "volume ninety nine",
    "set a timer for seven minutes", "start a forty five second timer",
    "set the hall lights to forty two percent",
)

# --- TTS speaker conditions -------------------------------------------------
# Two installed desktop voices x rates x pitch shifts = deterministic condition
# cells; the held-out "speaker" proxy is entire held-out cells. Honest caveat:
# these are TTS fixtures, and every comparison (model vs host reference) is
# condition-matched on the same set.
TTS_VOICES = ("Microsoft David Desktop", "Microsoft Zira Desktop")
TTS_RATES = (-1, 0, 1, 2, 3)           # SAPI Rate property
TTS_PITCHES = ("-10%", "0%", "+10%")   # SSML prosody pitch

TEST_CELLS = {("Microsoft Zira Desktop", 0, "+10%"),
              ("Microsoft David Desktop", 3, "-10%"),
              ("Microsoft Zira Desktop", 2, "0%")}
VALID_CELLS = {("Microsoft David Desktop", -1, "+10%"),
               ("Microsoft Zira Desktop", 3, "-10%"),
               ("Microsoft David Desktop", 1, "0%")}

# --- fixture sizing ---------------------------------------------------------
SINGLES_PER_TOOL = 150     # x12 = 1800
PAIRS = 480                # ordered distinct-tool pairs
NEG_PER_KIND = 120         # x3 speech kinds = 360 speech negatives
SILENCE_NEGATIVES = 40     # appended programmatically, not synthesized

# --- decision rule ----------------------------------------------------------
# Utterance-level aggregation (v3 of the decision rule): a tool fires at its
# best window above FIRE_THRESHOLD; calls ordered by that window's position.
# 0.8 frozen 2026-11-20 from the dev (valid) sweep: balanced exact (79.8%)
# against false actions (12.5%); 0.9 trades 5.4 pts of exact for 5 pts of
# false actions. The false-action gate still needs a rejection-head training
# iteration (confidently-wrong-on-negatives is a model issue, not a threshold).
FIRE_THRESHOLD = 0.8

# --- acceptance gate (declare BEFORE opening the sealed set) ----------------
ACCEPTANCE = {
    "exact_match_min": 0.90,     # full-call exact match on the held-out cells
    "false_action_max": 0.01,    # any call produced on a negative
    "p95_ms_max": 300,           # end-of-speech -> finalized JSON, batch 1
    "min_negatives": 300,
}
