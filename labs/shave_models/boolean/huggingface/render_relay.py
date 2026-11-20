"""Render the model card's original, animated support-policy illustration."""

from functools import lru_cache
import math
import os
from pathlib import Path
from random import Random

from PIL import Image, ImageDraw, ImageFont


W, H = 1080, 444
MUTE = "#a39eb8"
WHITE = "#fcf3dc"
MINT = "#8eefd0"
CORAL = "#ffab91"
AMBER = "#f6ca85"
DARK = "#110e21"
SCENES = (
    (("Duplicate charge.", "Review authorized."), "BILLING REVIEW", "true", "billing", "review_billing", MINT),
    (("Duplicate charge.", "Review authorized.", "Do not route."), "ACTION CLEARED", "false", "billing", "null", CORAL),
    (("General help requested.",), "NO CASE MATCH", "false", "null", "null", AMBER),
)


@lru_cache(maxsize=None)
def font(size, family="sans"):
    names = {
        "sans": ("segoeui.ttf", "DejaVuSans.ttf"),
        "bold": ("segoeuib.ttf", "DejaVuSans-Bold.ttf"),
        "mono": ("consola.ttf", "DejaVuSansMono.ttf"),
    }
    system_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name in names[family]:
        try:
            return ImageFont.truetype(str(system_fonts / name) if (system_fonts / name).exists() else name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def stroke(draw, points, color, width=1):
    draw.line(points, fill=color, width=width, joint="curve")


def dot(draw, x, y, r, color):
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def label(draw, x, y, value, color=MUTE):
    draw.text((x, y), value, font=font(13, "mono"), fill=color)


def panel(draw, xy):
    x, y, right, bottom = xy
    draw.rounded_rectangle((x + 5, y + 7, right + 5, bottom + 7), 15, fill="#090916")
    draw.rounded_rectangle(xy, 15, fill="#171427", outline="#3c3551", width=2)
    stroke(draw, ((x + 17, y + 1), (right - 17, y + 1)), "#514864")


def make_bench():
    width, height = 270, 111
    rng = Random(82)
    pixels = []
    for y in range(height):
        for x in range(width):
            u, v = x / width, y / height
            glow = math.exp(-8 * ((u - 0.51) ** 2 + (v - 0.43) ** 2))
            noise = rng.randrange(-1, 2)
            pixels.append(tuple(int(low + span * glow) + noise for low, span in ((10, 26), (9, 20), (20, 35))))
    base = Image.new("RGB", (width, height))
    base.putdata(pixels)
    base = base.resize((W, H), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(base)
    for x in range(-450, W, 54):
        stroke(draw, ((x, 310), (x + 460, H)), "#2e2945")
    for y in (310, 330, 355, 391, 438):
        stroke(draw, ((0, y), (W, y)), "#302946")
    for x in (11, W - 11):
        stroke(draw, ((x, 47), (x, 365)), "#3e3658")
    for x in (24, W - 24):
        for y in (23, H - 23):
            dot(draw, x, y, 3, "#544964")
    panel(draw, (37, 83, 339, 334))
    panel(draw, (743, 83, W - 39, 334))
    panel(draw, (94, 370, W - 96, 416))
    label(draw, 62, 102, "SUPPORT REQUEST")
    label(draw, 768, 102, "POLICY OUTPUT")
    stroke(draw, ((61, 137), (315, 137)), "#38324c")
    stroke(draw, ((768, 137), (1016, 137)), "#38324c")
    for a, b in (((339, 213), (495, 213)), ((623, 213), (743, 213))):
        stroke(draw, (a, b), "#100e1e", 10)
        stroke(draw, (a, b), "#52486b", 2)
    for x in (359, 387, 713, 726):
        dot(draw, x, 213, 3, "#6d6080")
    return base


def make_shell():
    size = 232
    image = Image.new("RGBA", (size, size))
    pixels = []
    for y in range(size):
        for x in range(size):
            u, v = (x - 115.5) / 114.5, (y - 115.5) / 114.5
            distance = u * u + v * v
            if distance > 1:
                pixels.append((0, 0, 0, 0))
                continue
            depth = math.sqrt(max(0, 1 - distance))
            light = max(0, -0.50 * u - 0.49 * v + 0.43 * depth)
            sheen = math.exp(-30 * ((u + 0.42) ** 2 + (v + 0.55) ** 2))
            rim = ((1 - depth) ** 4) * max(0.05, 0.65 * u - 0.83 * v + 0.5)
            color = (10 + 47 * light + 102 * sheen + 59 * rim,
                     10 + 37 * light + 96 * sheen + 104 * rim,
                     26 + 65 * light + 119 * sheen + 90 * rim)
            pixels.append(tuple(min(255, int(channel)) for channel in color) + (255,))
    image.putdata(pixels)
    draw = ImageDraw.Draw(image)
    draw.ellipse((67, 31, 166, 129), fill="#f8efcf", outline="#a29587", width=2)
    draw.text((116, 79), "8", anchor="mm", font=font(75, "bold"), fill="#1a1528")
    draw.arc((13, 9, 221, 221), 121, 239, fill="#8f70ab", width=2)
    draw.arc((10, 11, 225, 224), 277, 342, fill="#99c3bf", width=2)
    return image


def draw_eight(image, shell, y, blink, tint):
    draw = ImageDraw.Draw(image)
    x = 555
    draw.ellipse((x - 118, 326, x + 122, 353), fill="#0d0c1b")
    for side in (-1, 1):
        arm = [(x + side * 93, y + 29), (x + side * 123, y + 49), (x + side * 144, y + 26)]
        stroke(draw, arm, "#090916", 13)
        stroke(draw, arm, "#796994", 4)
        dot(draw, *arm[-1], 10, "#b5a3cb")
        dot(draw, *arm[-1], 6, "#352b4b")
        bx = x + side * 61
        draw.rounded_rectangle((bx - 31, y + 91, bx + 26, y + 109), 7,
                               fill="#302843", outline="#877397", width=2)
        for xx in (bx - 17, bx + 13):
            dot(draw, xx, y + 111, 6, tint)
            dot(draw, xx, y + 111, 3, "#362b4e")
    image.paste(shell, (x - 116, y - 116), shell)
    draw.rounded_rectangle((x - 62, y + 29, x + 63, y + 62), 16,
                           fill="#100f22", outline="#6b557b", width=2)
    eye = 2 if blink else 8
    for xx in (x - 24, x + 25):
        draw.rounded_rectangle((xx - 10, y + 45 - eye, xx + 10, y + 45 + eye), eye,
                               fill=tint)
    stroke(draw, ((x - 9, y + 77), (x + 9, y + 77)), "#9482a7", 2)


def draw_relay(draw, x, y, tint):
    for offset in (-17, -7, 3, 13):
        stroke(draw, ((x - 34, y + offset), (x - 23, y + offset)), "#dfb984", 3)
        stroke(draw, ((x + 23, y + offset), (x + 34, y + offset)), "#dfb984", 3)
    stroke(draw, ((x + 15, y - 23), (x + 28, y - 39)), "#dfb984", 2)
    dot(draw, x + 28, y - 39, 5, tint)
    draw.rounded_rectangle((x - 25, y - 25, x + 25, y + 25), 9,
                           fill="#183d38", outline="#9bd7b6", width=2)
    draw.rounded_rectangle((x - 18, y - 18, x + 18, y + 18), 6,
                           fill="#25354b", outline="#426e6c")
    for xx in (x - 7, x + 7):
        dot(draw, xx, y - 1, 3, tint)
    stroke(draw, ((x - 5, y + 9), (x + 5, y + 9)), "#a58eaa", 1)


def frame(base, shell, index, phase):
    evidence, title, route, category, action, tint = SCENES[index]
    image = base.copy()
    draw = ImageDraw.Draw(image)
    for line, words in enumerate(evidence):
        top = 165 + line * 45
        stroke(draw, ((62, top + 30), (315, top + 30)), "#2d283e")
        dot(draw, 69, top + 10, 4, tint if line == len(evidence) - 1 else "#887d9d")
        draw.text((82, top), words, font=font(18), fill=WHITE)
    draw.text((768, 154), title, font=font(25, "bold"), fill=tint)
    for row, (key, value) in enumerate((("may_route", route), ("category", category), ("action", action))):
        top = 219 + row * 29
        label(draw, 768, top, key)
        draw.text((1015, top), value, anchor="ra", font=font(16, "mono"), fill=WHITE)
    for x, value, enabled in ((122, "CASE FOUND", index < 2), (415, "AUTHORIZED", index < 2), (723, "LATE VETO", index == 1)):
        dot(draw, x, 393, 5, tint if enabled else "#6c6382")
        label(draw, x + 16, 384, value, WHITE if enabled else MUTE)
    for fraction in (phase, (phase + 0.42) % 1):
        xx = 346 + 391 * fraction
        dot(draw, xx, 213, 6, "#383052")
        dot(draw, xx, 213, 3, tint)
    float_phase = 2 * math.pi * phase
    draw_eight(image, shell, 214 + int(5 * math.sin(float_phase)), 0.43 < phase < 0.50, tint)
    draw_relay(draw, 415 + int(6 * math.sin(float_phase)), 123 + int(7 * math.cos(float_phase)), tint)
    return image


def render(output):
    output.mkdir(parents=True, exist_ok=True)
    base = make_bench()
    shell = make_shell()
    still = frame(base, shell, 0, 0.20)
    still.save(output / "relay-decisions-still.webp", quality=91)
    durations, frames = [], []
    phase_count = 14
    for index in range(len(SCENES)):
        for step in range(phase_count):
            frames.append(frame(base, shell, index, step / phase_count))
            durations.append(150 if step < phase_count - 1 else 1200)
    frames[0].save(output / "relay-decisions.webp", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, quality=86, method=4)


if __name__ == "__main__":
    render(Path(__file__).with_name("assets"))
