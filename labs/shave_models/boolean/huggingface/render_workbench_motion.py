"""Render the MagicEightBall hero as WebP or GIF using Chromium Canvas and FFmpeg.

The delivered animation is independent of the rendering tools. To rebuild it,
use Python with playwright and imageio-ffmpeg and an installed Chrome browser.
"""

from __future__ import annotations

import argparse
from base64 import b64decode, b64encode
from pathlib import Path
import subprocess

import imageio_ffmpeg
from playwright.sync_api import sync_playwright


ASSETS = Path(__file__).with_name("assets")
WIDTH, HEIGHT = 1903, 826
SCENE_SECONDS = 3
SECONDS = SCENE_SECONDS * 3

# Every coordinate is relative to the original workbench illustration. The
# diagram depicts the three support examples from the model card.
DRAW = r"""
async ({source, width, height, fixedCamera}) => {
  const poster = new Image();
  poster.src = source;
  await poster.decode();
  await document.fonts.ready;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  document.body.append(canvas);
  const ctx = canvas.getContext('2d', {alpha: false});
  const W = 1903, H = 826, TAU = Math.PI * 2;
  const cases = [
    {headline: 'Duplicate charge', detail: 'Review authorized', result: 'review_billing', answer: 'true',
     route: 'APPROVE', color: [160, 246, 183]},
    {headline: 'Late veto received', detail: 'Do not route', result: 'null', answer: 'false',
     route: 'HOLD', color: [255, 179, 123]},
    {headline: 'General help request', detail: 'No policy evidence', result: 'null', answer: 'false',
     route: 'ABSTAIN', color: [193, 174, 225]},
  ];
  const forward = [[840,618],[949,642],[922,659],[1104,692],[1155,699],
                   [1315,727],[1298,752],[1528,806]];
  const returnToBeacon = [[1155,699],[1104,691],[923,660],[950,642],[843,619],[807,620]];
  const controls = [[164,438],[418,438],[650,438]];
  const clamp = value => Math.min(1, Math.max(0, value));
  function smooth(from, to, value) {
    const t = clamp((value - from) / (to - from));
    return t * t * (3 - 2 * t);
  }
  const tint = (rgb, alpha) => `rgba(${rgb.join(',')},${clamp(alpha)})`;
  const font = (size, weight) => `${weight} ${size}px "Segoe UI", sans-serif`;
  const mono = (size, weight) => `${weight} ${size}px Consolas, monospace`;

  function glow(x, y, radius, rgb, power, spread = 1) {
    if (power <= 0) return;
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(spread, 1);
    const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
    gradient.addColorStop(0, tint(rgb, power));
    gradient.addColorStop(0.28, tint(rgb, power * 0.38));
    gradient.addColorStop(1, tint(rgb, 0));
    ctx.fillStyle = gradient;
    ctx.fillRect(-radius, -radius, radius * 2, radius * 2);
    ctx.restore();
  }

  function circle(x, y, radius, fill, stroke = null) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, TAU);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.stroke(); }
  }

  function pointAt(path, progress) {
    const steps = path.slice(1).map(([x, y], i) => Math.hypot(x - path[i][0], y - path[i][1]));
    let distance = clamp(progress) * steps.reduce((sum, step) => sum + step, 0);
    for (let i = 0; i < steps.length; i++) {
      if (distance <= steps[i] || i === steps.length - 1) {
        const amount = clamp(distance / steps[i]);
        return [path[i][0] + (path[i + 1][0] - path[i][0]) * amount,
                path[i][1] + (path[i + 1][1] - path[i][1]) * amount];
      }
      distance -= steps[i];
    }
  }

  function packet(path, progress, rgb, strength, scale) {
    if (progress < 0 || progress > 1 || strength <= 0) return;
    const [x, y] = pointAt(path, progress);
    for (let segment = 8; segment > 0; segment--) {
      const [sx, sy] = pointAt(path, progress - segment * 0.013);
      circle(sx, sy, (3.8 - segment * 0.25) * scale,
             tint(rgb, strength * (1 - segment / 10) * 0.62));
    }
    glow(x, y, 30 * scale, rgb, strength * 0.84);
    circle(x, y, 4.5 * scale, tint([252, 247, 232], strength));
  }

  function scene(time, local, index, step) {
    const angle = time / 9 * TAU;
    const zoom = 1.025 + (fixedCamera ? 0 : (1 - Math.cos(angle)) * 0.010);
    ctx.save();
    const drift = fixedCamera ? 0 : Math.sin(angle);
    ctx.translate(W / 2 + 8 * drift, H / 2 + 4 * drift);
    ctx.scale(zoom, zoom);
    ctx.translate(-W / 2, -H / 2);
    ctx.drawImage(poster, 0, 0, W, H);
    ctx.globalCompositeOperation = 'screen';

    for (let i = 0; i < 24; i++) {
      const theta = angle * (i % 2 + 1) + i * 2.399;
      const x = 1030 + (i * 77 % 460) + 26 * Math.sin(theta);
      const y = 145 + (i * 31 % 352) + 19 * Math.cos(theta + 0.2);
      const brightness = 0.09 + 0.21 * ((Math.cos(theta + i) + 1) / 2);
      glow(x, y, 3 + i % 3, [249, 198, 123], brightness);
      circle(x, y, 0.7 + i % 3 * 0.35, tint([245, 206, 157], brightness));
    }

    const settle = smooth(0.9, 1.8, local) * (1 - smooth(2.55, 3, local));
    const ambient = 0.5 + 0.5 * Math.sin(angle * 2);
    glow(1270, 152, 167, [246, 169, 97], 0.10 + ambient * 0.08, 1.30);
    glow(1078, 350, 52, [252, 191, 106], 0.15 + ambient * 0.23);
    glow(1057, 466, 78, step.color, 0.13 + settle * 0.21);
    glow(1329, 532, 35, [157, 250, 151], 0.11 + ambient * 0.12);
    glow(1424, 535, 37, [157, 250, 151], 0.11 + ambient * 0.12);
    glow(1150, 694, 85, step.color, settle * 0.31, 1.55);
    glow(797, 615, 61, [245, 113, 75], index === 1 ? 0.17 + settle * 0.55 : 0.14);

    if (index === 0) {
      packet(forward, (local - 0.45) / 2.5, step.color, 0.93, 1.0);
      packet(forward, (local - 0.89) / 2.5, step.color, 0.56, 0.67);
    } else if (index === 1) {
      packet(forward.slice(0, 5), (local - 0.39) / 0.72, [176, 228, 175], 0.83, 0.73);
      packet(returnToBeacon, (local - 1.38) / 1.05, step.color, 0.98, 1.0);
    } else {
      packet(forward.slice(0, 4), (local - 0.65) / 1.45, step.color, 0.54, 0.54);
    }

    // A slow raster pulse stays inside the illustrated oscilloscope glass.
    ctx.beginPath();
    ctx.moveTo(1697, 173); ctx.lineTo(1903, 145); ctx.lineTo(1903, 367);
    ctx.lineTo(1716, 373); ctx.closePath(); ctx.clip();
    const scanY = 143 + 242 * ((time * 2 / 9) % 1);
    const scan = ctx.createLinearGradient(0, scanY - 21, 0, scanY + 9);
    scan.addColorStop(0, 'rgba(118,236,150,0)');
    scan.addColorStop(0.76, 'rgba(118,236,150,0.22)');
    scan.addColorStop(1, 'rgba(118,236,150,0)');
    ctx.fillStyle = scan;
    ctx.fillRect(1696, scanY - 21, 207, 31);
    ctx.restore();
  }

  function trace(local, index, step) {
    const reveal = smooth(0, 0.24, local) * (1 - smooth(2.74, 3, local));
    const progress = smooth(0.44, 1.69, local);
    const answerReveal = smooth(1.48, 1.9, local);
    const fade = ctx.createLinearGradient(45, 0, 815, 0);
    fade.addColorStop(0, 'rgba(14,10,16,0.58)');
    fade.addColorStop(0.67, 'rgba(14,10,16,0.24)');
    fade.addColorStop(1, 'rgba(14,10,16,0)');
    ctx.fillStyle = fade;
    ctx.fillRect(0, 0, 815, H);

    ctx.fillStyle = '#d7c8ba';
    ctx.font = mono(24, 600);
    ctx.letterSpacing = '5px';
    ctx.fillText('DECISION TRACE', 141, 170);
    ctx.letterSpacing = '0px';
    ctx.lineWidth = 1.8;
    ctx.strokeStyle = 'rgba(228,209,184,0.35)';
    ctx.beginPath(); ctx.moveTo(141, 191); ctx.lineTo(694, 191); ctx.stroke();

    ctx.save();
    ctx.globalAlpha = reveal;
    ctx.translate(0, (1 - reveal) * 10);
    ctx.fillStyle = '#fbf1e5';
    ctx.font = font(53, 650);
    ctx.fillText(step.headline, 140, 278);
    ctx.fillStyle = '#d8c7cd';
    ctx.font = font(30, 400);
    ctx.fillText(step.detail, 143, 329);
    ctx.restore();

    const [inX, graphY] = controls[0], [outX] = controls[2];
    ctx.lineWidth = 2.8;
    ctx.strokeStyle = 'rgba(226,205,202,0.32)';
    ctx.beginPath(); ctx.moveTo(inX, graphY); ctx.lineTo(outX, graphY); ctx.stroke();
    ctx.strokeStyle = tint(step.color, 0.72 * reveal);
    ctx.beginPath(); ctx.moveTo(inX, graphY); ctx.lineTo(inX + (outX - inX) * progress, graphY); ctx.stroke();

    ctx.font = mono(20, 500);
    ctx.textAlign = 'center';
    for (const [i, [x, y]] of controls.entries()) {
      const active = smooth(i / 2 - 0.04, i / 2 + 0.03, progress) * reveal;
      glow(x, y, 39, step.color, active * 0.21);
      ctx.lineWidth = 2;
      circle(x, y, 21, '#231b27', tint(step.color, 0.34 + active * 0.5));
      circle(x, y, 4.5, tint(step.color, 0.36 + active * 0.64));
      ctx.fillStyle = '#e1d2cc';
      ctx.fillText(['EVIDENCE', 'RULES', 'ACTION'][i], x, y + 57);
    }
    if (progress > 0.02 && progress < 0.98) packet(controls, progress, step.color, 0.76, 0.68);
    ctx.textAlign = 'left';

    ctx.save();
    ctx.globalAlpha = answerReveal * reveal;
    ctx.fillStyle = tint(step.color, 0.12);
    ctx.strokeStyle = tint(step.color, 0.46);
    ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.roundRect(141, 559, 164, 49, 24); ctx.fill(); ctx.stroke();
    circle(166, 584, 4.8, tint(step.color, 1));
    ctx.fillStyle = tint(step.color, 1);
    ctx.font = mono(22, 600);
    ctx.fillText(step.route, 187, 590);
    ctx.fillStyle = '#f0e5dd';
    ctx.font = mono(26, 400);
    ctx.fillText('action: ' + step.result, 329, 591);
    ctx.fillStyle = '#bcaeb7';
    ctx.font = mono(19, 400);
    ctx.fillText('may_route = ' + step.answer, 142, 649);
    ctx.restore();

    for (let i = 0; i < 3; i++) {
      const x = 142 + i * 62;
      ctx.fillStyle = 'rgba(225,199,196,0.22)';
      ctx.fillRect(x, 704, 47, 3);
      if (i === index) {
        ctx.fillStyle = tint(step.color, 0.95);
        ctx.fillRect(x, 704, 47 * local / 3, 3);
      }
    }
  }

  window.renderMotionFrame = (time, quality) => {
    const index = Math.floor(time / 3) % cases.length;
    const local = time % 3;
    ctx.setTransform(width / W, 0, 0, height / H, 0, 0);
    scene(time, local, index, cases[index]);
    trace(local, index, cases[index]);
    return canvas.toDataURL('image/jpeg', quality).split(',')[1];
  };
}
"""


def encoder_for(args: argparse.Namespace) -> list[str]:
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-threads", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", str(args.fps),
               "-i", "pipe:0", "-threads", "1", "-an"]
    if args.output.suffix.lower() == ".gif":
        command.extend(["-filter_complex_threads", "1", "-filter_complex",
                        "split[video][colors];[colors]palettegen=stats_mode=diff[palette];"
                        "[video][palette]paletteuse=dither=bayer:bayer_scale=3"])
    else:
        command.extend(["-c:v", "libwebp_anim", "-q:v", str(args.quality),
                        "-compression_level", "4", "-preset", "photo"])
    return command + ["-loop", "0", str(args.output)]


def render(args: argparse.Namespace) -> None:
    height = round(args.width * HEIGHT / WIDTH)
    source = f"data:image/webp;base64,{b64encode(args.source.read_bytes()).decode('ascii')}"
    encoder = encoder_for(args)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.browser,
                                             args=["--disable-background-networking", "--renderer-process-limit=1"])
        page = browser.new_page(viewport={"width": args.width, "height": height})
        page.evaluate(DRAW, {"source": source, "width": args.width, "height": height,
                             "fixedCamera": args.fixed_camera})
        with subprocess.Popen(encoder, stdin=subprocess.PIPE) as ffmpeg:
            for frame in range(SECONDS * args.fps):
                encoded = page.evaluate("([time, quality]) => renderMotionFrame(time, quality)",
                                        [frame / args.fps, 0.94])
                ffmpeg.stdin.write(b64decode(encoded))
            ffmpeg.stdin.close()
            if ffmpeg.wait():
                raise RuntimeError(f"FFmpeg could not encode {args.output}")
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ASSETS / "workbench.webp")
    parser.add_argument("--output", type=Path, default=ASSETS / "workbench-motion.webp")
    parser.add_argument("--browser", default="chrome")
    parser.add_argument("--width", type=int, default=1136)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--fixed-camera", action="store_true", help="Reduce GIF size by holding the camera still")
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        parser.error("Use an output path distinct from the original still")
    if args.output.suffix.lower() not in (".gif", ".webp"):
        parser.error("Use a .webp or .gif output")
    if args.width < 320 or args.fps < 1 or not 1 <= args.quality <= 100:
        parser.error("Use width ≥320, fps ≥1 and quality from 1 to 100")
    render(args)


if __name__ == "__main__":
    main()
