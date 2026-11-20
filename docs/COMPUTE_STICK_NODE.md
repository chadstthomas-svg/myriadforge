# Compute Stick as a Host-Controlled Node

**Goal:** scope controlling an Intel Compute Stick (stick PC, Core m3-6Y30 /
4 GB / 64 GB eMMC) from the main machine — including what the HDMI port can
and cannot do — and slot the stick into myriadforge as a remote x86 benchmark
node. Scoped 2026-11-16, companion to [NCS2_ROADMAP.md](NCS2_ROADMAP.md).

---

## The HDMI reality check

**HDMI is unidirectional video.** Both machines' HDMI ports are *outputs*;
you cannot plug the stick's HDMI-out into another PC's HDMI port and get
anything. A laptop HDMI port accepts cables, not signals. So "control via
the HDMI port of another machine" decomposes into two honest planes:

| Plane | Channel | What it does |
|---|---|---|
| **Display/telemetry** (stick → host) | HDMI-out → USB3 capture dongle → host sees stick's screen as a webcam-class video device | viewing, provisioning, BIOS sessions, screen-state telemetry |
| **Control** (host → stick) | network: SSH / VNC / RDP over Wi-Fi or USB-Ethernet | the actual control channel — and the scriptable one myriadforge needs |

HDMI-CEC (the control-ish channel inside the HDMI spec) targets TV/AV sinks;
the Compute Stick doesn't implement it usefully. Not a path.

**Exception to note:** a few 2014–2019 Alienware laptops had true HDMI-*in*
ports. Standard laptops — including this project's i7-12650H host — do not.

---

## Architecture options

### A. Headless network-only (recommended default, $0)

Stick runs headless forever; SSH is the control plane.

- Windows 10 Home ships without RDP (Pro only) — use **OpenSSH server**
  (Windows optional feature) or a VNC server; Linux: everything works
- USB-Ethernet adapter recommended for benchmarking (Wi-Fi ac works, but
  latency variance pollutes percentile metrics)
- One-time cost: **initial Windows setup needs a display once** — any TV or
  monitor for the first boot, then headless forever. (Linux images can be
  provisioned fully headless.)
- myriadforge angle: this makes the stick a **networked benchmark node** — the
  x86 analog of the adb phone lane.

### B. Network control + HDMI capture display plane (+~$20)

Stick HDMI-out → USB3 UVC capture dongle → host. Adds:

- BIOS/UEFI sessions and boot verification without hauling out a monitor
- **Screen-state telemetry**: the host reads the stick's screen as frames and
  can gate on content — error dialogs, progress, hang detection. Research
  extension: feed captured frames to host-side **Mirage** (22 MB VLM, runs on
  CPU) as a "visual watchdog" — screen-content-triggered automation with zero
  agent software on the stick. Non-standard but real (vision-based QA
  automation pattern), and it composes with the NCS2 cascade work.
- Capture is also the fallback if network is broken — you can always *see*
  what the stick is doing.

### C. Zero-hardware display: Miracast (+$0)

Windows "Projecting to this PC": the laptop acts as a wireless display sink;
the stick projects to it over Wi-Fi Direct. Display-only (no input), higher
latency and worse quality than capture, but no dongle. Fine for casual
monitoring; not a benchmark-grade channel.

### D. Exotic: input without a control network (BLE HID proxy)

If the control network is down but capture is up, a BLE HID proxy (an Android
phone or ESP32 acting as a Bluetooth keyboard, driven over its own network
link) injects input into the stick. Niche rescue path; listed for
completeness. Not planned.

**Recommendation: A always, B as an option if screen-telemetry/BIOS access
matters.** Total hardware cost: USB-Ethernet ~$12, capture ~$18 (optional).

---

## myriadforge integration: remote node lane

The stick joins the device matrix as a **remote x86 host** (CPU lane, SSH
transport) — the orchestration analog of the Android/adb lane:

| Item | Path | Acceptance |
|---|---|---|
| SSH node protocol | `src/myriadforge/devices/remote_ssh.py` | exec myriadforge on the stick, stream the result JSON back; mirrors the adb lane's contract |
| Node probe | — | `myriadforge devices --remote stick` reports CPU/thermals/disk/uptime |
| Result fidelity | — | results carry the stick's device fingerprint (valid cross-machine comparison per result schema); `results/` merge is indistinguishable from local runs |
| Throttle logging | — | sample `CurrentClockSpeed`/thermal zone during bench; m3-6Y30 (6 W, active fan) will throttle — p50/p95 spread becomes the finding |
| OS choice | — | Windows 10 (license embedded in BIOS) or Linux; Linux frees ~2 GB of the 4 GB RAM for LiteRT CPU benchmarks |

Benchmark value: the m3-6Y30 (Skylake, LPDDR3-1866, ~29.8 GB/s) is a
**2016-era CPU-lane anchor** for the device matrix — the low end between
phones and the Core Ultra NPU target.

---

## What can be built before the stick arrives

The SSH-node orchestrator is hardware-independent — same trick as
`MYRIAD.SIM`:

- **`REMOTE.SIM`**: point the SSH node protocol at any disposable SSH target
  (WSL2 VM, spare box, a $4 VPS) and develop the full exec→result→merge path
  today; swap `REMOTE.SIM`→`REMOTE` when the stick lands
- Provisioning runbook doc (headless setup, OpenSSH enable, power/thermal
  settings, LiteRT install on py3.x, USB-Ethernet bring-up)
- Throttle-logging harness (runs on any Windows/Linux node)

Blocked on hardware: real first-boot provisioning, m3 benchmark numbers,
thermal behavior, capture-dongle verification.

---

## Sequencing

```
REMOTE.SIM protocol (any SSH box) ──▶ stick arrives ──▶ provision ──▶ node goes live
Screen-telemetry/Mirage watchdog (optional research) ◀── needs capture dongle only
```

Effort estimate: 2–3 sessions for the protocol + runbook; the stick side is
one provisioning session once hardware is in hand.
