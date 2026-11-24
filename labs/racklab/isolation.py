"""Small causal spectral subtraction baseline for 16 kHz mono audio."""

import numpy as np


class SpectralIsolation:
    """Overlap-add denoising with a bounded running estimate of stationary noise."""

    rate = 16_000
    hop = 256
    lookahead_samples = hop

    def __init__(self):
        size = 2 * self.hop
        self.window = np.sqrt(np.hanning(size + 1)[:-1])
        self.pending = np.zeros(self.hop, dtype=np.float32)
        self.overlap = np.zeros(self.hop)
        self.noise = None
        self.received = self.emitted = 0
        self.first = True
        self.closed = False

    def _drain(self):
        size = 2 * self.hop
        output = []
        offset = 0
        while offset + size <= len(self.pending):
            windowed = self.pending[offset:offset + size] * self.window
            spectrum = np.fft.rfft(windowed)
            power = np.abs(spectrum) ** 2
            if self.noise is None:
                self.noise = power.copy()
            gain = np.sqrt(np.maximum(0.18 ** 2, 1.0 - 1.25 * self.noise / np.maximum(power, 1e-12)))
            # Slow upward updates on loud arrivals limit absorption of new speech into the floor.
            rate = np.where(power <= 2.5 * self.noise, 0.10, 0.004)
            self.noise += rate * (power - self.noise)
            signal = np.fft.irfft(spectrum * gain, n=size) * self.window
            block = self.overlap + signal[:self.hop]
            self.overlap = signal[self.hop:]
            if self.first:
                self.first = False
            else:
                output.append(block)
            offset += self.hop
        self.pending = self.pending[offset:].copy()
        return np.asarray(output, dtype=np.float32).reshape(-1)

    def push(self, samples):
        if self.closed:
            raise ValueError("Start a new isolation session after flushing")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("Isolation expects finite mono samples")
        self.received += len(samples)
        self.pending = np.concatenate((self.pending, samples))
        output = self._drain()
        self.emitted += len(output)
        return output

    def flush(self):
        if self.closed:
            return np.empty(0, dtype=np.float32)
        self.closed = True
        self.pending = np.pad(self.pending, (0, 2 * self.hop))
        return self._drain()[:self.received - self.emitted]


def main():
    import argparse
    from pathlib import Path
    import wave

    parser = argparse.ArgumentParser(description="Experimental isolation of mono 16 kHz PCM16 WAV audio")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    session = SpectralIsolation()
    with wave.open(str(args.input), "rb") as original:
        if (original.getframerate(), original.getnchannels(), original.getsampwidth()) != (session.rate, 1, 2):
            parser.error("Input must be a mono 16 kHz PCM16 WAV")
        with args.output.open("xb") as stream, wave.open(stream, "wb") as output:
            output.setparams(original.getparams())

            def write(samples):
                pcm = np.rint(samples * 32768).clip(-32768, 32767).astype("<i2")
                output.writeframesraw(pcm.tobytes())

            for block in iter(lambda: original.readframes(320), b""):
                write(session.push(np.frombuffer(block, dtype="<i2").astype(np.float32) / 32768))
            write(session.flush())


if __name__ == "__main__":
    main()
