"""Audio front-end: WAV I/O, log-Mel spectrograms, acoustic descriptors.

The spectrogram "photo" is the input to the acoustic branch. The handcrafted
descriptors are gain-invariant summaries that make the fusion model more robust
to phone-microphone differences.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, resample_poly

from .constants import AUDIO


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------
def read_wav(path: str | Path, target_sr: int = AUDIO["sample_rate"]) -> np.ndarray:
    """Read a WAV file as mono float32 in [-1, 1] at ``target_sr``."""
    sr, y = wavfile.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    if np.issubdtype(y.dtype, np.integer):
        info = np.iinfo(y.dtype)
        y = y.astype(np.float32) / max(abs(info.min), info.max)
    else:
        y = y.astype(np.float32)
    y = np.nan_to_num(y)
    if sr != target_sr:
        gcd = int(np.gcd(int(sr), int(target_sr)))
        y = resample_poly(y, target_sr // gcd, sr // gcd).astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def write_wav(path: str | Path, y: np.ndarray, sr: int = AUDIO["sample_rate"]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(y, -1.0, 1.0)
    wavfile.write(str(path), sr, (y * 32767).astype(np.int16))


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------
def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int = AUDIO["sample_rate"],
    n_fft: int = AUDIO["n_fft"],
    n_mels: int = AUDIO["n_mels"],
    fmin: float = AUDIO["fmin"],
    fmax: float = AUDIO["fmax"],
) -> np.ndarray:
    """HTK-style triangular Mel filterbank, shape (n_mels, n_fft // 2 + 1)."""
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_bins)
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        left = (freqs - lo) / max(mid - lo, 1e-6)
        right = (hi - freqs) / max(hi - mid, 1e-6)
        fb[i] = np.maximum(0.0, np.minimum(left, right))
        # Area normalisation keeps broadband noise comparable across Mel bands.
        fb[i] /= max(fb[i].sum(), 1e-6)
    return fb


_FB_CACHE: dict[tuple[int, int, int, float, float], np.ndarray] = {}


def _get_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    key = (sr, n_fft, n_mels, fmin, fmax)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    return _FB_CACHE[key]


def log_mel(
    y: np.ndarray,
    sr: int = AUDIO["sample_rate"],
    n_fft: int = AUDIO["n_fft"],
    hop_length: int = AUDIO["hop_length"],
    n_mels: int = AUDIO["n_mels"],
    fmin: float = AUDIO["fmin"],
    fmax: float = AUDIO["fmax"],
    target_frames: int = AUDIO["target_frames"],
) -> np.ndarray:
    """Return log-Mel power in dB, shape (n_mels, target_frames)."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    window = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop_length]
    spec = np.fft.rfft(frames * window, axis=1) / (n_fft / 2.0)
    power = (np.abs(spec) ** 2).astype(np.float32)  # (frames, bins), approx dBFS
    fb = _get_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel = power @ fb.T  # (frames, mels)
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))  # relative dB
    db = db.T  # (mels, frames)
    if db.shape[1] != target_frames:
        db = cv2.resize(
            db,
            (target_frames, n_mels),
            interpolation=cv2.INTER_LINEAR if db.shape[1] > target_frames else cv2.INTER_CUBIC,
        )
    return db.astype(np.float32)


def normalize_spectrogram(
    db: np.ndarray,
    db_floor: float = AUDIO["db_floor"],
    db_ceil: float = AUDIO["db_ceil"],
) -> np.ndarray:
    """Convert dB to a fixed-range 2-channel tensor.

    Channel 0: absolute level (captures bruit loudness / absence).
    Channel 1: local spectral contrast (captures turbulent high-frequency shape).
    """
    base = (np.clip(db, db_floor, db_ceil) - db_floor) / (db_ceil - db_floor)
    base = (base * 2.0 - 1.0).astype(np.float32)
    smooth = cv2.GaussianBlur(db, (0, 0), sigmaX=2.5, sigmaY=2.5)
    contrast = np.clip((db - smooth) / 8.0, -1.0, 1.0).astype(np.float32)
    return np.stack([base, contrast], axis=0)


def spectrogram_tensor(db: np.ndarray) -> np.ndarray:
    return normalize_spectrogram(db)


def acoustic_features(db: np.ndarray, y: np.ndarray, sr: int = AUDIO["sample_rate"]) -> np.ndarray:
    """16 gain-invariant descriptors derived from the Mel spectrogram/waveform."""
    db = np.asarray(db, dtype=np.float32)
    power = 10.0 ** (db / 10.0)
    eps = 1e-10
    freqs = _mel_to_hz(
        np.linspace(_hz_to_mel(AUDIO["fmin"]), _hz_to_mel(AUDIO["fmax"]), db.shape[0])
    ).astype(np.float32)
    frame_energy = power.sum(axis=0) + eps
    log_e = np.log10(frame_energy)
    psum = power.sum(axis=0, keepdims=True) + eps
    centroid = (freqs[:, None] * power).sum(axis=0) / psum[0]
    bandwidth = np.sqrt((((freqs[:, None] - centroid[None, :]) ** 2) * power).sum(axis=0) / psum[0])
    cum = np.cumsum(power, axis=0)
    rolloff = freqs[np.argmax(cum >= 0.85 * cum[-1:], axis=0)]
    flatness = np.exp(np.mean(np.log(power + eps), axis=0)) / (np.mean(power, axis=0) + eps)

    high = power[freqs >= 1500.0].sum(axis=0) + eps
    low = power[freqs <= 500.0].sum(axis=0) + eps
    high_low = 10.0 * np.log10(high / low)

    # Pulse modulation: autocorrelation of the frame-energy envelope.
    env = log_e - np.mean(log_e)
    env = env / (np.std(env) + eps)
    ac = np.correlate(env, env, mode="full")[len(env) - 1 :]
    ac = ac / (ac[0] + eps)
    fps = sr / AUDIO["hop_length"]
    lo_lag = max(1, int(np.floor(fps / 2.2)))
    hi_lag = min(len(ac) - 1, int(np.ceil(fps / 0.55)))
    peak = ac[lo_lag : hi_lag + 1]
    if len(peak) == 0:
        mod_strength, bpm = 0.0, 0.0
    else:
        idx = int(np.argmax(peak))
        mod_strength = float(peak[idx])
        bpm = 60.0 * fps / (lo_lag + idx) if peak[idx] > 0 else 0.0

    flux = np.mean(np.abs(np.diff(db, axis=1))) if db.shape[1] > 1 else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(y))) > 0)) if len(y) > 1 else 0.0
    high_band_dyn = float(np.percentile(10.0 * np.log10(high), 95) - np.percentile(10.0 * np.log10(high), 10))
    low_high = 10.0 * np.log10(low / (power[freqs >= 1500.0].sum(axis=0) + eps))
    cent_p90 = float(np.percentile(centroid, 90))

    feats = np.array(
        [
            np.mean(log_e),
            np.std(log_e),
            np.percentile(log_e, 95) - np.percentile(log_e, 20),
            np.mean(centroid),
            np.std(centroid),
            np.mean(bandwidth),
            np.mean(rolloff),
            np.mean(flatness),
            np.mean(high_low),
            mod_strength,
            bpm,
            flux,
            zcr,
            high_band_dyn,
            np.mean(low_high),
            cent_p90,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Synthetic bruit (for the demo dataset only)
# ---------------------------------------------------------------------------
def _bandpass(y: np.ndarray, lo: float, hi: float, sr: int, order: int = 2) -> np.ndarray:
    lo = max(20.0, min(lo, sr / 2 - 100))
    hi = max(lo + 50.0, min(hi, sr / 2 - 50))
    b, a = butter(order, [lo / (sr / 2), hi / (sr / 2)], btype="band")
    return lfilter(b, a, y).astype(np.float32)


def _lowpass(y: np.ndarray, cutoff: float, sr: int, order: int = 2) -> np.ndarray:
    b, a = butter(order, min(cutoff, sr / 2 - 100) / (sr / 2), btype="low")
    return lfilter(b, a, y).astype(np.float32)


def synthesize_bruit(
    rng: np.random.Generator,
    sr: int = AUDIO["sample_rate"],
    duration: float = AUDIO["duration_s"],
    bruit_type: str = "laminar",
    severity: float = 0.6,
    heart_rate: float = 72.0,
    ambient_level: float = 0.035,
    gain_db: float = 0.0,
) -> np.ndarray:
    """Create a physiologically-inspired demo bruit.

    laminar   : low-pitched, low-pass 'whoosh' with cardiac harmonics.
    turbulent : broadband high-frequency 'hiss' from a stenosis.
    absent    : ambient clinic noise plus a faint cardiac thump.
    This is a simulation for pipeline development, NOT patient data.
    """
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr
    period = 60.0 / max(heart_rate, 40.0)
    phase = np.mod(t, period) / period
    envelope = np.exp(-((phase - 0.12) / 0.085) ** 2)
    envelope += 0.30 * np.exp(-((phase - 0.34) / 0.11) ** 2)
    envelope = envelope.astype(np.float32)

    ambient = rng.normal(0, 1, n).astype(np.float32)
    ambient = _lowpass(ambient, 1800.0, sr)
    ambient *= ambient_level
    hum_freq = rng.choice([50.0, 60.0])
    ambient += 0.010 * np.sin(2 * np.pi * hum_freq * t + rng.uniform(0, 6.28))
    ambient += 0.004 * np.sin(2 * np.pi * hum_freq * 2 * t + rng.uniform(0, 6.28))
    # Occasional corridor/clinic transients.
    for _ in range(rng.integers(0, 3)):
        start = rng.integers(0, max(1, n - sr // 2))
        length = rng.integers(sr // 40, sr // 8)
        click = rng.normal(0, 1, length) * np.exp(-np.linspace(0, 8, length))
        ambient[start : start + length] += 0.02 * click.astype(np.float32)

    carrier = np.zeros(n, dtype=np.float32)
    if bruit_type == "laminar":
        noise = _bandpass(rng.normal(0, 1, n), 60, 420 + 180 * severity, sr)
        carrier = 0.55 * noise + 0.30 * np.sin(2 * np.pi * (78 + 40 * severity) * t)
        carrier += 0.12 * np.sin(2 * np.pi * (156 + 60 * severity) * t)
        carrier *= envelope
        carrier *= 0.10 + 0.13 * severity
    elif bruit_type == "turbulent":
        noise = _bandpass(rng.normal(0, 1, n), 380, 4800, sr)
        hiss = _bandpass(rng.normal(0, 1, n), 900, 3600, sr)
        carrier = (0.45 * noise + 0.55 * hiss) * (envelope ** 1.4)
        carrier += 0.05 * np.sin(2 * np.pi * (1100 + 700 * severity) * t + 4 * np.sin(2 * np.pi * 3 * t))
        carrier *= envelope
        carrier *= 0.10 + 0.30 * severity
    elif bruit_type == "absent":
        carrier = 0.012 * np.sin(2 * np.pi * 62 * t) * envelope
        carrier += 0.006 * _lowpass(rng.normal(0, 1, n), 120, sr) * envelope
    else:
        raise ValueError(f"Unknown bruit_type: {bruit_type}")

    mic = _lowpass(rng.normal(0, 1, n), 90.0, sr) * 0.006 * envelope
    y = ambient + carrier + mic
    y *= 10.0 ** (gain_db / 20.0)
    peak = float(np.max(np.abs(y))) + 1e-8
    target_peak = rng.uniform(0.42, 0.86)
    y = y / peak * target_peak
    # A gentle fade avoids clicks at file boundaries.
    fade = int(0.05 * sr)
    y[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    y[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return np.clip(y, -0.99, 0.99).astype(np.float32)
