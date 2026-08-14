"""
Voice State Extraction — Layer 1 of the voice membrane architecture.

Extracts prosody features from raw audio BEFORE STT. These features are
biometric-equivalent signals (costly-to-fake because coupled to human
physiology: breath, prosody, pitch). Under arifOS they route to WELL as
sensor input for stress_load / homeostasis assessment — NOT to the LLM,
NOT into the transcript, and NEVER as claimed empathy (F9 ANTIHANTU).

Constitutional gates:
  F9  — Sensor measures. Sensor does NOT speak about what it measures.
        Output to human is FACT only (measured counts), never claimed feeling.
  F1  — VoiceState = biometric-equivalent. Access restricted.
  F7  — All features carry uncertainty. Confidence cap 0.90.
  W0  — WELL reflects, never gates. This module is a sensor, not a veto.

Wiring point:
  gateway/run.py _enrich_message_with_transcription() — after successful
  transcribe_audio, before transcript text is assembled. Audio path is
  already available at that point.

Usage:
  from tools.voice_state import extract_voice_state
  features = extract_voice_state("/path/to/audio.ogg")
  # features["speech_rate"], features["pause_density"], etc.
"""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import — librosa/numpy only loaded when extraction actually runs.
# Prevents import failures on headless installs without voice-state extra.
# ---------------------------------------------------------------------------

def _ensure_librosa():
    """Import librosa and numpy. Raises ImportError if not installed."""
    import librosa
    import numpy as np
    return librosa, np


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_voice_state(
    audio_path: str,
    *,
    sr: int = 22050,
    duration_cap: float = 120.0,
) -> Dict:
    """
    Extract prosody features from an audio file.

    Returns a dict of measured features suitable for WELL homeostasis input.
    All values are floats. On any failure, returns an empty dict (never raises).

    Features:
      speech_rate      — proxy syllables/sec via voiced-segment density
      pause_density    — fraction of audio that is silence (>200ms gaps)
      pitch_mean       — mean fundamental frequency (Hz), NaN if unvoiced
      pitch_std        — std of fundamental frequency
      energy_rms       — mean RMS energy (dB-relative)
      spectral_centroid_mean — mean spectral brightness (Hz)

    Duration cap: files longer than *duration_cap* seconds are trimmed from
    the start to keep extraction fast and avoid blocking the gateway loop.
    """
    try:
        librosa, np = _ensure_librosa()
    except ImportError:
        logger.debug("voice_state: librosa not available, skipping extraction")
        return {}

    try:
        y, sr_actual = librosa.load(
            audio_path,
            sr=sr,
            mono=True,
            duration=duration_cap,
        )
    except Exception as exc:
        logger.warning("voice_state: failed to load %s: %s", audio_path, exc)
        return {}

    if len(y) == 0:
        return {}

    features: Dict = {}

    # --- Duration ---
    duration = len(y) / sr_actual
    features["duration_s"] = round(duration, 3)

    # --- Energy / RMS ---
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # Silence threshold: 3dB below median RMS
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    median_db = np.median(rms_db)
    silence_thresh_db = median_db - 3.0
    silence_frames = rms_db < silence_thresh_db
    features["energy_rms_mean_db"] = round(float(np.mean(rms_db)), 3)

    # --- Pause density ---
    # Convert frame-level silence to time, then count gaps > 200ms
    frame_dur = hop_length / sr_actual
    silence_mask = silence_frames.astype(float)
    # Find contiguous silence runs
    runs = np.diff(np.concatenate(([0], silence_mask, [0])))
    starts = np.where(runs == 1)[0]
    ends = np.where(runs == -1)[0]
    if len(starts) > 0 and len(ends) > 0:
        run_lengths = (ends - starts) * frame_dur
        long_pauses = run_lengths[run_lengths > 0.2]  # >200ms
        total_silence = np.sum(silence_mask) * frame_dur
        features["pause_density"] = round(float(total_silence / duration), 3)
        features["long_pause_count"] = int(len(long_pauses))
        features["long_pause_total_s"] = round(float(np.sum(long_pauses)), 3)
    else:
        features["pause_density"] = 0.0
        features["long_pause_count"] = 0
        features["long_pause_total_s"] = 0.0

    # --- Pitch (f0) ---
    # Use pyin — more robust than yin for noisy/reverberant speech.
    # Fall back to spectral centroid proxy if pyin fails (common on short clips).
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),  # ~65 Hz
            fmax=librosa.note_to_hz("C6"),  # ~1047 Hz
            sr=sr_actual,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        voiced_f0 = f0[~np.isnan(f0)]
        if len(voiced_f0) > 0:
            features["pitch_mean_hz"] = round(float(np.mean(voiced_f0)), 2)
            features["pitch_std_hz"] = round(float(np.std(voiced_f0)), 2)
            features["pitch_min_hz"] = round(float(np.min(voiced_f0)), 2)
            features["pitch_max_hz"] = round(float(np.max(voiced_f0)), 2)
            features["voiced_fraction"] = round(float(len(voiced_f0) / len(f0)), 3)
        else:
            features["pitch_mean_hz"] = None
            features["pitch_std_hz"] = None
            features["voiced_fraction"] = 0.0
    except Exception:
        # pyin can fail on very short or silence-heavy clips
        features["pitch_mean_hz"] = None
        features["pitch_std_hz"] = None
        features["voiced_fraction"] = 0.0

    # --- Speech rate proxy ---
    # Voiced fraction already represents how much of the audio is speech.
    # A real speech_rate (syllables/sec) needs onset detection which is
    # segfault-prone on this system. Use voiced_fraction directly as a
    # 0-1 speech-activity ratio instead of a misleading rate metric.
    voiced_fraction = features.get("voiced_fraction", 0.0)
    features["speech_activity_ratio"] = round(voiced_fraction, 3)

    # --- Spectral centroid (brightness) ---
    try:
        spec_cent = librosa.feature.spectral_centroid(
            y=y, sr=sr_actual,
            n_fft=frame_length, hop_length=hop_length,
        )[0]
        features["spectral_centroid_mean_hz"] = round(float(np.mean(spec_cent)), 2)
    except Exception:
        features["spectral_centroid_mean_hz"] = None

    # --- Metadata ---
    features["extraction_sr"] = sr_actual
    features["extraction_status"] = "ok"

    return features


# ---------------------------------------------------------------------------
# WELL routing helper
# ---------------------------------------------------------------------------

def voice_state_to_well_features(features: Dict) -> Dict:
    """
    Map extracted voice features to WELL homeostasis input fields.

    Returns a dict suitable for well_assess_homeostasis(mode="fatigue") call:
      stress_load      — 0.0-1.0 composite from pause_density + energy + pitch_std
      cognitive_clarity — inverted proxy from speech_rate (low rate = possible fatigue)
      emotional_state  — neutral (F9: never inferred from prosody alone)
    """
    if not features or features.get("extraction_status") != "ok":
        return {}

    pause_density = features.get("pause_density", 0.0)
    energy_db = features.get("energy_rms_mean_db", -60.0)
    pitch_std = features.get("pitch_std_hz", 0.0) or 0.0
    speech_activity = features.get("speech_activity_ratio", 0.5)
    voiced_frac = features.get("voiced_fraction", 0.0)

    # Stress load composite: high pauses + low energy + high pitch variability
    # Each component normalized to 0-1 range, then weighted average.
    # These thresholds are heuristic — F7 confidence cap applies.
    stress_pause = min(pause_density / 0.5, 1.0)  # 50% silence = max
    stress_energy = min(max((-energy_db + 60) / 40, 0.0), 1.0)  # -60dB=0, -20dB=1
    stress_pitch_var = min(pitch_std / 80.0, 1.0)  # 80Hz std = max

    stress_load = round(
        0.4 * stress_pause + 0.3 * stress_energy + 0.3 * stress_pitch_var,
        3,
    )

    # Cognitive clarity: speech activity ratio. Low voiced = possible fatigue.
    # Range: 0.0 (silence) to 1.0 (fully voiced).
    clarity = min(speech_activity, 1.0)

    result = {
        "stress_load": stress_load,
        "cognitive_clarity": round(clarity, 3),
        "emotional_state": "neutral",  # F9: NEVER inferred from prosody alone
        "hrv_status": "unknown",  # Not derivable from voice — sensor gap
        "chronic_fatigue": False,  # Requires temporal tracking, not single-clip
        "source": "voice_state_layer1",
        "confidence": 0.70,  # F7: prosody is inherently ambiguous
        "features_raw": {
            k: v for k, v in features.items()
            if k not in ("extraction_status", "extraction_sr")
        },
    }

    return result
