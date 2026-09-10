"""Audio Forensic De-Obfuscation & Voice Restoration System.

The tool creates auditable pitch-shift candidates from PCM WAV evidence. It
does not identify a speaker and cannot prove the original voice of an unknown
recording; every result is therefore labelled as an analytical candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


APP_NAME = "Vocalis Forensic Restorer"
VERSION = "1.0.0"
DEFAULT_FACTORS = (0.78, 0.84, 0.90, 0.96, 1.00, 1.04, 1.10, 1.18, 1.28)


class AudioToolError(Exception):
    """Expected, user-facing processing error."""


@dataclass
class AudioAnalysis:
    filename: str
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    duration_seconds: float
    estimated_f0_hz: float
    hnr_db: float
    speech_ratio: float
    manipulation: str
    recommended_correction_factor: float


@dataclass
class CandidateScore:
    filename: str
    factor: float
    f0_hz: float
    hnr_db: float
    speech_ratio: float
    fiqi: float
    recommendation: str = ""


def _read_wav(path: Path) -> tuple[np.ndarray, int, int, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        raise AudioToolError(f"تعذر قراءة ملف WAV: {exc}") from exc

    if sample_width != 2:
        raise AudioToolError("يجب أن يكون الملف WAV بصيغة PCM غير مضغوطة 16-bit.")
    if channels not in (1, 2):
        raise AudioToolError("الأداة تدعم التسجيل الأحادي أو الاستيريو فقط.")
    if not sample_rate or not raw:
        raise AudioToolError("ملف الصوت فارغ أو يحتوي على معدل عينات غير صالح.")

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate, channels, frame_count


def _frame_signal(signal: np.ndarray, size: int, hop: int) -> Iterable[np.ndarray]:
    if signal.size < size:
        padded = np.pad(signal, (0, size - signal.size))
        yield padded
        return
    for start in range(0, signal.size - size + 1, hop):
        yield signal[start : start + size]


def _estimate_f0(signal: np.ndarray, sample_rate: int) -> float:
    frame_size = min(2048, max(512, signal.size))
    hop = max(128, frame_size // 2)
    estimates: list[float] = []
    minimum_lag = max(2, int(sample_rate / 320.0))
    maximum_lag = min(frame_size - 1, int(sample_rate / 65.0))
    if maximum_lag <= minimum_lag:
        return 0.0

    window = np.hanning(frame_size)
    for frame in _frame_signal(signal, frame_size, hop):
        centered = (frame - np.mean(frame)) * window
        energy = float(np.dot(centered, centered))
        if energy < 1e-5:
            continue
        correlation = np.correlate(centered, centered, mode="full")[frame_size - 1 :]
        correlation[:minimum_lag] = 0.0
        lag = int(np.argmax(correlation[: maximum_lag + 1]))
        if lag and correlation[lag] > energy * 0.18:
            estimates.append(sample_rate / lag)
    if not estimates:
        return 0.0
    return float(np.median(estimates))


def _estimate_hnr(signal: np.ndarray, sample_rate: int, f0: float) -> float:
    if f0 <= 0.0:
        return 0.0
    frame_size = min(4096, max(1024, signal.size))
    frame = next(iter(_frame_signal(signal, frame_size, frame_size)), signal)
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame.size))) ** 2
    frequencies = np.fft.rfftfreq(frame.size, 1.0 / sample_rate)
    harmonic_mask = np.zeros_like(spectrum, dtype=bool)
    for harmonic in range(1, 8):
        distance = np.abs(frequencies - f0 * harmonic)
        harmonic_mask |= distance <= max(18.0, f0 * 0.06)
    harmonic_energy = float(np.sum(spectrum[harmonic_mask]))
    noise_energy = float(np.sum(spectrum[~harmonic_mask]))
    if noise_energy <= 1e-12:
        return 60.0
    return float(10.0 * math.log10(max(harmonic_energy, 1e-12) / noise_energy))


def _speech_ratio(signal: np.ndarray) -> float:
    frame_size = 1024
    hop = 512
    energies = []
    for frame in _frame_signal(signal, frame_size, hop):
        energies.append(float(np.sqrt(np.mean(frame * frame))))
    if not energies:
        return 0.0
    values = np.asarray(energies)
    threshold = max(0.008, float(np.percentile(values, 25)) * 1.8)
    return float(np.mean(values > threshold))


def analyze(signal: np.ndarray, sample_rate: int, path: Path, channels: int, frames: int) -> AudioAnalysis:
    f0 = _estimate_f0(signal, sample_rate)
    hnr = _estimate_hnr(signal, sample_rate, f0)
    speech = _speech_ratio(signal)
    if f0 and f0 < 110:
        manipulation = "احتمال تضخيم/تعميق اصطناعي"
        correction = 1.25
    elif f0 and f0 > 220:
        manipulation = "احتمال ترقيق/رفع اصطناعي"
        correction = 0.80
    else:
        manipulation = "لا توجد علامة طبقة واضحة على تعديل شديد"
        correction = 1.00
    return AudioAnalysis(
        filename=path.name,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=2,
        frames=frames,
        duration_seconds=frames / sample_rate,
        estimated_f0_hz=round(f0, 2),
        hnr_db=round(hnr, 2),
        speech_ratio=round(speech, 4),
        manipulation=manipulation,
        recommended_correction_factor=correction,
    )


def _pitch_candidate(signal: np.ndarray, factor: float) -> np.ndarray:
    if signal.size < 2 or factor == 1.0:
        return signal.copy()
    pitch_length = max(2, int(round(signal.size / factor)))
    source_positions = np.linspace(0.0, signal.size - 1.0, pitch_length)
    pitched = np.interp(source_positions, np.arange(signal.size), signal).astype(np.float32)
    return _time_stretch(pitched, signal.size)


def _time_stretch(signal: np.ndarray, target_length: int) -> np.ndarray:
    if signal.size < 2 or signal.size == target_length:
        return signal.copy()
    frame_size = min(1024, signal.size)
    synthesis_hop = max(128, frame_size // 4)
    analysis_hop = max(1, int(round(synthesis_hop * signal.size / target_length)))
    window = np.hanning(frame_size)
    output = np.zeros(target_length + frame_size, dtype=np.float32)
    weights = np.zeros(target_length + frame_size, dtype=np.float32)
    frame_count = max(1, int(math.ceil((signal.size - frame_size) / analysis_hop)) + 1)
    for frame_index in range(frame_count):
        input_start = frame_index * analysis_hop
        output_start = frame_index * synthesis_hop
        frame = signal[input_start : input_start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        output_end = min(output_start + frame_size, output.size)
        usable = output_end - output_start
        if usable <= 0:
            break
        output[output_start:output_end] += frame[:usable] * window[:usable]
        weights[output_start:output_end] += window[:usable] ** 2
    result = output[:target_length] / np.maximum(weights[:target_length], 1e-6)
    return result.astype(np.float32)


def _write_wav(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(signal, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def _candidate_score(signal: np.ndarray, sample_rate: int) -> tuple[float, float, float, float]:
    f0 = _estimate_f0(signal, sample_rate)
    hnr = _estimate_hnr(signal, sample_rate, f0)
    speech = _speech_ratio(signal)
    f0_score = 1.0 if 110.0 <= f0 <= 220.0 else max(0.0, 1.0 - abs(f0 - 165.0) / 165.0)
    hnr_score = min(1.0, max(0.0, (hnr + 5.0) / 35.0))
    fiqi = round(100.0 * (0.55 * f0_score + 0.30 * hnr_score + 0.15 * speech), 2)
    return f0, hnr, speech, fiqi


def process(input_path: Path, output_root: Path, factors: tuple[float, ...] = DEFAULT_FACTORS) -> Path:
    signal, sample_rate, channels, frames = _read_wav(input_path)
    analysis = analyze(signal, sample_rate, input_path, channels, frames)
    case_dir = output_root / input_path.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[CandidateScore] = []
    for index, factor in enumerate(factors, start=1):
        restored = _pitch_candidate(signal, factor)
        filename = f"{index:02d}_VOICE_{factor:.2f}x.wav"
        _write_wav(case_dir / filename, restored, sample_rate)
        f0, hnr, speech, fiqi = _candidate_score(restored, sample_rate)
        candidates.append(CandidateScore(filename, factor, round(f0, 2), round(hnr, 2), round(speech, 4), fiqi))

    best = max(candidates, key=lambda item: item.fiqi)
    best_path = case_dir / "00_BEST_NATURAL_VOICE_RECOMMENDED.wav"
    (case_dir / best.filename).replace(best_path)
    best.recommendation = "الخيار المرشح آلياً وفق FIQI"
    report = {
        "tool": APP_NAME,
        "version": VERSION,
        "notice": "النتائج مرشحات تحليلية وليست إثباتاً لهوية المتحدث أو استعادة مؤكدة للصوت الأصلي.",
        "input": asdict(analysis),
        "candidates": [asdict(candidate) for candidate in candidates],
        "recommended_file": best_path.name,
    }
    (case_dir / "forensic_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vocalis", description="فحص واستعادة صوتية جنائية لملفات WAV 16-bit")
    parser.add_argument("input", nargs="?", type=Path, help="مسار ملف WAV المشبوه")
    parser.add_argument("-o", "--output", type=Path, default=Path("vocalis_results"), help="مجلد النتائج")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {VERSION}")
    parser.add_argument("--config", type=Path, help="ملف JSON اختياري يحوي factors")
    parser.add_argument("--log-level", choices=("quiet", "normal", "verbose"), default="normal")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input is None:
        parser.error("يجب إدخال مسار ملف WAV")
    if not args.input.is_file():
        print("خطأ: ملف الإدخال غير موجود.", file=sys.stderr)
        return 2
    factors = DEFAULT_FACTORS
    if args.config:
        try:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            factors = tuple(float(value) for value in config["factors"])
            if not factors or any(value <= 0 for value in factors):
                raise ValueError
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            print("خطأ: ملف الإعدادات يجب أن يحتوي JSON صالحاً ومفتاح factors موجباً.", file=sys.stderr)
            return 2
    try:
        result = process(args.input, args.output, factors)
    except AudioToolError as exc:
        print(f"خطأ: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"خطأ: تعذر إنشاء النتائج: {exc}", file=sys.stderr)
        return 2
    if args.log_level != "quiet":
        print(f"اكتملت المعالجة. النتائج: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
