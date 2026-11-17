"""Bounded decoding and normalization matching the CASIA training pipeline."""
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
MAX_DURATION = 60


class AudioValidationError(ValueError):
    pass


def load_audio(path):
    try:
        info = sf.info(path)
        if not info.frames or not 8000 <= info.samplerate <= 192000 or info.channels > 8:
            raise AudioValidationError("音频为空或采样率/声道不受支持，请转换为 WAV 后重试。")
        if info.duration > MAX_DURATION:
            raise AudioValidationError(f"音频最长支持 {MAX_DURATION} 秒，请截取后重试。")
        if info.duration < .5:
            raise AudioValidationError("音频过短，请至少录制 0.5 秒，建议 3–10 秒。")
        y, sr = sf.read(path, dtype="float32", always_2d=True)
    except AudioValidationError:
        raise
    except (RuntimeError, ValueError) as exc:
        raise AudioValidationError("无法解码音频，请使用有效的 WAV、MP3、FLAC 或 OGG 文件。") from exc
    if not np.isfinite(y).all():
        raise AudioValidationError("音频含无效采样值，请重新录制。")
    clipping = float(np.mean(np.abs(y) >= .999))
    y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        divisor = gcd(sr, SAMPLE_RATE)
        y = resample_poly(y, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms < .0001:
        raise AudioValidationError("没有检测到足够的有效声音，请检查麦克风或提高录音音量。")
    frame = SAMPLE_RATE // 50
    power = np.array([np.sqrt(np.mean(y[i:i+frame] ** 2)) for i in range(0, len(y), frame)])
    active_ratio = float(np.mean(power > max(.0005, float(power.max()) * .05)))
    if active_ratio < .08:
        raise AudioValidationError("有效声音过少，请减少静音并录制一段完整语音。")
    warnings = []
    if rms < .008:
        warnings.append("录音音量偏低，可能影响识别。")
    if clipping > .01:
        warnings.append("音频存在削波失真，建议降低麦克风增益。")
    if active_ratio < .4:
        warnings.append("静音占比较高，建议截取有效语音。")
    waveform = [round(float(np.max(np.abs(a))), 4) for a in np.array_split(y, min(180, len(y))) if len(a)]
    quality = {"duration": round(len(y) / SAMPLE_RATE, 3), "sample_rate": SAMPLE_RATE,
               "original_sample_rate": sr, "channels": info.channels, "rms_db": round(20 * np.log10(max(rms, 1e-10)), 1),
               "clipping_ratio": round(clipping, 4), "active_ratio": round(active_ratio, 3),
               "warnings": warnings, "waveform": waveform}
    return y, quality


def normalize_for_model(y):
    import librosa
    y = librosa.util.normalize(np.asarray(y, dtype=np.float32))
    trimmed, _ = librosa.effects.trim(y, top_db=20)
    if len(trimmed) < SAMPLE_RATE * .3:
        raise AudioValidationError("有效音频过短，请重新录制。")
    return np.ascontiguousarray(trimmed, dtype=np.float32)


def segments(y, window_seconds=6):
    if len(y) <= SAMPLE_RATE * 8:
        yield 0, len(y), y
        return
    window = window_seconds * SAMPLE_RATE
    start = 0
    while start < len(y):
        end = min(start + window, len(y))
        if len(y) - end < SAMPLE_RATE:
            end = len(y)
        yield start, end, y[start:end]
        start = end
