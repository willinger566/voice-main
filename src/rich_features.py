"""Frame statistics and acoustic descriptors for the deployed classifier."""
import numpy as np

EMBEDDING_DIM = 1024
ACOUSTIC_DIM = 103
FEATURE_DIM = 2 * EMBEDDING_DIM + ACOUSTIC_DIM
FEATURE_SPEC = {"name": "mean_std_acoustic_v1", "embedding_dim": EMBEDDING_DIM,
                "acoustic_dim": ACOUSTIC_DIM, "input_dim": FEATURE_DIM, "sample_rate": 16000}


def egemaps_features(y):
    import threading
    import opensmile
    if not hasattr(egemaps_features, "local"):
        egemaps_features.local = threading.local()
    local = egemaps_features.local
    if not hasattr(local, "extractor"):
        local.extractor = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                        feature_level=opensmile.FeatureLevel.Functionals, num_workers=1, loglevel=0)
    result = local.extractor.process_signal(np.asarray(y, dtype=np.float32), 16000).iloc[0].to_numpy(dtype=np.float32)
    if result.shape != (88,): raise ValueError("Unexpected eGeMAPS feature schema")
    return np.nan_to_num(result, nan=0., posinf=0., neginf=0.)


def acoustic_features(y, sr=16000):
    import librosa
    y = np.asarray(y, dtype=np.float32)
    spectrum = np.abs(librosa.stft(y, n_fft=1024, hop_length=160))
    mel = librosa.feature.melspectrogram(S=spectrum ** 2, sr=sr, n_mels=64)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=1.0), n_mfcc=20)
    delta = librosa.feature.delta(mfcc, width=5, mode="nearest")
    blocks = [mfcc.mean(1), mfcc.std(1), delta.mean(1), delta.std(1)]
    rms = librosa.feature.rms(S=spectrum, frame_length=1024, hop_length=160)[0]
    flatness = librosa.feature.spectral_flatness(S=spectrum)[0]
    pitch = librosa.yin(y, fmin=65, fmax=600, sr=sr, frame_length=1024, hop_length=160)
    n = min(len(pitch), len(rms), len(flatness))
    voiced = (rms[:n] > max(float(rms.max()) * .08, .001)) & (flatness[:n] < .2)
    log_pitch = np.log2(np.maximum(pitch[:n][voiced], 1))
    def summary(values):
        if not len(values): return np.zeros(6, dtype=np.float32)
        return np.array([values.mean(), values.std(), *np.percentile(values, [10, 50, 90]),
                         np.abs(np.diff(values)).mean() if len(values) > 1 else 0], dtype=np.float32)
    blocks.extend([summary(log_pitch), np.array([voiced.mean()]), summary(np.log(np.maximum(rms, 1e-6)))])
    spectral = [librosa.feature.spectral_centroid(S=spectrum, sr=sr)[0],
                librosa.feature.spectral_bandwidth(S=spectrum, sr=sr)[0],
                librosa.feature.spectral_rolloff(S=spectrum, sr=sr)[0], flatness,
                librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=160)[0]]
    blocks.append(np.array([[a.mean(), a.std()] for a in spectral]).reshape(-1))
    result = np.concatenate(blocks).astype(np.float32)
    if result.shape != (ACOUSTIC_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid acoustic feature shape/values: {result.shape}")
    return result


def compose_features(frame_features, normalized_audio, acoustic=None):
    frames = np.asarray(frame_features, dtype=np.float32)
    if frames.ndim == 3 and frames.shape[0] == 1: frames = frames[0]
    if frames.ndim != 2 or frames.shape[1] != EMBEDDING_DIM or not len(frames):
        raise ValueError(f"Expected frame-level (T,{EMBEDDING_DIM}), got {frames.shape}")
    acoustic = acoustic_features(normalized_audio) if acoustic is None else np.asarray(acoustic,dtype=np.float32)
    if acoustic.shape != (ACOUSTIC_DIM,) or not np.isfinite(acoustic).all(): raise ValueError("Invalid cached acoustic features")
    return np.concatenate([frames.mean(0), frames.std(0), acoustic]).astype(np.float32)


