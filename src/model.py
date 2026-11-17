"""Deployed fusion classifier, probability summaries and artifact signatures."""
import numpy as np
import hashlib
import json
from pathlib import Path
import torch
from torch import nn

CLASS_NAMES = ["neutral", "happy", "sad", "angry", "fear", "surprise"]
EXTRACTOR = "iic/emotion2vec_plus_large"


def default_config_path():
    root = Path(__file__).resolve().parents[1]
    return root / "models/production/model_config.json"


def bundle_signature(config_path):
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("architecture") == "audio_ensemble_v1":
        digest = hashlib.sha256(path.read_bytes())
        for branch in config["branches"]:
            child = (path.parent / branch["config"]).resolve()
            if not child.is_relative_to(path.parent.resolve()): raise ValueError("Invalid branch configuration path")
            digest.update(bundle_signature(child).encode())
        return digest.hexdigest()
    files = [path, path.parent / config.get("scaler", "scaler.pkl")]
    files.extend(path.parent / member.get("checkpoint", "best_model.pth") for member in config.get("members", [config]))
    if config.get("backbone_adapter"):
        files.append(path.parent / config["backbone_adapter"])
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode()); digest.update(file.read_bytes())
    return digest.hexdigest()


class FusionClassifier(nn.Module):
    """Separate temporal embedding statistics from lower-dimensional prosody."""
    def __init__(self, input_dim=2151, hidden_dim=256, num_classes=6, dropout=.3):
        super().__init__()
        if input_dim <= 2048:
            raise ValueError("融合分类器需要帧级统计和声学特征")
        self.embedding = nn.Sequential(nn.Linear(2048, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.prosody = nn.Sequential(nn.Linear(input_dim - 2048, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden_dim + 64, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x):
        return self.head(torch.cat([self.embedding(x[:, :2048]), self.prosody(x[:, 2048:])], dim=1))


def build_classifier(config, member):
    architecture = config.get("architecture")
    if architecture != "fusion_v1":
        raise ValueError(f"未知分类器架构: {architecture}")
    return FusionClassifier(config["input_dim"], member["hidden_dim"], config["num_classes"], member["dropout"])


def pool_features(features, input_dim=None, single=False):
    """(N,T,D)->(N,D); single utterance (D,)/(T,D)->(1,D)."""
    x = np.asarray(features, dtype=np.float32)
    if single:
        if x.ndim == 1:
            x = x[None, :]
        elif x.ndim == 2:
            x = x.mean(axis=0, keepdims=True)
        elif x.ndim == 3 and x.shape[0] == 1:
            x = x.mean(axis=1)
        else:
            raise ValueError(f"无法解析单条音频特征维度: {x.shape}")
    elif x.ndim == 3:
        x = x.mean(axis=1)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("特征必须为非空、有限数值的二维数组")
    if input_dim is not None and x.shape[1] != input_dim:
        raise ValueError(f"特征模型不匹配: 需要 {input_dim} 维，实际 {x.shape[1]} 维")
    return x


def summarize_probabilities(probabilities, class_names=CLASS_NAMES):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(class_names),) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("无效的情绪概率分布")
    p = p / p.sum()
    order = np.argsort(-p)
    confidence = float(p[order[0]])
    margin = float(p[order[0]] - p[order[1]])
    uncertain = confidence < .55 or margin < .15
    return {
        "predicted_emotion": class_names[order[0]], "predicted_id": int(order[0]),
        "confidence": confidence, "all_probabilities": dict(zip(class_names, p.tolist())),
        "uncertain": uncertain, "margin": margin,
        "candidates": [{"emotion": class_names[i], "probability": float(p[i])} for i in order[:3]],
        "interpretation": "情绪倾向不明确，建议结合语境判断或重新录制。" if uncertain else "结果表示声音中的情绪倾向，请结合语境理解。",
    }
