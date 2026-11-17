"""End-to-end recognition with a versioned feature contract."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from model import CLASS_NAMES, EXTRACTOR, pool_features, summarize_probabilities, bundle_signature, build_classifier, default_config_path
from audio import SAMPLE_RATE, AudioValidationError, load_audio, normalize_for_model, segments

ROOT = Path(__file__).resolve().parents[1]


class EmotionRecognizer:
    def __init__(self, emotion2vec_model=None, config_path=None, device="cpu", feature_extractor=None):
        self.device = device
        config_path = Path(config_path or default_config_path())
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.signature = bundle_signature(config_path)
        self.class_names = self.config.get("class_names", CLASS_NAMES)
        if len(self.class_names) != self.config["num_classes"]:
            raise ValueError("类别配置与分类器不匹配")
        expected = self.config.get("emotion2vec_model", EXTRACTOR)
        if emotion2vec_model and not Path(emotion2vec_model).exists() and emotion2vec_model != expected:
            raise ValueError(f"分类器需要特征模型 {expected}")
        folder = config_path.parent
        self.branches = []
        if self.config.get("architecture") == "audio_ensemble_v1":
            self.models, self.scaler = [], None
            self.temperature = 1.0
            if hasattr(feature_extractor, "model"): raise ValueError("融合分支需要各自独立的语音骨干实例")
            branches = self.config.get("branches", [])
            if not 2 <= len(branches) <= 3: raise ValueError("融合分支数无效")
            for branch in branches:
                weight = float(branch["weight"])
                if not np.isfinite(weight) or weight <= 0: raise ValueError("无效的融合权重")
                child = EmotionRecognizer(emotion2vec_model=emotion2vec_model, config_path=folder / branch["config"], device=device, feature_extractor=feature_extractor)
                if child.class_names != self.class_names or child.branches: raise ValueError("融合分支协议不一致")
                self.branches.append((weight, child))
            return
        self.scaler = joblib.load(folder / self.config.get("scaler", "scaler.pkl"))
        if self.scaler.n_features_in_ != self.config["input_dim"]:
            raise ValueError("标准化参数维度不匹配")
        self.models = []
        for member in self.config.get("members", [self.config]):
            model = build_classifier(self.config, member)
            checkpoint = folder / member.get("checkpoint", "best_model.pth")
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            self.models.append(model.to(device).eval())
        self.temperature = float(self.config.get("temperature", 1))
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("无效的概率校准参数")
        if feature_extractor is None:
            backbone_location = emotion2vec_model or str(ROOT / "models/emotion2vec_plus_large")
            if self.config.get("backbone_sha256"):
                import hashlib
                checkpoint = Path(backbone_location) / "model.pt"
                if not checkpoint.is_file():
                    raise ValueError("此模型要求使用已下载并校验的本地 emotion2vec 骨干，请先运行 prepare_model.py。")
                with checkpoint.open("rb") as source:
                    actual_sha = hashlib.file_digest(source, "sha256").hexdigest()
                if actual_sha != self.config["backbone_sha256"]:
                    raise ValueError("语音骨干权重与训练版本不一致")
            from funasr import AutoModel
            feature_extractor = AutoModel(model=backbone_location, hub="ms", device=device, disable_update=True, disable_pbar=True, log_level="ERROR")
        self.feature_extractor = feature_extractor
        if self.config.get("backbone_adapter") and hasattr(feature_extractor, "model"):
            state = torch.load(folder / self.config["backbone_adapter"], map_location="cpu", weights_only=True)
            outcome = feature_extractor.model.load_state_dict(state, strict=False)
            if outcome.unexpected_keys or not state:
                raise ValueError("语音骨干微调参数不匹配")
            feature_extractor.model.eval()

    def predict_features(self, features):
        if self.branches:
            if not isinstance(features, (list, tuple)) or len(features) != len(self.branches):
                raise ValueError("需要按分支提供各自骨干提取的特征，不能把同一向量冒充微调后的特征")
            return np.average([child.predict_features(x) for (_,child),x in zip(self.branches,features)], axis=0, weights=[w for w,_ in self.branches])
        x = pool_features(features, self.config["input_dim"], single=True)
        x = torch.as_tensor(self.scaler.transform(x), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            logits = torch.stack([m(x) for m in self.models]).mean(0) / self.temperature
            return torch.softmax(logits, dim=1)[0].cpu().numpy()

    def predict(self, audio_path, _shared=None):
        started = time.perf_counter()
        if self.branches:
            # Share only deterministic waveform/acoustic preprocessing within
            # this request. Each backbone still generates its own embeddings.
            context = {}
            results = [child.predict(audio_path, _shared=context) for _,child in self.branches]
            weights = [w for w,_ in self.branches]
            def blend(items):
                return np.average([[item["all_probabilities"][name] for name in self.class_names] for item in items], axis=0, weights=weights)
            timeline = []
            boundaries = [[(s["start"],s["end"]) for s in r["segments"]] for r in results]
            if any(b != boundaries[0] for b in boundaries[1:]): raise RuntimeError("融合分支音频分段不一致")
            for index,(start,end) in enumerate(boundaries[0]):
                timeline.append({"start":start,"end":end, **summarize_probabilities(blend([r["segments"][index] for r in results]), self.class_names)})
            return {**summarize_probabilities(blend(results), self.class_names), "audio":results[0]["audio"], "segments":timeline, "processing_ms":round((time.perf_counter()-started)*1000), "model":self.config["display_name"], "schema_version":self.config["schema_version"]}
        context = {} if _shared is None else _shared
        if "audio" not in context: context["audio"] = load_audio(audio_path)
        y, quality = context["audio"]
        timeline, probabilities, weights = [], [], []
        for start, end, chunk in segments(y):
            if float(np.sqrt(np.mean(chunk ** 2))) < .0001:
                continue
            try:
                normalization_key = ("normalized",start,end)
                if normalization_key not in context: context[normalization_key] = normalize_for_model(chunk)
                normalized = context[normalization_key]
            except AudioValidationError:
                continue
            rich = self.config.get("feature_spec", {}).get("name") == "mean_std_acoustic_v1"
            captured = {}
            handle = None
            intermediate_layer = self.config.get("feature_spec", {}).get("intermediate_layer")
            if intermediate_layer is not None:
                backbone = self.feature_extractor.model
                def capture(module, args, kwargs):
                    extra = backbone.modality_encoders["AUDIO"].modality_cfg.num_extra_tokens
                    captured["frames"] = args[0][0, extra:].detach().cpu().numpy().copy()
                handle = backbone.blocks[intermediate_layer + 1].register_forward_pre_hook(capture, with_kwargs=True)
            try:
                outputs = self.feature_extractor.generate(normalized, fs=SAMPLE_RATE, granularity="frame" if rich else "utterance", extract_embedding=True)
            finally:
                if handle is not None: handle.remove()
            if not isinstance(outputs, list) or not outputs or outputs[0].get("feats") is None:
                raise RuntimeError("特征提取器没有返回有效特征")
            if rich:
                from rich_features import compose_features,acoustic_features
                acoustic_key = ("acoustic_v1",start,end)
                if acoustic_key not in context: context[acoustic_key] = acoustic_features(normalized)
                features = compose_features(outputs[0]["feats"], normalized,acoustic=context[acoustic_key])
                if self.config.get("feature_spec", {}).get("egemaps"):
                    from rich_features import egemaps_features
                    egemaps_key = ("egemaps_v02",start,end)
                    if egemaps_key not in context: context[egemaps_key] = egemaps_features(normalized)
                    features = np.concatenate([features, context[egemaps_key]])
                if intermediate_layer is not None:
                    frames = captured["frames"]
                    features = np.concatenate([features, frames.mean(0), frames.std(0)])
            else:
                features = outputs[0]["feats"]
            p = self.predict_features(features)
            timeline.append({"start": round(start / SAMPLE_RATE, 2), "end": round(end / SAMPLE_RATE, 2), **summarize_probabilities(p, self.class_names)})
            probabilities.append(p); weights.append(len(normalized))
        if not timeline:
            raise AudioValidationError("没有足够的有效声音供分析，请重新录制。")
        result = summarize_probabilities(np.average(probabilities, axis=0, weights=weights), self.class_names)
        return {**result, "audio": quality, "segments": timeline,
                "processing_ms": round((time.perf_counter() - started) * 1000),
                "model": self.config.get("display_name", "emotion2vec+ · calibrated classifier"), "schema_version": self.config.get("schema_version", 2)}

    def batch_predict(self, audio_paths):
        results = []
        for path in audio_paths:
            try:
                results.append({"audio_path": str(path), "success": True, **self.predict(path)})
            except Exception as exc:
                results.append({"audio_path": str(path), "success": False, "error": str(exc)})
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "batch"], default="single")
    parser.add_argument("--audio_path")
    parser.add_argument("--audio_dir")
    parser.add_argument("--output")
    parser.add_argument("--config_path", default=str(default_config_path()))
    parser.add_argument("--emotion2vec_model", default=str(ROOT / "models/emotion2vec_plus_large"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    recognizer = EmotionRecognizer(args.emotion2vec_model, args.config_path, args.device)
    if args.mode == "single":
        if not args.audio_path:
            parser.error("single 模式需要 --audio_path")
        result = recognizer.predict(args.audio_path)
    else:
        if not args.audio_dir:
            parser.error("batch 模式需要 --audio_dir")
        result = recognizer.batch_predict(sorted(Path(args.audio_dir).glob("*.wav")))
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
