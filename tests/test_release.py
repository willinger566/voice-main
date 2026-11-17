import json
from pathlib import Path, PureWindowsPath
import numpy as np
import pytest
from model import bundle_signature, default_config_path
from inference import EmotionRecognizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "models/production/model_config.json"
pytestmark = pytest.mark.skipif(not CONFIG.exists(), reason="Production release not staged")


def release_report():
    signature = bundle_signature(CONFIG)
    report = json.loads((ROOT / "results/release_report.json").read_text())
    assert report["model_signature"] == signature
    return report


def test_production_is_default_and_matches_measured_release():
    report = release_report()
    assert default_config_path() == CONFIG
    assert bundle_signature(CONFIG) == report["model_signature"]
    assert report["release_gate_passed"]
    for corpus in ["CASIA", "CSEMOTIONS"]:
        assert report["audio_evaluation"][corpus]["accuracy"] >= report["previous_model"][corpus]["accuracy"]


def test_production_features_produce_complete_probabilities():
    recognizer = EmotionRecognizer(feature_extractor=object())
    assert recognizer.config["input_dim"] >= 2151
    features = [np.zeros(child.config["input_dim"],dtype=np.float32) for _,child in recognizer.branches] if recognizer.branches else np.zeros(recognizer.config["input_dim"], dtype=np.float32)
    p = recognizer.predict_features(features)
    assert p.shape == (6,) and np.isfinite(p).all()
    assert p.sum() == pytest.approx(1)


def test_training_provenance_is_non_base_and_source_manifest_matches():
    report = release_report()
    runtime = report["runtime"]
    # Historical provenance must remain verifiable when the checkout is moved.
    assert PureWindowsPath(runtime["prefix"]).name == ".venv-ser"
    assert runtime["isolated_packages"] and runtime["cuda_available"]
    config = json.loads(CONFIG.read_text())
    dataset = json.loads((ROOT / "results/dataset_report.json").read_text(encoding="utf-8"))
    assert config["training_manifest_sha256"] == dataset["manifest_sha256"]
