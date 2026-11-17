import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from inference import EmotionRecognizer
from model import bundle_signature, summarize_probabilities

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ensemble(tmp_path):
    old = ROOT / "models/production/branch_0"
    for index in range(2):
        folder = tmp_path / f"branch_{index}"; folder.mkdir()
        config = json.loads((old / "model_config.json").read_text())
        for filename in [config["scaler"], *[m["checkpoint"] for m in config["members"]]]:
            shutil.copy2(old / filename, folder / filename)
        config["temperature"] *= index+1
        (folder / "model_config.json").write_text(json.dumps(config))
    config = {"architecture":"audio_ensemble_v1","schema_version":4,"input_dim":4287,"num_classes":6,"display_name":"test ensemble","branches":[{"config":"branch_0/model_config.json","weight":.25},{"config":"branch_1/model_config.json","weight":.75}]}
    path = tmp_path / "model_config.json"; path.write_text(json.dumps(config))
    return path, EmotionRecognizer(config_path=path,feature_extractor=object())


def test_ensemble_requires_each_branch_features_and_hashes_children(ensemble):
    path,recognizer = ensemble
    features = [np.zeros(4287,dtype=np.float32),np.ones(4287,dtype=np.float32)]
    expected = sum(w*child.predict_features(x) for (w,child),x in zip(recognizer.branches,features))
    np.testing.assert_allclose(recognizer.predict_features(features),expected,atol=1e-7)
    with pytest.raises(ValueError,match="按分支"): recognizer.predict_features(features[0])
    original = bundle_signature(path)
    child = path.parent / "branch_1/model_config.json"; config=json.loads(child.read_text());config["temperature"]+=.1;child.write_text(json.dumps(config))
    assert bundle_signature(path) != original


def test_ensemble_blends_audio_and_timeline_probabilities(ensemble,monkeypatch):
    _,recognizer = ensemble
    values = [np.array([.7,.1,.05,.05,.05,.05]),np.array([.1,.7,.05,.05,.05,.05])]
    for (_,child),probabilities in zip(recognizer.branches,values):
        result = {**summarize_probabilities(probabilities),"audio":{"duration":2},"segments":[{"start":0,"end":2,**summarize_probabilities(probabilities)}]}
        monkeypatch.setattr(child,"predict",lambda _,_shared=None,result=result:result)
    result = recognizer.predict("unused.wav")
    expected = .25*values[0]+.75*values[1]
    np.testing.assert_allclose(list(result["all_probabilities"].values()),expected)
    np.testing.assert_allclose(list(result["segments"][0]["all_probabilities"].values()),expected)
    assert result["predicted_id"] == 1
