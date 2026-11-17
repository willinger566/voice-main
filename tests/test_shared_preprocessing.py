from types import SimpleNamespace
import numpy as np
import torch
from inference import EmotionRecognizer


def test_two_backbones_share_only_request_local_preprocessing(monkeypatch):
    import inference
    import rich_features
    recognizer=EmotionRecognizer(feature_extractor=object())
    calls={key:0 for key in ['decode','normalize','acoustic','egemaps','generate']}
    def decoded(_):
        calls['decode']+=1
        return np.full(16000,.1,dtype=np.float32),{'duration':1}
    def normalized(wave):
        calls['normalize']+=1
        return wave
    def acoustic(_):
        calls['acoustic']+=1
        return np.zeros(103,dtype=np.float32)
    def egemaps(_):
        calls['egemaps']+=1
        return np.zeros(88,dtype=np.float32)
    monkeypatch.setattr(inference,'load_audio',decoded);monkeypatch.setattr(inference,'normalize_for_model',normalized)
    monkeypatch.setattr(rich_features,'acoustic_features',acoustic);monkeypatch.setattr(rich_features,'egemaps_features',egemaps)
    captured=[]
    class Extractor:
        def __init__(self,value):
            self.value=value
            self.model=SimpleNamespace(blocks=[torch.nn.Identity() for _ in range(8)],modality_encoders={'AUDIO':SimpleNamespace(modality_cfg=SimpleNamespace(num_extra_tokens=0))})
        def generate(self,*args,**kwargs):
            calls['generate']+=1
            frames=np.full((5,1024),self.value,dtype=np.float32)
            self.model.blocks[6](torch.from_numpy(frames)[None])
            return [{'feats':frames}]
    for index,(_,child) in enumerate(recognizer.branches):
        child.feature_extractor=Extractor(index+1)
        def prediction(features):
            captured.append(features.copy())
            return np.array([.5,.1,.1,.1,.1,.1])
        monkeypatch.setattr(child,'predict_features',prediction)
    recognizer.predict('first.wav')
    assert calls=={'decode':1,'normalize':1,'acoustic':1,'egemaps':1,'generate':2}
    assert [features[0] for features in captured]==[1,2]
    recognizer.predict('second.wav')
    assert calls=={'decode':2,'normalize':2,'acoustic':2,'egemaps':2,'generate':4}
