import numpy as np
import pytest
import soundfile as sf
from audio import load_audio,normalize_for_model,segments,AudioValidationError


def test_audio_decodes_stereo_resamples_and_reports_quality(tmp_path):
    t=np.arange(44100)/44100;wave=.2*np.sin(2*np.pi*220*t)
    path=tmp_path/'stereo.wav';sf.write(path,np.stack([wave,wave],axis=1),44100)
    actual,quality=load_audio(path)
    assert actual.shape==(16000,) and quality['channels']==2 and quality['sample_rate']==16000
    assert np.isfinite(normalize_for_model(actual)).all()


def test_invalid_silent_short_and_long_audio_are_rejected(tmp_path):
    for name,wave in [('silent',np.zeros(16000)),('short',np.ones(1000)),('long',np.ones(61*16000))]:
        path=tmp_path/(name+'.wav');sf.write(path,wave,16000)
        with pytest.raises(AudioValidationError):load_audio(path)


def test_segments_cover_audio_and_merge_short_tail():
    wave=np.ones(13*16000,dtype=np.float32);parts=list(segments(wave))
    assert [(start,end) for start,end,_ in parts]==[(0,96000),(96000,192000),(192000,208000)]
    assert sum(len(chunk) for _,_,chunk in parts)==len(wave)
    assert [(start,end) for start,end,_ in segments(np.ones(200000,dtype=np.float32))]==[(0,96000),(96000,200000)]
