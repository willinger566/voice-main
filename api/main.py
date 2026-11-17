"""Same-origin web app and bounded, serialized inference API."""
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

import anyio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audio import AudioValidationError
from model import default_config_path

logger = logging.getLogger("voice")
MAX_UPLOAD = 15 * 1024 * 1024
EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
recognizer = None
load_state = "loading"
load_message = "正在加载语音模型，首次启动可能需要一些时间。"
inference_gate = asyncio.Semaphore(1)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_recognizer():
    global recognizer, load_state, load_message
    try:
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        from inference import EmotionRecognizer
        backbone = resolve_path(os.getenv("EMOTION2VEC_MODEL", "models/emotion2vec_plus_large"))
        if not (backbone / "model.pt").exists():
            load_state = "unavailable"
            load_message = "缺少语音模型，请在项目目录运行 python prepare_model.py，完成后重启服务。"
            return
        recognizer = EmotionRecognizer(
            emotion2vec_model=str(backbone),
            config_path=resolve_path(os.environ["CONFIG_PATH"]) if os.getenv("CONFIG_PATH") else default_config_path(),
            device=os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        )
        load_state, load_message = "ready", "模型已就绪"
    except Exception:
        logger.exception("模型加载失败")
        load_state, load_message = "error", "模型加载失败，请查看服务日志并检查依赖、权重与配置。"


@asynccontextmanager
async def lifespan(app):
    global load_state
    task = None
    if os.getenv("VOICE_SKIP_MODEL_LOAD") != "1":
        task = asyncio.create_task(anyio.to_thread.run_sync(load_recognizer))
    else:
        load_state = "unavailable"
    yield
    if task is not None:
        await task


app = FastAPI(title="声绪 · 中文语音情绪分析", version="3.0.0", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")


@app.middleware("http")
async def request_limits(request, call_next):
    from fastapi.responses import JSONResponse
    length = request.headers.get("content-length")
    if request.method == "POST" and request.url.path == "/predict" and (not length or request.headers.get("transfer-encoding")):
        return JSONResponse({"detail": "上传需提供 Content-Length，请使用网页或标准文件上传客户端。"}, status_code=411)
    if length:
        try:
            if int(length) > MAX_UPLOAD + 1024 * 1024:
                return JSONResponse({"detail": "文件过大，最大支持 15 MB。"}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "无效的请求长度。"}, status_code=400)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(ROOT / "web/index.html")


@app.get("/health")
async def health():
    return {"status": "healthy" if recognizer is not None else "degraded",
            "model_loaded": recognizer is not None, "state": load_state,
            "message": load_message, "busy": inference_gate.locked(), "version": "3.0.0"}


@app.get("/info")
async def info():
    from model import CLASS_NAMES, bundle_signature
    config = resolve_path(os.environ["CONFIG_PATH"]) if os.getenv("CONFIG_PATH") else default_config_path()
    signature = getattr(recognizer, "signature", None)
    if signature is None and config.exists():
        signature = bundle_signature(config)
    audio_report_path = ROOT / "results/optimized_audio_benchmark.json"
    audio_report = json.loads(audio_report_path.read_text(encoding="utf-8")) if audio_report_path.exists() else {}
    audio_accuracy = audio_report.get("accuracy") if audio_report.get("model_signature") == signature else None
    feature_accuracy = None
    releases = [json.loads(path.read_text(encoding="utf-8")) for path in [ROOT / "results/release_report.json"] if path.exists()]
    release = next((entry for entry in releases if entry.get("model_signature") == signature), {})
    release_matches = release.get("model_signature") == signature
    evaluation = release.get("audio_evaluation", {}) if release_matches else {}
    if evaluation:
        audio_accuracy = evaluation["CASIA"]["accuracy"]
    model_config = recognizer.config if recognizer is not None and hasattr(recognizer, "config") else json.loads(config.read_text()) if config.exists() else {}
    data_path = ROOT / "results/dataset_report.json"
    dataset = json.loads(data_path.read_text(encoding="utf-8")) if release_matches and data_path.exists() else {}
    return {"model_name": model_config.get("display_name", "emotion2vec+ large + calibrated classifier"), "class_names": CLASS_NAMES,
            "device": str(recognizer.device) if recognizer else None,
            "model_loaded": recognizer is not None,
            "limits": {"max_upload_mb": 15, "max_duration_seconds": 60, "min_duration_seconds": .5},
            "formats": sorted(EXTENSIONS), "test_accuracy": audio_accuracy,
            "embedding_test_accuracy": feature_accuracy, "model_signature": signature,
            "evaluations": evaluation, "release_id": model_config.get("release_id", "ser-v2"),
            "training_samples": sum(dataset.get("corpora", {}).get(name, {}).get("train", {}).get("samples", 0) for name in model_config.get("training_datasets", [])) or None,
            "data_scope_note": f"训练来源共整理 {dataset['samples']} 条原始录音：{'、'.join(dataset['corpora'])}；外部数据按说话人隔离。原 CASIA 测试仍为同说话人划分。" if dataset else "原 CASIA 为四名说话人录制的 1,200 条语音，训练与原测试集共享说话人。",
            "model_description": "融合 emotion2vec+ 的帧级语音表征、帧间变化程度与韵律声学特征进行六分类；长音频按有效时长汇总片段概率。" if model_config.get("feature_spec") else "使用 emotion2vec+ 提取句级语音特征，由经过标准化和校准的分类器识别六类情绪倾向。",
            "test_samples": evaluation.get("CASIA", {}).get("samples"),
            "evaluation_note": ("CASIA 为原 180 条固定回归测试；外部中文与英语测试按说话人隔离，各自报告成绩。" + ("CREMA-D 为听者一致率≥60%的五类筛选子集，中性占多数，宏F1按五类计算。" if "CREMA-D" in evaluation else "") + "这些录音库结果不代表真实麦克风场景的普遍准确率。") if release_matches else "当前端到端结果来自 CASIA 180 条音频、同说话人测试划分；不代表真实场景或新说话人准确率。更换权重后，需要重新评估。",
            "privacy": "上传音频仅用于本次推理，临时文件在处理完成后删除；网页历史仅存于当前会话内存。"}


def sample_catalog():
    from model import CLASS_NAMES
    folder = ROOT / "data/samples"
    result = []
    for emotion in CLASS_NAMES:
        candidates = sorted(folder.glob(f"*_{emotion}_*.wav"))
        if candidates:
            result.append({"id": emotion, "label": emotion, "path": candidates[0]})
    return result


@app.get("/samples")
async def samples():
    return [{"id": s["id"], "label": s["label"], "url": f"/samples/{s['id']}"} for s in sample_catalog()]


@app.get("/samples/{sample_id}")
async def sample(sample_id: str):
    entry = next((s for s in sample_catalog() if s["id"] == sample_id), None)
    if entry is None:
        raise HTTPException(404, "示例不存在")
    return FileResponse(entry["path"], media_type="audio/wav", filename=f"CASIA_{sample_id}.wav")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in EXTENSIONS:
        await file.close()
        raise HTTPException(415, "支持 WAV、MP3、FLAC 和 OGG；浏览器录音会自动转为 WAV。")
    if recognizer is None:
        await file.close()
        raise HTTPException(503, load_message)
    try:
        await asyncio.wait_for(inference_gate.acquire(), timeout=.1)
    except asyncio.TimeoutError:
        await file.close()
        raise HTTPException(429, "正在处理另一段音频，请稍后重试。")
    temporary = None
    try:
        total = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as target:
            temporary = target.name
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD:
                    raise HTTPException(413, "文件过大，最大支持 15 MB。")
                target.write(chunk)
        if not total:
            raise HTTPException(422, "音频文件为空。")
        result = await anyio.to_thread.run_sync(recognizer.predict, temporary)
        return {"success": True, "message": "分析完成", **result}
    except AudioValidationError as exc:
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("语音识别失败")
        raise HTTPException(500, "分析失败，请重试；若问题持续，请查看服务日志。") from exc
    finally:
        inference_gate.release()
        await file.close()
        if temporary:
            Path(temporary).unlink(missing_ok=True)


@app.post("/predict_stream", deprecated=True)
async def predict_stream():
    raise HTTPException(501, "暂不提供流式识别，请录制后提交 /predict。")
