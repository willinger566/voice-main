"use strict";
const $ = (id) => document.getElementById(id);
const EMOTIONS = {
  neutral: {name: "中性", color: "#87a995"}, happy: {name: "高兴", color: "#d5af64"},
  sad: {name: "悲伤", color: "#88a9bf"}, angry: {name: "愤怒", color: "#ca8b7b"},
  fear: {name: "恐惧", color: "#a699b9"}, surprise: {name: "惊讶", color: "#9ab781"},
};
let selectedFile = null, audioUrl = null, currentResult = null, modelReady = false;
let busy = false, recording = false, requestingMic = false, decoding = false;
let recorder = null, stream = null, audioContext = null, analyser = null;
let recordInterval = null, recordStarted = 0, animationId = null, toastTimer = null;
let history = [], waveformValues = [], selectionToken = 0;

function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 6500);
}

function updateControls() {
  const locked = busy || recording || requestingMic || decoding;
  $("analyzeBtn").disabled = !selectedFile || !modelReady || locked;
  $("analyzeText").textContent = busy ? "正在分析，请稍候…" : decoding ? "正在处理音频…" : "开始分析";
  ["dropZone", "audioFile", "removeFile", "uploadTab", "recordTab"].forEach(id => { $(id).disabled = locked; });
  $("recordBtn").disabled = busy || requestingMic || decoding;
  document.querySelectorAll(".sample-button").forEach(button => { button.disabled = locked; });
  $("exportBtn").disabled = !currentResult || busy;
}

function setView(name) {
  if (recording || requestingMic) { toast("请先完成麦克风录音，再切换工作空间。"); return; }
  ["workspace", "history", "about"].forEach(view => { $(view + "View").hidden = view !== name; });
  document.querySelectorAll(".nav-item").forEach(button => {
    const active = button.dataset.view === name;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
  });
  $("pageTitle").textContent = {workspace: "情绪分析", history: "会话记录", about: "模型与说明"}[name];
  if (name === "history") renderHistory();
  window.scrollTo({top: 0, behavior: "instant"});
}

async function checkHealth() {
  try {
    const response = await fetch("/health", {signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error("服务响应异常");
    const data = await response.json();
    modelReady = data.model_loaded === true;
    $("statusText").textContent = modelReady ? "模型已就绪" : data.state === "loading" ? "模型加载中" : "模型未就绪";
    $("serviceStatus").className = "service-status " + (modelReady ? "ready" : "error");
    $("serviceNotice").hidden = modelReady;
    $("serviceNotice").textContent = data.message || "模型尚未就绪，请检查服务日志。";
  } catch {
    modelReady = false;
    $("statusText").textContent = "服务未连接";
    $("serviceStatus").className = "service-status error";
    $("serviceNotice").hidden = false;
    $("serviceNotice").textContent = "无法连接分析服务，请运行项目的 run.py，并从服务地址打开网页。";
  }
  updateControls();
}

function switchTab(mode) {
  if (busy || recording || requestingMic || decoding) return;
  ["upload", "record"].forEach(name => {
    const active = mode === name;
    $(name + "Tab").classList.toggle("active", active);
    $(name + "Tab").setAttribute("aria-selected", String(active));
    $(name + "Panel").hidden = !active;
  });
}

function drawWaveform(values = waveformValues) {
  const canvas = $("waveform"), rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(60 * ratio);
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  const w = rect.width, h = 60;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#e1e9df"; ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  if (!values.length) return;
  const count = Math.min(values.length, Math.floor(w / 4)), peak = Math.max(...values, .01);
  ctx.fillStyle = recording ? "#bb8270" : "#82ad91";
  for (let i = 0; i < count; i++) {
    const v = values[Math.floor(i * values.length / count)];
    const barHeight = Math.max(2, v / peak * (h - 10));
    ctx.fillRect(i * w / count, (h - barHeight) / 2, Math.max(1.5, w / count - 2), barHeight);
  }
}

function clearResult() {
  currentResult = null;
  $("emptyResult").hidden = false; $("resultContent").hidden = true;
  $("loadingResult").hidden = true; $("detailSection").hidden = true;
  $("exportBtn").disabled = true;
}

function removeAudio() {
  selectionToken++;
  selectedFile = null; waveformValues = [];
  $("audioPlayer").pause(); $("audioPlayer").removeAttribute("src"); $("audioPlayer").load();
  if (audioUrl) { URL.revokeObjectURL(audioUrl); audioUrl = null; }
  $("audioFile").value = ""; $("audioPreview").hidden = true;
  clearResult(); updateControls();
}

async function chooseFile(file) {
  if (!file || busy || recording || requestingMic || decoding) return;
  if (!/\.(wav|mp3|flac|ogg)$/i.test(file.name)) { toast("请选择 WAV、MP3、FLAC 或 OGG 音频文件。"); return; }
  if (file.size > 15 * 1024 * 1024) { toast("文件超过 15 MB，请压缩或截取后重试。"); return; }
  if (!file.size) { toast("音频文件为空，请重新选择。"); return; }
  removeAudio();
  const token = ++selectionToken;
  selectedFile = file; audioUrl = URL.createObjectURL(file);
  $("audioPlayer").src = audioUrl;
  $("audioPreview").hidden = false;
  $("fileName").textContent = file.name;
  $("fileMeta").textContent = formatSize(file.size) + " · 准备分析";
  decoding = true; updateControls();
  let context;
  try {
    context = new AudioContext();
    const buffer = await context.decodeAudioData(await file.arrayBuffer());
    if (token !== selectionToken) return;
    if (buffer.duration > 60 || buffer.duration < .5) {
      removeAudio(); toast("请选择 0.5–60 秒的音频，建议 3–10 秒。"); return;
    }
    $("fileMeta").textContent = `${formatTime(buffer.duration)} · ${formatSize(file.size)} · 可试听`;
    const data = buffer.getChannelData(0), step = Math.max(1, Math.floor(data.length / 180));
    waveformValues = [];
    for (let i = 0; i < data.length; i += step) {
      let peak = 0;
      for (let j = i; j < Math.min(i + step, data.length); j++) peak = Math.max(peak, Math.abs(data[j]));
      waveformValues.push(peak);
    }
    drawWaveform();
  } catch {
    if (token === selectionToken) $("fileMeta").textContent = `${formatSize(file.size)} · 浏览器无法预览，将由服务端检查`;
  } finally {
    if (context) await context.close().catch(() => {});
    decoding = false; updateControls();
  }
}

function formatSize(bytes) { return bytes >= 1024 * 1024 ? (bytes / 1024 / 1024).toFixed(1) + " MB" : (bytes / 1024).toFixed(0) + " KB"; }
function formatTime(seconds) { return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`; }

async function toggleRecording() {
  if (recording) { stopRecording(); return; }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { toast("浏览器不支持录音，请使用 localhost / HTTPS 和现代浏览器，或上传音频。"); return; }
  requestingMic = true; updateControls();
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false}});
    removeAudio();
    audioContext = new AudioContext();
    analyser = audioContext.createAnalyser(); analyser.fftSize = 256;
    audioContext.createMediaStreamSource(stream).connect(analyser);
    const mimeType = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"].find(type => MediaRecorder.isTypeSupported(type));
    recorder = new MediaRecorder(stream, mimeType ? {mimeType} : undefined);
    const chunks = [];
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onerror = () => { toast("录音发生错误，请重新录制。"); stopRecording(); };
    recorder.onstop = async () => {
      const blob = new Blob(chunks, {type: recorder.mimeType});
      cleanupRecording();
      decoding = true; updateControls();
      try {
        const buffer = await audioContext.decodeAudioData(await blob.arrayBuffer());
        if (buffer.duration < .5) throw new Error("录音过短，请至少录制 0.5 秒。");
        const duration = Math.min(60, buffer.duration);
        const offline = new OfflineAudioContext(1, Math.floor(duration * 16000), 16000);
        const source = offline.createBufferSource(); source.buffer = buffer; source.connect(offline.destination); source.start();
        const resampled = await offline.startRendering();
        const wav = encodeWav(resampled.getChannelData(0), 16000);
        const file = new File([wav], `录音_${new Date().toISOString().replace(/[:.]/g, "-")}.wav`, {type: "audio/wav"});
        decoding = false;
        await chooseFile(file);
      } catch (error) { toast(error.message || "录音转换失败，请重试或上传 WAV 文件。"); }
      finally { if (audioContext) await audioContext.close().catch(() => {}); audioContext = null; decoding = false; updateControls(); }
    };
    recording = true; requestingMic = false; recordStarted = Date.now();
    $("recordBtn").classList.add("recording"); $("recordBtn").setAttribute("aria-label", "停止录音");
    $("recordLabel").textContent = "正在录音，点击停止";
    $("audioPreview").hidden = false; $("fileName").textContent = "正在录制你的声音…"; $("fileMeta").textContent = "完成后自动转换为 16 kHz WAV";
    $("audioPlayer").hidden = true;
    recorder.start(250);
    recordInterval = setInterval(() => {
      const seconds = (Date.now() - recordStarted) / 1000;
      $("recordTimer").textContent = `${formatTime(seconds)} / 01:00`;
      if (seconds >= 60) stopRecording();
    }, 200);
    const samples = new Uint8Array(analyser.fftSize);
    const animate = () => {
      if (!recording) return;
      analyser.getByteTimeDomainData(samples);
      drawWaveform(Array.from(samples, n => Math.abs(n - 128) / 128));
      animationId = requestAnimationFrame(animate);
    };
    animate(); updateControls();
  } catch (error) {
    cleanupRecording();
    if (audioContext) await audioContext.close().catch(() => {});
    audioContext = null;
    toast(error.name === "NotAllowedError" ? "麦克风访问未获允许，请在浏览器设置中开放权限。" : "无法启动麦克风，请检查设备和浏览器权限。");
  } finally { requestingMic = false; updateControls(); }
}

function stopRecording() {
  if (recorder?.state === "recording") {
    decoding = true;
    recorder.stop();
    recording = false; clearInterval(recordInterval); cancelAnimationFrame(animationId);
    updateControls();
  }
}

function cleanupRecording() {
  recording = false; clearInterval(recordInterval); cancelAnimationFrame(animationId);
  stream?.getTracks().forEach(track => track.stop()); stream = null;
  $("recordBtn").classList.remove("recording"); $("recordBtn").setAttribute("aria-label", "开始录音");
  $("recordLabel").textContent = "点击麦克风，开始表达"; $("recordTimer").textContent = "00:00 / 01:00";
  $("audioPlayer").hidden = false;
  if (!selectedFile) $("audioPreview").hidden = true;
}

function encodeWav(samples, sampleRate) {
  const result = new ArrayBuffer(44 + samples.length * 2), view = new DataView(result);
  const text = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
  text(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE"); text(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, samples.length * 2, true);
  samples.forEach((value, i) => { const v = Math.max(-1, Math.min(1, value)); view.setInt16(44 + i * 2, v < 0 ? v * 32768 : v * 32767, true); });
  return result;
}

async function submitAudio() {
  if (!selectedFile || busy || recording || decoding || !modelReady) return;
  busy = true; updateControls();
  $("emptyResult").hidden = true; $("resultContent").hidden = true; $("detailSection").hidden = true;
  $("loadingResult").hidden = false; document.querySelector(".result-card").setAttribute("aria-busy", "true");
  const form = new FormData(); form.append("file", selectedFile, selectedFile.name);
  try {
    const response = await fetch("/predict", {method: "POST", body: form, signal: AbortSignal.timeout(180000)});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "音频分析失败，请重试。");
    const entry = {filename: selectedFile.name, timestamp: new Date().toISOString(), result};
    history.unshift(entry); history = history.slice(0, 10);
    $("historyCount").textContent = history.length;
    displayResult(entry);
  } catch (error) {
    toast(error.name === "TimeoutError" ? "分析超时，服务可能仍在处理，请稍后重试。" : error.message || "无法连接分析服务。");
    if (currentResult) displayResult(currentResult, currentResult.historical); else clearResult();
  } finally {
    busy = false; $("loadingResult").hidden = true;
    document.querySelector(".result-card").setAttribute("aria-busy", "false"); updateControls();
  }
}

function displayResult(entry, historical = false) {
  currentResult = {...entry, historical};
  const result = entry.result, emotion = EMOTIONS[result.predicted_emotion] || EMOTIONS.neutral;
  $("emptyResult").hidden = true; $("loadingResult").hidden = true; $("resultContent").hidden = false;
  $("resultSource").textContent = (historical ? "历史分析 · " : "分析文件 · ") + entry.filename;
  $("resultQualifier").textContent = result.uncertain ? "待确认的情绪倾向" : "主要情绪倾向";
  $("mainEmotion").textContent = emotion.name;
  $("mainEmotion").style.color = emotion.color;
  $("resultInterpretation").textContent = result.interpretation;
  $("confidenceValue").textContent = (result.confidence * 100).toFixed(1) + "%";
  $("confidenceRing").style.setProperty("--confidence", result.confidence * 100 + "%");
  $("confidenceRing").style.setProperty("--result-color", emotion.color);
  $("uncertainNote").hidden = !result.uncertain;
  $("probabilities").replaceChildren();
  Object.entries(EMOTIONS).forEach(([key, style]) => {
    const probability = result.all_probabilities[key] || 0;
    const row = document.createElement("div"); row.className = "probability-row";
    // All labels/colors are fixed local constants; file names never enter HTML.
    row.innerHTML = `<span class="probability-label"><i style="background:${style.color}"></i>${style.name}</span><div class="probability-track"><div class="probability-fill" style="width:${Math.max(0, Math.min(100, probability * 100))}%;background:${style.color}"></div></div><span class="probability-value">${(probability * 100).toFixed(1)}%</span>`;
    $("probabilities").append(row);
  });
  $("durationMetric").textContent = `音频时长 ${result.audio.duration.toFixed(1)} s`;
  $("timeMetric").textContent = `分析用时 ${(result.processing_ms / 1000).toFixed(2)} s`;
  $("detailSection").hidden = false; $("timeline").replaceChildren();
  (result.segments || []).forEach(segment => {
    const button = document.createElement("button"); button.className = "segment-button";
    const label = document.createTextNode(`${EMOTIONS[segment.predicted_emotion]?.name || "未知"}${segment.uncertain ? " · 待确认" : ""}`);
    const time = document.createElement("span"); time.textContent = `${formatTime(segment.start)} – ${formatTime(segment.end)} · ${(segment.confidence * 100).toFixed(0)}%`;
    button.append(label, time); button.disabled = historical;
    button.addEventListener("click", () => { $("audioPlayer").currentTime = segment.start; $("audioPlayer").play().catch(() => toast("请先点击音频播放按钮。")); });
    $("timeline").append(button);
  });
  const q = result.audio;
  $("qualityBadge").textContent = q.warnings.length ? "建议改善录音" : "基础检查通过";
  $("qualityMetrics").replaceChildren();
  [[q.rms_db + " dB", "音频平均电平"], [(q.active_ratio * 100).toFixed(0) + "%", "有效声音占比"], ["16 kHz", "模型输入采样率"]].forEach(([value, label]) => {
    const div = document.createElement("div"), strong = document.createElement("strong"), span = document.createElement("span");
    strong.textContent = value; span.textContent = label; div.append(strong, span); $("qualityMetrics").append(div);
  });
  $("qualityWarnings").replaceChildren();
  q.warnings.forEach(warning => { const p = document.createElement("p"); p.textContent = warning; $("qualityWarnings").append(p); });
  if (!historical && q.waveform?.length) { waveformValues = q.waveform; drawWaveform(); }
  updateControls();
}

function renderHistory() {
  $("historyList").replaceChildren();
  $("clearHistory").disabled = !history.length;
  if (!history.length) { const div = document.createElement("div"); div.className = "history-empty"; div.textContent = "还没有分析记录。完成一次分析后，结果会出现在这里。"; $("historyList").append(div); return; }
  history.forEach(entry => {
    const button = document.createElement("button"); button.className = "history-item";
    const info = document.createElement("div"), title = document.createElement("strong"), subtitle = document.createElement("small"), badge = document.createElement("span");
    title.textContent = entry.filename;
    subtitle.textContent = `${new Date(entry.timestamp).toLocaleString("zh-CN")} · ${entry.result.audio.duration.toFixed(1)} 秒`;
    info.append(title, subtitle); badge.className = "history-emotion";
    badge.textContent = `${EMOTIONS[entry.result.predicted_emotion]?.name || "未知"} · ${(entry.result.confidence * 100).toFixed(1)}%`;
    button.append(info, badge);
    button.addEventListener("click", () => { if (busy || recording || decoding || requestingMic) { toast("请先完成当前录音或分析。"); return; } setView("workspace"); displayResult(entry, true); });
    $("historyList").append(button);
  });
}

function exportResult() {
  if (!currentResult) return;
  const blob = new Blob([JSON.stringify(currentResult, null, 2)], {type: "application/json;charset=utf-8"});
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = "声绪分析_" + new Date().toISOString().slice(0, 10) + ".json";
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function loadInfo() {
  try {
    const response = await fetch("/info"); if (!response.ok) throw new Error();
    const info = await response.json();
    $("evaluationNote").textContent = info.evaluation_note;
    if (info.model_description) $("modelDescription").textContent = info.model_description;
    if (info.data_scope_note) $("dataScopeNote").textContent = info.data_scope_note;
    $("externalEvaluations").replaceChildren();
    if (Object.keys(info.evaluations || {}).length) {
      const table = document.createElement("table"); table.className = "evaluation-table";
      const head = document.createElement("thead");
      const headRow = document.createElement("tr");
      ["测试范围", "原始录音", "准确率", "宏 F1"].forEach(label => {const th = document.createElement("th"); th.textContent = label; headRow.append(th);});
      head.append(headRow); table.append(head);
      const body = document.createElement("tbody");
      Object.entries(info.evaluations).forEach(([dataset, result]) => {
        const tr = document.createElement("tr");
        [dataset + (dataset === "CREMA-D" ? " · 英语五类筛选" : dataset === "RAVDESS" ? " · 英语" : " · 中文"), String(result.samples), (result.accuracy * 100).toFixed(2) + "%", (result.macro_f1 * 100).toFixed(2) + "%"].forEach(value => { const td = document.createElement("td"); td.textContent = value; tr.append(td); });
        body.append(tr);
      });
      table.append(body); $("externalEvaluations").append(table);
    }
    $("modelFacts").replaceChildren();
    [["6 类", "中文情绪类别"], [info.test_accuracy == null ? "未评估" : (info.test_accuracy * 100).toFixed(2) + "%", "CASIA 端到端准确率"], [info.training_samples ? info.training_samples.toLocaleString() : "60 秒", info.training_samples ? "原始训练录音" : "最长音频时长"], ["15 MB", "单文件大小上限"]].forEach(([value, label]) => {
      const div = document.createElement("div"), strong = document.createElement("strong"), span = document.createElement("span");
      strong.textContent = value; span.textContent = label; div.append(strong, span); $("modelFacts").append(div);
    });
  } catch { $("evaluationNote").textContent = "暂时无法读取模型评估信息，请检查服务连接。"; }
}

async function loadSamples() {
  try {
    const response = await fetch("/samples"); if (!response.ok) throw new Error();
    const samples = await response.json(); $("sampleButtons").replaceChildren();
    if (!samples.length) { $("sampleButtons").textContent = "暂无可用示例"; return; }
    samples.forEach(sample => {
      const button = document.createElement("button"); button.className = "sample-button";
      button.textContent = EMOTIONS[sample.label]?.name || sample.label;
      button.addEventListener("click", async () => {
        if (busy || recording || requestingMic || decoding) return;
        button.disabled = true;
        try {
          const response = await fetch(sample.url); if (!response.ok) throw new Error("示例加载失败");
          await chooseFile(new File([await response.blob()], `CASIA_示例_${EMOTIONS[sample.label]?.name}.wav`, {type: "audio/wav"}));
        } catch { toast("无法加载示例，请上传本地音频。"); } finally { updateControls(); }
      });
      $("sampleButtons").append(button);
    });
  } catch { $("sampleButtons").textContent = "示例暂不可用"; }
}

document.querySelectorAll(".nav-item").forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
$("uploadTab").addEventListener("click", () => switchTab("upload"));
$("recordTab").addEventListener("click", () => switchTab("record"));
document.querySelector(".tabs").addEventListener("keydown", event => {
  if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  event.preventDefault(); const mode = $("uploadTab").getAttribute("aria-selected") === "true" ? "record" : "upload";
  switchTab(mode); $(mode + "Tab").focus();
});
$("dropZone").addEventListener("click", () => $("audioFile").click());
$("audioFile").addEventListener("change", event => chooseFile(event.target.files[0]));
["dragenter", "dragover"].forEach(name => $("dropZone").addEventListener(name, event => { event.preventDefault(); if (!busy && !recording && !decoding) $("dropZone").classList.add("dragging"); }));
["dragleave", "drop"].forEach(name => $("dropZone").addEventListener(name, event => { event.preventDefault(); $("dropZone").classList.remove("dragging"); }));
$("dropZone").addEventListener("drop", event => chooseFile(event.dataTransfer.files[0]));
$("removeFile").addEventListener("click", removeAudio);
$("recordBtn").addEventListener("click", toggleRecording);
$("analyzeBtn").addEventListener("click", submitAudio);
$("exportBtn").addEventListener("click", exportResult);
$("clearHistory").addEventListener("click", () => { history = []; $("historyCount").textContent = "0"; renderHistory(); });
window.addEventListener("resize", () => drawWaveform());
window.addEventListener("pagehide", () => { stream?.getTracks().forEach(track => track.stop()); if (audioUrl) URL.revokeObjectURL(audioUrl); });
checkHealth(); loadInfo(); loadSamples();
setInterval(checkHealth, 10000);
