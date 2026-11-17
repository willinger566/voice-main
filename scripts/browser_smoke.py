"""Real browser + live backend integration test. Uses CASIA audio, never user audio."""
import argparse
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright
import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--channel", default="msedge" if os.name == "nt" else "chromium")
    args = parser.parse_args()
    capture = next((ROOT / "data/samples").glob("*_happy_*.wav"))
    folder = ROOT / "results/screenshots"; folder.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=None if args.channel == "chromium" else args.channel, headless=True,
            args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--use-file-for-fake-audio-capture=" + str(capture)])
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, permissions=["microphone"], accept_downloads=True)
        page = context.new_page(); errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle")
        assert page.locator("#analyzeBtn").is_disabled()
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path=folder / "desktop-initial.png", full_page=True)
        page.get_by_role("button", name="高兴", exact=True).click()
        page.wait_for_function('!document.getElementById("analyzeBtn").disabled')
        page.locator("#analyzeBtn").click()
        page.locator("#resultContent").wait_for(state="visible", timeout=120000)
        assert page.locator(".probability-row").count() == 6
        page.screenshot(path=folder / "desktop-result.png", full_page=True)
        with page.expect_download() as pending:
            page.locator("#exportBtn").click()
        pending.value.save_as(ROOT / "results/browser_export.json")
        assert json.loads((ROOT / "results/browser_export.json").read_text(encoding="utf-8"))["result"]["success"]
        page.locator("[data-view=history]").click()
        assert page.locator(".history-item").count() == 1
        page.locator(".history-item").click()
        assert page.locator("#resultSource").inner_text().startswith("历史")
        page.locator("[data-view=about]").click()
        model_info = requests.get(args.url + "/info", timeout=10).json()
        accuracy = model_info["test_accuracy"]
        if accuracy is not None:
            assert f"{accuracy * 100:.2f}%" in page.locator("#modelFacts").inner_text()
        evaluations = model_info.get("evaluations", {})
        assert page.locator("#externalEvaluations tbody tr").count() == len(evaluations)
        for dataset, metrics in evaluations.items():
            row = page.locator("#externalEvaluations tbody tr").filter(has_text=dataset)
            assert f"{metrics['accuracy'] * 100:.2f}%" in row.inner_text()
            if dataset == "CREMA-D": assert "英语五类筛选" in row.inner_text()
        page.screenshot(path=folder / "desktop-about.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path=folder / "mobile-about.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1100})
        page.locator("[data-view=workspace]").click()
        page.locator("#recordTab").click(); page.locator("#recordBtn").click()
        page.wait_for_function('document.getElementById("recordBtn").classList.contains("recording")')
        page.wait_for_timeout(2200); page.locator("#recordBtn").click()
        page.wait_for_function('!document.getElementById("analyzeBtn").disabled', timeout=30000)
        magic = page.evaluate('''async () => {
            const b = await selectedFile.arrayBuffer();
            return {riff:new TextDecoder().decode(b.slice(0,4)),wave:new TextDecoder().decode(b.slice(8,12)),
                    sr:new DataView(b).getUint32(24,true),channels:new DataView(b).getUint16(22,true)};
        }''')
        assert magic == {"riff": "RIFF", "wave": "WAVE", "sr": 16000, "channels": 1}
        page.locator("#analyzeBtn").click()
        page.locator("#resultContent").wait_for(state="visible", timeout=120000)
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path=folder / "mobile-result.png", full_page=True)
        assert not errors, errors
        page.locator("#removeFile").click()
        page.set_input_files("#audioFile", {"name": "invalid.txt", "mimeType": "text/plain", "buffer": b"test"})
        assert page.locator("#toast").is_visible()
        assert page.locator("#analyzeBtn").is_disabled()
        page.locator("#uploadTab").click()
        page.screenshot(path=folder / "mobile-initial.png", full_page=True)
        browser.close()
    report = {"passed": True, "model_signature": model_info.get("model_signature"), "javascript_errors": errors, "recording_format": magic,
              "checks": ["desktop/mobile layout", "sample inference", "six probabilities", "JSON export", "session history", "model facts", "real WAV encoding", "microphone inference", "invalid input"]}
    (ROOT / "results/browser_checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
