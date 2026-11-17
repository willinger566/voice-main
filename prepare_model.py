"""Explicitly download the official backbone into this project before serving."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / "models/cache"))

if __name__ == "__main__":
    from modelscope import snapshot_download
    path = snapshot_download("iic/emotion2vec_plus_large", local_dir=str(ROOT / "models/emotion2vec_plus_large"))
    print(f"模型下载完成: {path}")
