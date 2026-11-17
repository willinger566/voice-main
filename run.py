"""Run from any working directory: python /path/to/run.py."""
from pathlib import Path
import argparse
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run("api.main:app", host=args.host, port=args.port, log_level="info")
