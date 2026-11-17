"""Check the lean deployed app against saved, public-audio reference outputs."""
import argparse
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--url",default="http://127.0.0.1:8000");args=parser.parse_args()
    expected=json.loads((ROOT/"results/runtime_golden.json").read_text())
    session=requests.Session();health=session.get(args.url+"/health",timeout=10);health.raise_for_status()
    if not health.json()["model_loaded"]:raise RuntimeError("模型尚未就绪，请稍后再运行验证")
    info=session.get(args.url+"/info",timeout=10).json()
    if info["model_signature"]!=expected["model_signature"]:raise ValueError("当前权重与参考预测不一致")
    differences=[]
    for sample in expected["samples"]:
        path=ROOT/sample["path"]
        with path.open("rb") as source:response=session.post(args.url+"/predict",files={"file":(path.name,source,"audio/wav")},timeout=120)
        response.raise_for_status();actual=response.json();reference=sample["expected"]
        if actual["predicted_id"]!=reference["prediction"]:raise AssertionError(f"Prediction changed: {path.name}")
        delta=max(abs(actual["all_probabilities"][name]-value) for name,value in reference["probabilities"].items())
        if delta>1e-5:raise AssertionError(f"Probability changed: {path.name}: {delta}")
        differences.append(delta)
    report={"passed":True,"samples":len(differences),"max_probability_difference":max(differences),"model_signature":info["model_signature"],"evaluations_present":sorted(info["evaluations"]),"training_samples":info["training_samples"]}
    (ROOT/"results/runtime_verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))


if __name__=="__main__":main()
