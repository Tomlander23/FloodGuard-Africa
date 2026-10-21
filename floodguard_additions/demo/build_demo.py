"""Build demo/index.html (one self-contained file) from template.html + model + data.

Run from the repo root after analysis/run_analysis.py:  python demo/build_demo.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
demo = ROOT / "demo"
html = (demo / "template.html").read_text(encoding="utf-8")
scenarios = json.loads((ROOT / "data" / "test_dataset_50_scenarios.json").read_text())["scenarios"]
metrics = json.loads((ROOT / "analysis" / "output" / "metrics.json").read_text())
slim = {k: metrics[k] for k in ("model", "missed_high_or_critical", "high_or_critical_scenarios")}
slim["model"] = {"accuracy": metrics["model"]["accuracy"]}

html = html.replace("/*__MODEL_JS__*/", (demo / "model.js").read_text(encoding="utf-8"))
html = html.replace("/*__MODEL_JSON__*/", (demo / "model.json").read_text(encoding="utf-8"))
html = html.replace("/*__SCENARIOS__*/", json.dumps(scenarios))
html = html.replace("/*__METRICS__*/", json.dumps(slim))
(demo / "index.html").write_text(html, encoding="utf-8")
# GitHub Pages can publish only from the repo root or /docs, so write a copy there too.
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "index.html").write_text(html, encoding="utf-8")
print("wrote demo/index.html and docs/index.html", f"({len(html)//1024} KB)")
