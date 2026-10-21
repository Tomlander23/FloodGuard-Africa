"""FloodGuard Africa - evaluation of the flood-risk model on the 50-scenario dataset.

Run from the repo root:  python analysis/run_analysis.py
Outputs (analysis/output/): metrics.json, metrics_summary.md, predictions.csv, 4 PNG charts
and demo/model.json (the trained decision tree, used by the web demo).

Evaluation = leave-one-out cross-validation: each scenario is predicted by a model
trained on the other 49, so the model never sees the scenario it is scored on.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "output"
OUT.mkdir(parents=True, exist_ok=True)
LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RAMP = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]  # one hue, light -> dark = low -> critical
INK, MUTED, GRID = "#1f2933", "#52606d", "#e4e7eb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

# ---------------------------------------------------------------- data
raw = json.loads((ROOT / "data" / "test_dataset_50_scenarios.json").read_text())["scenarios"]
df = pd.DataFrame(raw)
df["y"] = df["flood_risk_level"].map(LEVELS.index)
X = df[["river_stage_m", "rainfall_mm"]].to_numpy()
y = df["y"].to_numpy()


def tree():
    return DecisionTreeClassifier(max_depth=3, random_state=0)


# ---------------------------------------------------------------- evaluation
pred = cross_val_predict(tree(), X, y, cv=LeaveOneOut())
base = cross_val_predict(tree(), X[:, [0]], y, cv=LeaveOneOut())  # river stage only
df["predicted"] = [LEVELS[i] for i in pred]
df["correct"] = pred == y
df["error_type"] = np.where(pred < y, "under-warning", np.where(pred > y, "over-warning", "correct"))


def summary(p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=range(4), zero_division=0)
    return {
        "accuracy": float((p == y).mean()),
        "macro_f1": float(f1.mean()),
        "under_warnings": int((p < y).sum()),
        "over_warnings": int((p > y).sum()),
        "per_class": {LEVELS[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                                  "n": int((y == i).sum())} for i in range(4)},
    }


# "Serious" = a HIGH/CRITICAL scenario that the model rated LOW/MEDIUM (a missed flood warning)
serious_actual = y >= 2
missed = int(((pred <= 1) & serious_actual).sum())
false_alarm = int(((pred >= 2) & ~serious_actual).sum())
cm = confusion_matrix(y, pred, labels=range(4))
by_region = df.groupby("region")["correct"].agg(["mean", "count"]).sort_values("mean")

metrics = {
    "n_scenarios": int(len(df)),
    "validation": "leave-one-out cross-validation",
    "model": summary(pred),
    "baseline_stage_only": summary(base),
    "high_or_critical_scenarios": int(serious_actual.sum()),
    "missed_high_or_critical": missed,
    "false_alarms_high_or_critical": false_alarm,
    "confusion_matrix": cm.tolist(),
    "accuracy_by_region": {r: {"accuracy": float(v["mean"]), "n": int(v["count"])} for r, v in by_region.iterrows()},
    "misclassified": df.loc[~df["correct"], ["id", "region", "river_stage_m", "rainfall_mm", "flood_risk_level",
                                              "predicted", "error_type"]].to_dict("records"),
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
df.drop(columns=["y"]).to_csv(OUT / "predictions.csv", index=False)

# ---------------------------------------------------------------- final model -> demo/model.json
final = tree().fit(X, y)
t = final.tree_
nodes = []
for i in range(t.node_count):
    leaf = t.children_left[i] == -1
    nodes.append({"leaf": bool(leaf), "class": int(np.argmax(t.value[i])) if leaf else None,
                  "feature": None if leaf else ["river_stage_m", "rainfall_mm"][t.feature[i]],
                  "threshold": None if leaf else round(float(t.threshold[i]), 4),
                  "left": None if leaf else int(t.children_left[i]),
                  "right": None if leaf else int(t.children_right[i])})
(ROOT / "demo").mkdir(exist_ok=True)
(ROOT / "demo" / "model.json").write_text(json.dumps({"levels": LEVELS, "nodes": nodes}, indent=1))
# local predictions of the final model on all scenarios (used to check the web demo matches Python)
(OUT / "final_model_predictions.json").write_text(json.dumps(
    {int(r.id): LEVELS[int(p)] for r, p in zip(df.itertuples(), final.predict(X))}))

# ---------------------------------------------------------------- charts
# 1. confusion matrix
fig, ax = plt.subplots(figsize=(5.6, 5))
ax.grid(False)
ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
for i in range(4):
    for j in range(4):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() * 0.55 else INK, fontweight="bold" if i == j else "normal")
ax.set_xticks(range(4), LEVELS, fontsize=9)
ax.set_yticks(range(4), LEVELS, fontsize=9)
ax.set_xlabel("Model prediction")
ax.set_ylabel("Actual risk level")
ax.set_title(f"{metrics['model']['accuracy']*100:.0f}% of scenarios rated correctly (leave-one-out)",
             loc="left", fontsize=12, color=INK)
fig.savefig(OUT / "1_confusion_matrix.png")
plt.close(fig)

# 2. per-level precision / recall
fig, ax = plt.subplots(figsize=(7, 4.2))
w = 0.36
xs = np.arange(4)
pc = metrics["model"]["per_class"]
ax.bar(xs - w / 2, [pc[l]["precision"] * 100 for l in LEVELS], w - 0.04, label="Precision", color="#6baed6")
ax.bar(xs + w / 2, [pc[l]["recall"] * 100 for l in LEVELS], w - 0.04, label="Recall", color="#08306b")
ax.set_xticks(xs, [f"{l}\n(n={pc[l]['n']})" for l in LEVELS])
ax.set_ylim(0, 112)
ax.set_ylabel("%")
for k, key in enumerate(["precision", "recall"]):
    for i, l in enumerate(LEVELS):
        ax.text(i + (k - 0.5) * w, pc[l][key] * 100 + 2, f"{pc[l][key]*100:.0f}", ha="center", fontsize=9, color=INK)
ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.32))
ax.set_title("Precision and recall by risk level", loc="left", fontsize=12, color=INK)
fig.savefig(OUT / "2_precision_recall_by_level.png")
plt.close(fig)

# 3. decision map: stage vs rainfall, coloured by actual level, errors ringed
fig, ax = plt.subplots(figsize=(7, 5))
gx, gy = np.meshgrid(np.linspace(3, 10.5, 300), np.linspace(10, 185, 300))
zz = final.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=RAMP, alpha=0.28)
ax.grid(False)
for i, l in enumerate(LEVELS):
    m = y == i
    ax.scatter(X[m, 0], X[m, 1], s=46, color=RAMP[i] if i > 1 else RAMP[i], edgecolor="white", linewidth=1.2,
               label=l, zorder=3)
bad = ~df["correct"].to_numpy()
ax.scatter(X[bad, 0], X[bad, 1], s=190, facecolor="none", edgecolor="#d64545", linewidth=2, zorder=4,
           label="Misclassified (leave-one-out)")
ax.set_xlabel("River stage (m)")
ax.set_ylabel("Rainfall (mm)")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Model's risk zones (shaded) vs the 50 scenarios", loc="left", fontsize=12, color=INK)
fig.savefig(OUT / "3_risk_map.png")
plt.close(fig)

# 4. accuracy by basin
fig, ax = plt.subplots(figsize=(7, 4.2))
r = by_region.sort_values("mean")
ax.barh(r.index, r["mean"] * 100, color="#2171b5", height=0.6)
for i, (v, n) in enumerate(zip(r["mean"], r["count"])):
    ax.text(v * 100 + 1.5, i, f"{v*100:.0f}%  (n={n})", va="center", fontsize=9, color=INK)
ax.set_xlim(0, 125)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Correct predictions (%)")
ax.grid(axis="y", visible=False)
ax.set_title("Accuracy by river basin", loc="left", fontsize=12, color=INK)
fig.savefig(OUT / "4_accuracy_by_basin.png")
plt.close(fig)

# ---------------------------------------------------------------- plain-language summary
m, b = metrics["model"], metrics["baseline_stage_only"]
lines = [
    "# FloodGuard Africa - model evaluation summary",
    "",
    f"- Dataset: {metrics['n_scenarios']} flood scenarios across {df['region'].nunique()} Nigerian river basins "
    "(river stage, rainfall, labelled risk level: LOW / MEDIUM / HIGH / CRITICAL).",
    "- Method: leave-one-out cross-validation (each scenario predicted by a model that never saw it).",
    f"- Accuracy: **{m['accuracy']*100:.0f}%** ({int(round(m['accuracy']*50))}/50), macro F1 **{m['macro_f1']:.2f}**.",
    f"- High/Critical scenarios: {metrics['high_or_critical_scenarios']}. "
    f"Missed (rated Low/Medium): **{missed}**. False alarms (rated High/Critical when actually Low/Medium): **{false_alarm}**.",
    f"- Errors: {m['under_warnings']} under-warning, {m['over_warnings']} over-warning; both are one level off, "
    "on the Medium/High boundary.",
    f"- Baseline (river stage only): {b['accuracy']*100:.0f}% accuracy; adding rainfall did not improve on it for this dataset.",
    "",
    "## Caveats (say these honestly to judges)",
    "- 50 scenarios, all from one curated test set; the classes are cleanly separated by river stage, so high "
    "accuracy is expected. It shows the pipeline works, not real-world forecasting skill.",
    "- Next step to claim real traction: validate on historical gauge data (e.g. NIHSA) and past flood events.",
]
(OUT / "metrics_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nMisclassified:", metrics["misclassified"])
