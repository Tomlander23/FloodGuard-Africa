# FloodGuard Africa - model evaluation summary

- Dataset: 50 flood scenarios across 7 Nigerian river basins (river stage, rainfall, labelled risk level: LOW / MEDIUM / HIGH / CRITICAL).
- Method: leave-one-out cross-validation (each scenario predicted by a model that never saw it).
- Accuracy: **96%** (48/50), macro F1 **0.96**.
- High/Critical scenarios: 27. Missed (rated Low/Medium): **1**. False alarms (rated High/Critical when actually Low/Medium): **1**.
- Errors: 1 under-warning, 1 over-warning; both are one level off, on the Medium/High boundary.
- Baseline (river stage only): 98% accuracy; adding rainfall did not improve on it for this dataset.

## Caveats (say these honestly to judges)
- 50 scenarios, all from one curated test set; the classes are cleanly separated by river stage, so high accuracy is expected. It shows the pipeline works, not real-world forecasting skill.
- Next step to claim real traction: validate on historical gauge data (e.g. NIHSA) and past flood events.
